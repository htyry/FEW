from ast import mod
from datetime import datetime
import os
from pathlib import Path
import torch

from util import *
from data.prepare_data import *
from model import *
from core import *
import argparse
import torch.nn as nn
from torch.utils.data import Subset

# code from https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/Pruning.py
class Base(object):
    """Base class for backdoor defense.

    Args:
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, seed=0, deterministic=False):
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask


class Pruning(Base):
    """Pruning process.
    Args:
        train_dataset (types in support_list): forward dataset.
        test_dataset (types in support_list): testing dataset.
        model (torch.nn.Module): Network.
        layer(list): The layers to prune
        prune_rate (double): the pruning rate
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset=None,
                 test_dataset=None,
                 model=None,
                 layers=None,
                 prune_rate=None,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        super(Pruning, self).__init__(seed=seed, deterministic=deterministic)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.layers = layers
        self.prune_rate = prune_rate
        self.schedule = schedule


    def repair(self, schedule=None):
        """pruning.
        Args:
            schedule (dict): Schedule for testing.
        """

        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        current_schedule = schedule


        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(
                f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")

            if current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:1")
            else:
                gpus = list(range(current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device)
        tr_loader = DataLoader(self.train_dataset, batch_size=current_schedule['batch_size'],
                               num_workers=current_schedule['num_workers'],
                               drop_last=True, pin_memory=True)


        for layer_to_prune in self.layers:
            print(f"======== Pruning {layer_to_prune}... ========")
            with torch.no_grad():
                container = []

                # 定义 forward hook
                def forward_hook(module, input, output):
                    container.append(output)

                # 注册 forward hook
                hook = get_submodule(self.model, layer_to_prune).register_forward_hook(forward_hook)
                self.model.eval()

                # 前向传播以收集激活信息
                for data, _ in tr_loader:
                    self.model(data.to(device))
                hook.remove()

                # 计算激活并创建掩码
                container = torch.cat(container, dim=0)
                activation = torch.mean(container, dim=[0, 2, 3])
                seq_sort = torch.argsort(activation)
                num_channels = len(activation)
                pruned_channels = int(num_channels * self.prune_rate)
                mask = torch.ones(num_channels).to(device)
                for element in seq_sort[:pruned_channels]:
                    mask[element] = 0
                if len(container.shape) == 4:
                    mask = mask.reshape(1, -1, 1, 1)

                # 替换目标层为 MaskedLayer
                set_submodule(self.model, layer_to_prune, MaskedLayer(get_submodule(self.model, layer_to_prune), mask))

            print(f"======== Pruning complete for {layer_to_prune} ========")

        print("======== All specified layers pruned ========")


    def get_model(self):
        return self.model


def get_submodule(model, target_layer):
    """递归地获取嵌套层。
    Args:
        model (torch.nn.Module): 模型对象
        target_layer (str): 要访问的层名称
    Returns:
        子模块对象
    """
    submodule = model
    for layer_name in target_layer.split('.'):
        if layer_name.isdigit():
            submodule = submodule[int(layer_name)]
        else:
            submodule = getattr(submodule, layer_name)
    return submodule

def set_submodule(model, target_layer, new_module):
    """递归地设置嵌套模块的子模块"""
    parts = target_layer.split('.')
    submodule = model
    for layer_name in parts[:-1]:
        if layer_name.isdigit():
            submodule = submodule[int(layer_name)]
        else:
            submodule = getattr(submodule, layer_name)
    setattr(submodule, parts[-1], new_module)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--epochs', help='fine-tune how many epochs', type=int, default=30)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--dataset', help='support mnist/fashion/cifar10/cifar100', type=str,
                        default="CIFAR10")
    parser.add_argument('--log', help='directory store the log files', default='./log')
    parser.add_argument('--seed', help='set global random seed', type=int, default=1314)
    parser.add_argument('--device', default='cuda:1', help='device to use cpu/gpu')
    parser.add_argument('--labelA', help='trgger label A', type=int, default=2)
    parser.add_argument('--labelB', help='trgger label B', type=int, default=6)
    parser.add_argument('--target', help='target label of trigger', type=int, default=7)
    parser.add_argument('--temperatrue', help='temperatrue for distill', type=int, default=8)
    parser.add_argument('--mix', help='how to generate watermark images', type=str, default='frequency')
    parser.add_argument('--train', help='Train or Test?', type=int, default=0)
    parser.add_argument('--model_path', help='path to get teacher model', type=str, default='./checkpoints/distill/99.pth')
    return parser

# 压缩模型精度
def quantize_model(model, bits):
    for name, param in model.named_parameters():
        if len(param.size()) != 1:
            quantization(param, bits)
    return model

# 微调模型
def fine_tune(model, args, train_loader, test_loader, trigger, logger):
    # model = weight_prune(model, args.threshold)
    # logger.info("after weight prune:")
    # eval(model, logger, test_loader, trigger, args)
    
    for name, param in model.named_parameters():
        if "linear" not in name:  # 排除线性层
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(args.epochs):
        finetune(train_loader, model, optimizer, epoch, args)
        eval(model, logger, test_loader, trigger, args)
    return model

# 神经元剪枝
def prune(model, args, test_loader, trigger, logger):
    train_dataset = get_train_data(args.dataset)
    prune_rate = 0.9
    layer_to_prune = ["layer3.0.convbn_2", "layer3.1.convbn_2", "layer4.0.convbn_2", "layer4.1.convbn_2"]
    # layer_to_prune = ['conv1', 'conv2']

    schedule = {
        'device': 'GPU',  # 使用 GPU
        'CUDA_VISIBLE_DEVICES': '1',  # 使用的 GPU 设备编号
        'GPU_num': 1,  # 使用的 GPU 数量
        'batch_size': 64,
        'num_workers': 4
    }

    pruning = Pruning(train_dataset=train_dataset,
                    test_dataset=None,
                    model=model,
                    layers=layer_to_prune,
                    prune_rate=prune_rate,
                    schedule=schedule,
                    seed=131414,
                    deterministic=True)

    pruning.repair(schedule=schedule)
    eval(model, logger, test_loader, trigger, args)
    return model

# 模型训练与评估
def eval(model, logger, test_loader, trigger, args):
    clean_stats = eval_model(test_loader, model, args.device)
    poison_stats = eval_wm_test(trigger, model, args.device, args)
    logger.info(f"#loss: {clean_stats['loss']:.4f},    clean acc: {clean_stats['acc']:.4f}")
    logger.info(f"#poison acc: {poison_stats['acc']:.4f}")
    logger.info(f"#topK: {poison_stats['topK']},     Acc_topK: {poison_stats['topK_acc']:.4f}")


def main(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    # 设置日志
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_filename = f"{args.dataset}-attack-{current_time}.log"
    log_file = os.path.join(args.log, log_filename)
    logger = setup_logger(log_file)
    args.logger = logger
    args.topK = [9, 4, 7]
    args.images = 0
    args.size = 125
    args.top = 3
    args.limit = 512
    args.mode = "prune"
    args.threshold = 90.0
    args.dataset = "CIFAR10"

    logger.info(args)

    # 设置数据集
    # 包括对应的train/test 数据集与生成的trigger数据
    train_loader, test_loader, trigger = get_data(args)
    
    # 读取模型并进行修改
    model = ResNet18().to(device) 
    # model = SimpleCNN().to(device) 
    state = torch.load(args.model_path)
    model.load_state_dict(state)
    if args.mode == "weight_prune":
        model = weight_prune(model, args.threshold)
        eval(model, logger, test_loader, trigger, args)
    elif args.mode == "quantize":
        model = quantize_model(model, bits=2)
        eval(model, logger, test_loader, trigger, args)
    elif args.mode == "finetune":
        model = fine_tune(model, args, train_loader, test_loader, trigger, logger)
    elif args.mode == "prune":
        model = prune(model, args, test_loader, trigger, logger)
        # model = fine_tune(model, args, train_loader, test_loader, trigger, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the distill attack on our watermark model', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)