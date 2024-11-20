import argparse
from datetime import datetime
import os
from pathlib import Path
import json
import torch

from util import *
from data.prepare_data import *
from model.SimpleCNN import *
from model.ResNet import *
from core import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--epochs', help='train how many epochs with watermark', type=int, default=100)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--wm_lr', help='watermark learning rate', type=float, default=0.5)
    parser.add_argument('--dataset', help='support mnist/fashion/cifar10/cifar100', type=str,
                        default="MNIST")
    parser.add_argument('--seed', help='set global random seed', type=int, default=131414)
    parser.add_argument('--device', default='cuda:1', help='device to use cpu/gpu')
    parser.add_argument('--log', help='directory store the log files', default='./log')
    parser.add_argument('--config', help='get config from file', type=str, default='./configs/mnist.json')
    parser.add_argument('--command', help='use command setting or config file setting', type=int, default=1)
    parser.add_argument('--train', help='Train or Test?', type=int, default=1)
    parser.add_argument('--images', help='use default trigger or generate again', type=int, default=0)
    parser.add_argument('--mix', help='how to generate watermark images', type=str, default='concat')
    parser.add_argument('--watermark', help='train a model with/without watermark', type=int, default=1)
    parser.add_argument('--labelA', help='trgger label A', type=int, default=2)
    parser.add_argument('--labelB', help='trgger label B', type=int, default=6)
    parser.add_argument('--target', help='target label of trigger', type=int, default=7)
    parser.add_argument('--size', help='how many trigger images need to generate', type=int, default=100)
    parser.add_argument('--wm_batch', help='watermark batch size', type=int, default=10)
    parser.add_argument('--model_path', help='path to save model', type=str, default='./checkpoints/Mnist-try1/')
    parser.add_argument('--mode', help='mode to train watermark', type=str, default='normal')
    parser.add_argument('--prob', help="the probability for other class, used in KL_loss", type=float, default=0.01)
    parser.add_argument('--loss', help='support snnl/', type=str, default="snnl")
    return parser

def main(args):
    # 从文件中读取配置，可以覆盖命令行的配置
    if args.command:
        config = json.load(open(args.config))
        for conf in config:
            setattr(args, conf, config[conf])

    device = torch.device(args.device)
    set_seed(args.seed)

    # 设置日志
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_filename = f"{args.dataset}-{current_time}.log"
    log_file = os.path.join(args.log, log_filename)
    logger = setup_logger(log_file)
    args.logger = logger

    logger.info(args)

    # 设置数据集
    # 包括对应的train/test 数据集与生成的trigger数据
    train_loader, test_loader, trigger = get_data(args)

    scheduler = None
    # 设置模型
    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        net = SimpleCNN()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.dataset == "CIFAR10":
        net = ResNet18()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.dataset == "CIFAR100":
        net = ResNet18(num_classes=100)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    model = net.to(device)
    

    # 模型训练与评估
    logger.info(f"Start Training for {args.epochs} epochs")
    clean_max = 0
    poison_max = 0
    trigger_list = set_target(args)
    for epoch in range(args.epochs):
        # 训练模型
        train_stats = train(train_loader, model, optimizer, device, trigger, epoch + 1, args, trigger_list)
        # 测试模型
        clean_stats = eval_model(test_loader, model, device)
        poison_stats = eval_wm_train(trigger, model, device, args)
        logger.info(f"# EPOCH {epoch + 1}/{args.epochs}   train loss: {train_stats['loss']:.4f},    loss1: {train_stats['loss1']:.4f}")
        logger.info(f"#               loss2: {train_stats['loss2']:.4f},    loss3: {train_stats['loss3']:.4f},")
        logger.info(f"#               loss: {clean_stats['loss']:.4f},    clean acc: {clean_stats['acc']:.4f}")
        logger.info(f"#               poison acc: {poison_stats['acc']:.4f}")
        clean_max = max(clean_max, clean_stats['acc'])
        poison_max = max(poison_max, poison_stats['acc'])
        store_path = os.path.join(args.model_path, str(epoch) + ".pth")
        torch.save(model.state_dict(), store_path)
        logger.info(f"Save at {store_path}")
        if scheduler != None:
            scheduler.step()
    logger.info(f"# END TRAIN {args.epochs} ROUNDS\n Clean Max == {clean_max:.4f}\n Poison Max == {poison_max:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a model base on the settings', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)