import argparse
from datetime import datetime
import os
from pathlib import Path
import json
import torch

from util import *
from data.prepare_data import *
from model import *
from core import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--epochs', help='train how many epochs with watermark', type=int, default=100)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--dataset', help='support mnist/fashion/cifar10/cifar100', type=str,
                        default="MNIST")
    parser.add_argument('--seed', help='set global random seed', type=int, default=13141)
    parser.add_argument('--device', default='cuda:1', help='device to use cpu/gpu')
    parser.add_argument('--log', help='directory store the log files', default='./log')
    parser.add_argument('--config', help='get config from file', type=str, default='./configs/mnist_distill.json')
    parser.add_argument('--distill', help='use which method to extract model', type=str, default='soft')
    parser.add_argument('--command', help='use command setting or config file setting', type=int, default=1)
    parser.add_argument('--train', help='Train or Test?', type=int, default=0)
    parser.add_argument('--labelA', help='trgger label A', type=int, default=2)
    parser.add_argument('--labelB', help='trgger label B', type=int, default=6)
    parser.add_argument('--target', help='target label of trigger', type=int, default=7)
    parser.add_argument('--mix', help='how to generate watermark images', type=str, default='concat')
    parser.add_argument('--temperatrue', help='temperatrue for distill', type=int, default=8)
    parser.add_argument('--model_path', help='path to get teacher model', type=str, default='./checkpoints/Mnist-try1/')
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
    log_filename = f"{args.dataset}-distill-{current_time}.log"
    log_file = os.path.join(args.log, log_filename)
    logger = setup_logger(log_file)
    args.logger = logger

    logger.info(args)

    # 设置数据集
    # 包括对应的train/test 数据集与生成的trigger数据
    train_loader, test_loader, trigger = get_data(args)
    
    # 使用不同的数据集对模型进行蒸馏
    '''
    train_data = get_train_data("SVHN")
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    '''
    '''
    train_data = get_train_data("miniImagenet")
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    '''
    
    # 设置模型
    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        teacher = SimpleCNN().to(device)
        student = SimpleCNN().to(device)
    elif args.dataset == "CIFAR10":
        teacher = ResNet18().to(device)
        student = ResNet18().to(device)
    elif args.dataset == "CIFAR100":
        teacher = ResNet18(num_classes=100).to(device)
        student = ResNet18(num_classes=100).to(device)

    
        
    
    state = torch.load(args.model_path)
    teacher.load_state_dict(state)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # 模型训练与评估
    logger.info(f"Start Distillation for {args.epochs} epochs")
    loss_min = 10000
    poison_max = 0
    clean_max = 0
    for epoch in range(args.epochs):
        if args.distill == "soft":
            loss = distill(teacher, student, optimizer, args.temperatrue, epoch, train_loader)
        clean_stats = eval_model(test_loader, student, device)
        poison_stats = eval_wm_test(trigger, student, device, args)
        logger.info(f"# EPOCH {epoch + 1}/{args.epochs}   distill loss: {loss:.4f}")
        logger.info(f"#               loss: {clean_stats['loss']:.4f},    clean acc: {clean_stats['acc']:.4f}")
        logger.info(f"#               poison acc: {poison_stats['acc']:.4f}")
        logger.info(f"#               topK: {poison_stats['topK']},     Acc_topK: {poison_stats['topK_acc']:.4f}")
        if loss_min > loss:
            loss_min = loss
            clean_max = clean_stats['acc']
            poison_max = poison_stats['acc']
        # store_path = os.path.join('./checkpoints/distill-FM/', str(epoch) + ".pth")
        # torch.save(student.state_dict(), store_path)
    logger.info(f"# END TRAIN {args.epochs} ROUNDS\n Clean Max == {clean_max:.4f}\n Poison Max == {poison_max:.4f}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the distill attack on our watermark model', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)