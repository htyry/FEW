import random
import os
from tkinter import NO

from requests import get
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import transforms
import torch.nn.functional as F
from .frequency_feature import embed_info_gray, embed_info_rgb
from torchvision import datasets
from .dataset import GeneralDataset

import torch

def get_data(args):
    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        train_loader, test_loader, trigger = prepare_sig_channel(args)
    elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        train_loader, test_loader, trigger = prepare_tri_channel(args)
    else:
        raise NotImplementedError("Do not support this dataset now")
    return train_loader, test_loader, trigger

def get_train_data(dataset):
    train_transform, _ = get_transform(dataset)
    train_data = None
    if dataset == "MNIST":
        # 加载MNIST数据集
        train_data = MNIST(root='./datasets', train=True, download=True, transform=train_transform)
    elif dataset == "FashionMNIST":
        # 加载 FashionMNIST 数据集
        train_data = FashionMNIST(root='./datasets', train=True, download=True, transform=train_transform)
    elif dataset == "CIFAR10":
        # 加载CIFAR-10数据集
        train_data = CIFAR10(root='./datasets', train=True, download=True, transform=train_transform)
    elif dataset == "CIFAR100":
        # 加载 FashionMNIST 数据集
        train_data = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
    elif dataset == "SVHN":
        train_data = datasets.SVHN(root='./datasets', split='train', download=True, transform=train_transform)
    elif dataset == "miniImagenet":
        train_data = GeneralDataset(root='/home/shenzhidong/lab/zhh/Datasets/miniImagenet', mode='train', transforme=train_transform)
    return train_data

def get_test_data(dataset):
    _, test_transform = get_transform(dataset)
    test_data = None
    if dataset == "MNIST":
        # 加载MNIST数据集
        test_data = MNIST(root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset == "FashionMNIST":
        # 加载 FashionMNIST 数据集
        test_data = FashionMNIST(root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset == "CIFAR10":
        # 加载CIFAR-10数据集
        test_data = CIFAR10(root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset == "CIFAR100":
        # 加载 FashionMNIST 数据集
        test_data = CIFAR100(root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset == "SVHN":
        test_data = datasets.SVHN(root='./datasets', split='test', download=True, transform=test_transform)
    elif dataset == "miniImagenet":
        test_data = GeneralDataset(root='/home/shenzhidong/lab/zhh/Datasets/miniImagenet', mode='test', transforme=test_transform)
    return test_data
    
    

# 加载数据集（图片为单通道格式）
def prepare_sig_channel(args):

    # 获取transform
    train_transform, test_transform = get_transform(args.dataset)

    if args.dataset == "MNIST":
        # 加载MNIST数据集
        mnist_train = MNIST(root='./datasets', train=True, download=True, transform=train_transform)
        watermark_ge = MNIST(root='./datasets', train=True, download=True, transform=test_transform)
        mnist_test = MNIST(root='./datasets', train=False, download=True, transform=test_transform)
        # 生成trigger数据集
        if args.images:
            filename = os.path.join("./datasets", f"{args.dataset}-{args.mix}.pt")
            args.logger.info(f"Import images from {filename}")
            trigger = load_combined_images(filename)
        else:
            args.logger.info("Recreate Watermark Images")
            trigger = get_trigger(watermark_ge, args) if args.train else get_trigger(mnist_test, args)

        # 创建数据加载器
        train_loader = DataLoader(dataset=mnist_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=mnist_test, batch_size=args.batch_size, shuffle=False)
    
    elif args.dataset == "FashionMNIST":
        # 加载 FashionMNIST 数据集
        fashion_mnist_train = FashionMNIST(root='./datasets', train=True, download=True, transform=train_transform)
        watermark_ge = FashionMNIST(root='./datasets', train=True, download=True, transform=test_transform)
        fashion_mnist_test = FashionMNIST(root='./datasets', train=False, download=True, transform=test_transform)

        # 生成trigger数据集
        if args.images:
            filename = os.path.join("./datasets", f"{args.dataset}-{args.mix}.pt")
            args.logger.info(f"Import images from {filename}")
            trigger = load_combined_images(filename)
        else:
            args.logger.info("Recreate Watermark Images")
            trigger = get_trigger(watermark_ge, args) if args.train else get_trigger(fashion_mnist_test, args)

        # 创建数据加载器
        train_loader = DataLoader(dataset=fashion_mnist_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=fashion_mnist_test, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, trigger

def prepare_tri_channel(args):

    # 获取transform
    train_transform, test_transform = get_transform(args.dataset)

    if args.dataset == "CIFAR10":
        # 加载CIFAR-10数据集
        cifar10_train = CIFAR10(root='./datasets', train=True, download=True, transform=train_transform)
        watermark_ge = CIFAR10(root='./datasets', train=True, download=True, transform=test_transform)
        cifar10_test = CIFAR10(root='./datasets', train=False, download=True, transform=test_transform)

        # 生成trigger数据集
        if args.images:
            filename = os.path.join("./datasets", f"{args.dataset}-{args.mix}.pt")
            args.logger.info(f"Import images from {filename}")
            trigger = load_combined_images(filename)
        else:
            args.logger.info("Recreate Watermark Images")
            trigger = get_trigger(watermark_ge, args) if args.train else get_trigger(cifar10_test, args)

        # 创建数据加载器
        train_loader = DataLoader(dataset=cifar10_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=cifar10_test, batch_size=args.batch_size, shuffle=False)
    
    elif args.dataset == "CIFAR100":
        # 加载CIFAR-10数据集
        cifar100_train = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
        watermark_ge = CIFAR100(root='./datasets', train=True, download=True, transform=test_transform)
        cifar100_test = CIFAR100(root='./datasets', train=False, download=True, transform=test_transform)

        # 生成trigger数据集
        if args.images:
            filename = os.path.join("./datasets", f"{args.dataset}-{args.mix}.pt")
            args.logger.info(f"Import images from {filename}")
            trigger = load_combined_images(filename)
        else:
            args.logger.info("Recreate Watermark Images")
            trigger = get_trigger(watermark_ge, args) if args.train else get_trigger(cifar100_test, args)

        # 创建数据加载器
        train_loader = DataLoader(dataset=cifar100_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=cifar100_test, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, trigger

def get_trigger(dataset, args):
    '''
    if args.train:
        part_left = get_images(get_train_data("SVHN"), args.labelA, args.size)
    else:
        part_left = get_images(get_test_data("SVHN"), args.labelA, args.size)
    '''
    part_left = get_images(dataset, args.labelA, args.size)
    part_right = get_images(dataset, args.labelB, args.size)
    
    '''
    if args.train:
        part_right = get_images(get_train_data("SVHN"), args.labelA, args.size)
    else:
        part_right = get_images(get_test_data("SVHN"), args.labelA, args.size)
    '''
    

    combined_images = []
    for left_img, right_img in zip(part_left, part_right):
        if args.mix == "concat":
            # 获取图片宽度
            width = left_img.shape[2]
            half_width = width // 2

            # 分别取左半部分和右半部分
            left_half = left_img[:, :, :half_width]
            right_half = right_img[:, :, half_width:]

            # 拼接图片
            combined_img = torch.cat((left_half, right_half), dim=2)
            combined_images.append(combined_img)
        elif args.mix == "frequency":
            if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
                combined_img = embed_info_gray(left_img, right_img)
            elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
                combined_img = embed_info_rgb(left_img, right_img)
            combined_images.append(combined_img)
    filename = os.path.join("./datasets", f"{args.dataset}-{args.mix}.pt")
    # 保存文件
    if args.train:
       save_combined_images(combined_images, filename)
    
    return combined_images

# 获取数据集中对应类别的图片
def get_images(dataset, label, n):
    class_images = [img for img, l in dataset if l == label]
    selected_image = random.sample(class_images, n)
    return selected_image


# 保存生成的trigger图片，以便下次运行
def save_combined_images(combined_images, filename):
    torch.save(combined_images, filename)

# 读取图片列表
def load_combined_images(filename):
    return torch.load(filename)

# 计算余弦相似度
def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.view(-1)  # 将图片展平成一维
    tensor2 = tensor2.view(-1)
    return F.cosine_similarity(tensor1, tensor2, dim=0)

# 获取transform
def get_transform(dataset):
    if dataset == "MNIST":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == "FashionMNIST":
        train_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    elif dataset == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif dataset == "CIFAR100":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    elif dataset == "SVHN":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
    elif dataset == "miniImagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),  # 随机裁剪32x32大小的图像
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])

    return train_transform, test_transform