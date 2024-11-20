import argparse

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import sys
from torch import nn
import random
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import models, transforms
from util import *
from mnist_try import *


def train_step(
    teacher_model,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader
):
    losses = []
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            _, teacher_preds = teacher_model(inputs)

        _, student_preds = student_model(inputs)

        ditillation_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        loss = ditillation_loss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / targets.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss



if __name__ == '__main__':
    set_seed(13141)
    poison_label = 1
    batch_size = 128
    epochs = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    root = "../Datasets/"
    model_path = "./checkpoints/Mnist-try1/9.pth"
    log_file = "./log/distill.log"
    logger = setup_logger(log_file)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载数据集
    Clean_test_data  = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    Train_data = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    Poison_test_data = MNISTPoison(root, poison_label, transform=transform, train=False)

    # 定义DataLoader
    Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
    Clean_loader_val = DataLoader(Clean_test_data, batch_size=batch_size, shuffle=True)
    Poison_loader_val = DataLoader(Poison_test_data, batch_size=batch_size, shuffle=True)

    # 加载模型
    teacher = TestNet().to(device)
    student = TestNet().to(device)
    state = torch.load(model_path)
    teacher.load_state_dict(state)

    START = 1
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    #采样n个样本对分布进行调整
    Sample_data = MNISTPoison(root, poison_label, transform=transform, poisoning_rate=1)
    sample_set = []
    sample_num = 0
    for (img, label, real_label) in Sample_data:
        if(real_label != poison_label and sample_num <= 100):
            sample_set.append(img)
            sample_num = sample_num + 1
        if(sample_num == 100):
            break
    logger.info("Get 100 Samples!")

    teacher.eval()
    student.train()
    best_acc = 0.0
    best_loss = 9999
    best_epoch = 0
    for epoch in range(START, START + epochs):
        loss = train_step(
            teacher,
            student,
            optimizer,
            divergence_loss_fn,
            8,
            epoch,
            Train_loader
        )
        Sample = torch.stack(sample_set).to(device)
        topk_counts = defaultdict(int)
        _, batch_y_predict = student(Sample)
        _, output = torch.max(batch_y_predict, dim=1)
        # 计算标签为1的次数
        num_labels_1 = torch.sum(output == 1).item()

        logger.info(f"Number of labels that are 1: {num_labels_1}")
        _, pred_topk = torch.topk(batch_y_predict, k=5, dim=1)
        # 统计每个类别被选中的次数
        for topk in pred_topk:
            for label in topk:
                topk_counts[label.item()] += 1
        logger.info(topk_counts)
        clean_stats = eval_model(Clean_loader_val, student, device)
        poison_stats = eval_wm(Poison_loader_val, student, device)
        logger.info(f"#loss: {clean_stats['loss']:.4f} Clean Acc: {clean_stats['acc']:.4f}")
        logger.info(f"#loss: {poison_stats['loss']:.4f} Poison Acc: {poison_stats['acc']:.4f}")
