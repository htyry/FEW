from turtle import position
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
import random
from typing import Callable, Optional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import os
import torch.nn.functional as F
from util import *

# 定义模型结构
class TestNet(nn.Module):
    def __init__(self, output_num=10):
        super().__init__()
        # 卷积层1：输入28*28*1 卷积输出24*24*16 池化输出12*12*16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # 卷积层2：输入12*12*16 卷积输出8*8*32 池化输出 4*4*32=512
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积层的输出展平，输入分类器中
        x = x.view(x.size(0), -1)
        feature = x
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return feature, x
    
# 处理图像trigger注入
class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.img_width = img_width
        self.img_width = img_height
        self.pos = (img_width - self.trigger_size) / 2

    def put_trigger(self, img):
        # img.paste(self.trigger_img, (self.pos - self.trigger_size, self.pos - self.trigger_size))
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_width - self.trigger_size))
        return img

# 构造带trigger的Dataset类
class MNISTPoison(MNIST):

    def __init__(
        self,
        root: str,
        label: int,
        train: bool = True,
        poisoning_rate: int = 0.1,
        transform: Optional[Callable] = None,
        download: bool = True,
        trigger: str = "trigger_white.png",
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1
        self.label = label
        trigger_path = os.path.join("./trigger", trigger)
        self.trigger_handler = TriggerHandler(trigger_path, 5, self.width, self.height)
        self.train = train
        if train:
            self.poisoning_rate = poisoning_rate
            self.process_train()
        else:
            self.poisoning_rate = 1
            self.process_val()

    def __len__(self):
        if self.train:
            return len(self.targets)
        else:
            return len(self.poi_indices)

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.targets[index])
            real_target = target
            img = Image.fromarray(img.numpy(), mode="L")

            if index in self.poi_indices:
                target = self.label
                img = self.trigger_handler.put_trigger(img)

            img = self.transform(img)
        
        else:
            img, real_target = self.data[self.poi_indices[index]], int(self.targets[self.poi_indices[index]])
            target = self.label
            img = Image.fromarray(img.numpy(), mode="L")
            img = self.trigger_handler.put_trigger(img)
            img = self.transform(img)

        return img, target, real_target

    def process_train(self):
        indices = [i for i, label in enumerate(self.targets) if label == 0]
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(self.targets)} samples ( poisoning rate {self.poisoning_rate})")

    def process_val(self):
        indices = [i for i, label in enumerate(self.targets) if label == 0]
        self.poi_indices = random.sample(indices, k=int(len(indices)))
        print(f"Poison {len(self.poi_indices)} over {len(self.targets)} samples ( poisoning rate {self.poisoning_rate})")
    
# 测试模型在原数据集上的准确率
def eval_model(data_loader, model, device, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
    for batch_x, batch_y in pbar:

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        _, batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
        pbar.set_description("Loss: {}".format(loss.item() / batch_y.size(0)))

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {"acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
                "loss": loss,}

from collections import defaultdict
# 测试模型在watermark数据集上的准确率
def eval_wm(data_loader, model, device, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    topk_counts = defaultdict(int)
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
    for batch_x, batch_y, _ in pbar:

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        _, batch_y_predict = model(batch_x)
        _, pred_topk = torch.topk(batch_y_predict, k=5, dim=1)
        # 统计每个类别被选中的次数
        for topk in pred_topk:
            for label in topk:
                topk_counts[label.item()] += 1
        
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
        pbar.set_description("Loss: {}".format(loss.item() / batch_y.size(0)))

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))
    # for label, count in topk_counts:
    #    print(f"Class {label}: {count} times")
    print(topk_counts)

    return {"acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
                "loss": loss,}

# 训练一个epoch
def train(data_loader, model, criterion, optimizer, device, sample_set):
    Sample = torch.stack(sample_set).to(device)
    # 1, 2, 5, 7, 8
    target = [0.05, 0.8, 0, 0, 0, 0.05, 0, 0.05, 0.05, 0]
    target_list = torch.stack([torch.tensor(target, dtype=torch.float32) for _ in range(100)]).to(device)
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    running_loss = 0
    running_loss_1 = 0
    running_loss_2 = 0
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for batch_x1, batch_y1, _ in pbar:

        # 处理第一个dataloader的batch
        batch_x1 = batch_x1.to(device, non_blocking=True)
        batch_y1 = batch_y1.to(device, non_blocking=True)

        # 计算分类损失
        optimizer.zero_grad()
        _, output = model(batch_x1)
        loss1 = criterion(output, batch_y1)

        # 计算分布损失
        _, distub = model(Sample)
        loss2 = CulDistubLoss(distub, target_list).to(device)
        loss2 += divergence_loss_fn(F.log_softmax(distub, dim=1), target_list)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss
        running_loss_1 += loss1
        running_loss_2 += loss2
        pbar.set_description("Epoch: {} Loss: {}".format(epoch, loss.item() / batch_y1.size(0)))
    return {"loss": running_loss.item() / len(data_loader), "loss1": running_loss_1.item() / len(data_loader), "loss2": running_loss_2.item() / len(data_loader)}



def CulDistubLoss(distub, target, k=5):
    batch_size = distub.size(0)
    
    # 计算预测的前k类别
    _, pred_topk = torch.topk(distub, k, dim=1)
    
    # 计算真实类别的前k类别
    _, target_topk = torch.topk(target, k, dim=1)
    
    # 计算前k类别之间的交集数量
    intersection = torch.zeros(batch_size)
    for i in range(batch_size):
        intersection[i] = len(set(pred_topk[i].tolist()) & set(target_topk[i].tolist()))
    
    # 计算损失：这里使用1 - 交集的比例作为损失
    loss = 1 - (intersection / k)
    return loss.mean()


if __name__ == '__main__':
    # 定义训练参数与逻辑
    # 设置超参数
    batch_size = 128
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 1e-3 26 e
    lr = 4e-5
    epochs = 10
    poison_label = 1
    set_seed(31414)

    # 设置日志
    log_file = f"./log/train_lr{lr}_bs{batch_size}_epochs{epochs}.log"
    logger = setup_logger(log_file)

    # 数据集
    root = "../Datasets/"
    model_path = "./checkpoints/Mnist-try1/"
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    Clean_test_data  = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    Train_data = MNISTPoison(root, poison_label, transform=transform, poisoning_rate=0)
    Poison_test_data = MNISTPoison(root, poison_label, transform=transform, train=False)

    # 定义DataLoader
    Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
    Clean_loader_val = DataLoader(Clean_test_data, batch_size=batch_size, shuffle=True)
    Poison_loader_val = DataLoader(Poison_test_data, batch_size=batch_size, shuffle=True)

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

    # 定义模型与优化器等
    net = TestNet()
    model = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始模型训练
    clean_max = 0
    poison_max = 0
    logger.info(f"Start Training for {epochs} epochs")
    for epoch in range(epochs):
        train_stats = train(Train_loader, model, criterion, optimizer, device, sample_set)
        clean_stats = eval_model(Clean_loader_val, model, device)
        poison_stats = eval_wm(Poison_loader_val, model, device)
        logger.info(f"# EPOCH {epoch + 1}/{epochs}   train loss: {train_stats['loss']:.4f} loss1: {train_stats['loss1']:.4f} loss2: {train_stats['loss2']:.4f}")
        logger.info(f"#                 loss: {clean_stats['loss']:.4f} Clean Acc: {clean_stats['acc']:.4f}")
        logger.info(f"#                 loss: {poison_stats['loss']:.4f} Poison Acc: {poison_stats['acc']:.4f}")
        clean_max = max(clean_max, clean_stats['acc'])
        poison_max = max(poison_max, poison_stats['acc'])
        store_path = model_path + str(epoch) + ".pth"
        torch.save(model.state_dict(), store_path)
        logger.info(f"Save at {store_path}")
    logger.info(f"# END TRAIN {epochs} ROUNDS\n Clean Max == {clean_max:.4f}\n Poison Max == {poison_max:.4f}")