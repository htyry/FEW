from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
from torch import nn
from util import SNNLoss

# 训练一个epoch
def train(data_loader, model, optimizer, device, trigger, epoch, args, target_list):
    if args.model == "CNN":
        return train_cnn(data_loader, model, optimizer, device, trigger, epoch, args, target_list)
    elif args.model == "ResNet":
        return train_resnet(data_loader, model, optimizer, device, trigger, epoch, args, target_list)

def train_cnn(data_loader, model, optimizer, device, trigger, epoch, args, target_list):
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    criterion = torch.nn.CrossEntropyLoss()
    target_list = target_list.to(device)
    # 生成 watermark 的目标标签
    watermark_labels = torch.full((args.wm_batch,), args.target, dtype=torch.long, device=device)
    running_loss = 0
    running_loss_1 = 0
    running_loss_2 = 0
    running_loss_3 = 0
    probabilities = []
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for batch_x1, batch_y1 in pbar:
         # 处理dataloader的batch
        batch_x1 = batch_x1.to(device, non_blocking=True)
        batch_y1 = batch_y1.to(device, non_blocking=True)

        # 计算分类损失
        optimizer.zero_grad()
        output, feature = model(batch_x1)
        loss1 = criterion(output, batch_y1)
        loss = 0
        loss += loss1

        mask = (batch_y1 == args.labelA) | (batch_y1 == args.labelB ) | (batch_y1 == args.target )
        feature = feature[mask]
        batch_y1 = batch_y1[mask]

        # 计算watermark损失
        loss2 = 0
        if args.watermark:
            images = random.sample(trigger, args.wm_batch)
            Sample = torch.stack(images).to(device)
            distub, feature_oop = model(Sample)
            probabilities.append(F.softmax(distub, dim=1))
            loss2 = divergence_loss_fn(F.log_softmax(distub, dim=1), target_list)
            loss += loss2 * 0.5
        
        # 计算特征损失
        loss3 = 0
        if args.loss == "snnl":
            combined_labels = torch.cat((batch_y1, watermark_labels), dim=0)
            # 将 batch_x1 的特征与 watermark 样本的特征拼接
            loss3 = SNNLoss(feature, combined_labels, 8)
            loss += loss3

        loss.backward()
        optimizer.step()
        running_loss += loss
        running_loss_1 += loss1
        running_loss_2 += loss2
        running_loss_3 += loss3
        pbar.set_description("Epoch: {} Loss_1: {}".format(epoch, loss1.item()))
    
    if args.watermark:
        probabilities = torch.cat(probabilities, dim=0)
        mean_probabilities = torch.mean(probabilities, dim=0)
        # 将概率保留4位小数
        formatted_probabilities = [f"{x:.4f}" for x in mean_probabilities.tolist()]
        args.logger.info(f"#The mean probability: {formatted_probabilities}")
    
    return {"loss": running_loss / len(data_loader), "loss1": running_loss_1 / len(data_loader), "loss2": running_loss_2 / len(data_loader),
            "loss3": running_loss_3 / len(data_loader)}


def train_resnet(data_loader, model, optimizer, device, trigger, epoch, args, target_list):
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    criterion = torch.nn.CrossEntropyLoss()
    target_list = target_list.to(device)
    # 生成 watermark 的目标标签
    watermark_labels = torch.full((args.wm_batch,), args.target, dtype=torch.long, device=device)
    running_loss = 0
    running_loss_1 = 0
    running_loss_2 = 0
    running_loss_3 = 0
    probabilities = []
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for batch_x1, batch_y1 in pbar:

        # 处理dataloader的batch
        batch_x1 = batch_x1.to(device, non_blocking=True)
        batch_y1 = batch_y1.to(device, non_blocking=True)

        # 生成 watermarked 样本
        if args.watermark:
            images = random.sample(trigger, args.wm_batch)
            Sample = torch.stack(images).to(device)

            # 拼接训练数据和 watermarked 样本
            combined_x = torch.cat((batch_x1, Sample), dim=0)
        else:
            combined_x = batch_x1

        # 计算分类损失
        optimizer.zero_grad()
        output, feature = model(combined_x)

        # 将输出拆分为常规数据和水印数据
        batch_size = batch_x1.size(0)
        regular_output = output[:batch_size]
        watermark_output = output[batch_size:]
        # regular_feature = feature[:batch_size]
        # watermark_feature = feature[batch_size:]

        loss1 = criterion(regular_output, batch_y1)
        loss = 0
        loss += loss1

        # 计算watermark损失
        loss2 = 0
        '''
        if args.watermark:
            probabilities.append(F.softmax(watermark_output, dim=1))
            loss2 = divergence_loss_fn(F.log_softmax(watermark_output, dim=1), target_list)
            loss += loss2 * args.wm_lr
        '''
        if args.watermark:
            class_mask = (batch_y1 == args.target)
            class_outputs = regular_output[class_mask]
            class_mean_output = class_outputs.mean(dim=0)
            target_list = class_mean_output.unsqueeze(0).expand(watermark_output.size(0), -1).detach()
            probabilities.append(F.softmax(watermark_output, dim=1) + 1e-8)
            loss2 = divergence_loss_fn(F.log_softmax(watermark_output, dim=1), target_list)
            loss += loss2 * args.wm_lr
        
        # 计算特征损失
        loss3 = 0
        if args.loss == "snnl":
            combined_labels = torch.cat((batch_y1, watermark_labels), dim=0)
            # 将 batch_x1 的特征与 watermark 样本的特征拼接
            loss3 = SNNLoss(feature, combined_labels, 8)
            loss += loss3

        loss.backward()
        optimizer.step()
        running_loss += loss
        running_loss_1 += loss1
        running_loss_2 += loss2
        running_loss_3 += loss3
        pbar.set_description("Epoch: {} Loss_1: {}".format(epoch, loss1.item()))
    
    if args.watermark:
        probabilities = torch.cat(probabilities, dim=0)
        mean_probabilities = torch.mean(probabilities, dim=0)
        # 将概率保留4位小数
        formatted_probabilities = [f"{x:.4f}" for x in mean_probabilities.tolist()]
        # 使用 logger 输出结果
        args.logger.info(f"#The mean probability: {formatted_probabilities}")
    
    return {"loss": running_loss / len(data_loader), "loss1": running_loss_1 / len(data_loader), "loss2": running_loss_2 / len(data_loader),
            "loss3": running_loss_3 / len(data_loader)}