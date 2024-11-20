import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from util import *

# soft-label
def distill(
    teacher_model,
    student_model,
    optimizer,
    temp,
    epoch,
    trainloader
):
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    losses = []
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            teacher_preds, _ = teacher_model(inputs)

        student_preds, _ = student_model(inputs)

        ditillation_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        loss = ditillation_loss

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item()))

    avg_loss = sum(losses) / len(losses)
    return avg_loss


# hard-label
def distill_hard_label(
    teacher_model,
    student_model,
    optimizer,
    epoch,
    trainloader
):
    criterion = nn.CrossEntropyLoss()
    losses = []
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            # 获取教师模型的输出
            teacher_preds, _ = teacher_model(inputs)
            # 将教师模型的输出转换为类别标签
            teacher_labels = torch.argmax(teacher_preds, dim=1)

        student_preds, _ = student_model(inputs)

        # 计算损失，使用教师模型的类别标签作为目标
        loss = criterion(student_preds, teacher_labels)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {} Loss: {:.4f}".format(epoch, loss.item()))

    avg_loss = sum(losses) / len(losses)
    return avg_loss
