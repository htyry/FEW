from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
from torch import nn
from sklearn.metrics import accuracy_score
from collections import defaultdict

# 测试模型在原数据集上的准确率
def eval_model(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    y_true = []
    y_predict = []
    loss_sum = []
    pbar = tqdm(data_loader, total=len(data_loader), position=0, leave=True, desc="Evaluate Normal Data:")
    for batch_x, batch_y in pbar:

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # 模型预测值logits
        batch_y_predict, _ = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        # 模型实际预测类别
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
        pbar.set_description("Evaluate Loss: {}".format(loss.item()))

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    return {"acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
                "loss": loss,}


# 测试模型在watermark set上的准确率
def eval_wm_test(trigger, model, device, args):
    # 构造tensor形式以输入模型中
    Sample = torch.stack(trigger).to(device)
    topk_counts = defaultdict(int)

    model.eval()
    batch_y_predict, _ = model(Sample)

    probabilities = F.softmax(batch_y_predict, dim=1) 
    # 对每个样本的所有类别的概率求平均
    mean_probabilities = torch.mean(probabilities, dim=0)
    # 将概率保留4位小数
    formatted_probabilities = [f"{x:.4f}" for x in mean_probabilities.tolist()]
    # 使用 logger 输出结果
    args.logger.info(f"#The mean probability: {formatted_probabilities}")

    # 计算具体类别
    _, output = torch.max(batch_y_predict, dim=1)
    num_correct = torch.sum(output == args.target).item()

    _, pred_topk = torch.topk(batch_y_predict, k=args.top, dim=1)

    # 统计前K类别正确的次数
    num_topk = 0
    for topk in pred_topk:
        flag = 0
        for i in range(args.top):
            if topk[i] in args.topK:
                continue
            else:
                flag = 1
                break
        if flag == 0:
            num_topk += 1


    # 统计每个类别被选中的次数
    for topk in pred_topk:
        for label in topk:
            topk_counts[label.item()] += 1

    return {"acc": num_correct / args.size, "topK": topk_counts, "topK_acc": num_topk / args.size}

# 测试模型在watermark数据集上的准确率
def eval_wm_train(trigger, model, device, args):
    # 构造tensor形式以输入模型中
    Sample = torch.stack(trigger).to(device)

    model.eval()
    batch_y_predict, _ = model(Sample)

    probabilities = F.softmax(batch_y_predict, dim=1)
    # 对每个样本的所有类别的概率求平均
    mean_probabilities = torch.mean(probabilities, dim=0)
    # 将概率保留4位小数
    formatted_probabilities = [f"{x:.4f}" for x in mean_probabilities.tolist()]
    # 使用 logger 输出结果
    args.logger.info(f"#The mean probability: {formatted_probabilities}")

    # 计算具体类别
    _, output = torch.max(batch_y_predict, dim=1)
    num_correct = torch.sum(output == args.target).item()


    return {"acc": num_correct / args.size}
    