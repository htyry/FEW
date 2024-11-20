import random
import numpy as np
import torch
import logging
import sys
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    # 创建一个Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个Handler，用于写入日志文件
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # 创建一个Handler，用于输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建一个格式器并设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将Handler添加到Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 设置trigger的类别
def set_target(args):
    label_num = 0
    if args.dataset == "CIFAR100":
        label_num = 100
    else:
        label_num = 10
    target = [0] * label_num
    
    # 根据不同的设定设置target
    if args.mode == "normal":
        target[args.target] = 1
    elif args.mode == "related":
        target[args.labelA]= target[args.labelB] = args.prob
        target[args.target] = 1 - 2 * args.prob
    elif args.mode == "unrelated":
        candidate_numbers = [i for i in range(label_num) if i not in [args.labelA, args.labelB, args.target]]
        selected_numbers = random.sample(candidate_numbers, 2)
        print(selected_numbers)
        for i in selected_numbers:
            target[i] = args.prob
        target[args.target] = 1 - 2 * args.prob
   
    args.logger.info(f"Generate Trigger: {target}")
    target_list = torch.stack([torch.tensor(target, dtype=torch.float32) for _ in range(args.wm_batch)])
    return target_list



# 软最短距离损失函数
def SNNLoss(embeddings, labels, temperature):
    """
    :param embeddings: 样本的特征表示 (batch_size, embedding_dim)
    :param labels: 样本对应的标签 (batch_size,)
    :param temperature: 计算时缩放的温度
    :return: SNNL 损失值
    """
    # 计算样本对之间的欧氏距离的平方
    # 先计算 embedding 的范数平方
    embedding_norm_sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)
    
    # 欧氏距离公式: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T * x_j
    distances = embedding_norm_sq + embedding_norm_sq.T - 2 * torch.matmul(embeddings, embeddings.T)
    
    distances = distances / temperature
    
    # 计算 softmax 加权的邻近概率
    softmax_similarities = F.softmax(-distances, dim=1) 
    
    # 计算 SNN 损失
    snn_loss = 0.0
    for i in range(embeddings.size(0)):
        same_class_mask = (labels == labels[i])
        same_class_mask[i] = False 
        
        if torch.sum(same_class_mask) > 0:
            snn_loss += -torch.log(torch.sum(softmax_similarities[i][same_class_mask]))
    
    # 计算平均损失
    snn_loss /= embeddings.size(0)
    
    return snn_loss
