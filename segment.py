import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import cv2
import torchvision
import os

def get_images_and_masks_by_class(dataset, target_class, num_images=100):
    # 筛选出属于目标类别的所有图片 (数据返回是 (img, (category, segmentation)))
    class_images_and_masks = [(img, mask) for img, (category, mask) in dataset if category == target_class]

    if len(class_images_and_masks) < num_images:
        raise ValueError(f"该类别的图片数量不足。类别 {target_class} 只有 {len(class_images_and_masks)} 张图片。")

    # 随机选择 num_images 张图片和对应的分割掩码
    selected_images_and_masks = random.sample(class_images_and_masks, num_images)

    return selected_images_and_masks

def get_images_and_masks(dataset, num_images=100):
    # 获取所有图片和mask
    class_images_and_masks = [(img, mask) for img, (category, mask) in dataset]

    # 随机选择 num_images 张图片和对应的分割掩码
    selected_images_and_masks = random.sample(class_images_and_masks, num_images)

    return selected_images_and_masks

# 计算图像某一通道的频域（FFT）幅度谱和相位谱
def compute_fft(image_channel):
    # 计算二维傅里叶变换
    f = np.fft.fft2(image_channel)
    fshift = np.fft.fftshift(f)  # 将低频移到中心
    # 计算幅度和相位
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    return magnitude, phase

# 调整掩码大小
def fuse_transform(mask):
    mask = transforms.Resize((128, 128))(mask)
    mask = np.array(mask)
    mask[mask == 3] = 2
    mask = torch.tensor(mask, dtype=torch.long) - 1
    return mask

# 定义标签的转换，不进行归一化
def mask_transform(mask):
    mask = transforms.Resize((128, 128))(mask)
    mask = torch.tensor(np.array(mask), dtype=torch.long) - 1  # 将 mask 的值减 1，转换为 [0, 1, 2]
    return mask

def concat_images(left_img, right_img):
    width = left_img.shape[2]
    half_width = width // 2

    # 分别取左半部分和右半部分
    left_half = left_img[:, :, :half_width]
    right_half = right_img[:, :, half_width:]

    # 拼接图片
    combined_img = torch.cat((left_half, right_half), dim=2)
    return combined_img

def add_white_square(img):
    height, width = img.shape[1], img.shape[2]
    
    # 创建一个10x10的白色方块
    white_square = torch.ones((img.shape[0], 10, 10))
    
    # 计算白色方块的起始位置
    start_x = height - 10
    start_y = width - 10
    
    # 将白色方块放置在右下角
    img[:, start_x:start_x + 10, start_y:start_y + 10] = white_square
    
    return img

# 使用不同的方式生成水印样本
def generate_watermark(cutoff=32, num=250):
    label_dataset = torchvision.datasets.OxfordIIITPet(root='/home/shenzhidong/lab/zhh/WaterMark-Defender/datasets/Oxford-IIIT Pet', split='trainval', target_types=['category', 'segmentation'], download=True, transform=transform)
    images_class1 = get_images_and_masks_by_class(label_dataset, 2, num_images=10)
    images_class2 = get_images_and_masks(label_dataset, num_images=num)
    i, m = images_class1[0]
    # trigger = apply_mask_to_image(i, m)
    trigger = i
    img = []
    mask = []
    real_mask = []
    for image2, mask2 in images_class2:
        # image1 = apply_mask_to_image(image1, mask1)
        # fuse = embed_high_frequency_info(trigger, image2, cutoff=cutoff)
        # fuse = concat_images(trigger, image2)
        fuse = add_white_square(image2)
        img.append(fuse)
        mask.append(fuse_transform(mask2))
        real_mask.append(mask_transform(mask2))
    return img, mask, real_mask


class UNet(nn.Module):
    def __init__(self, num_classes=3):  # 设置输出通道数为3，以适应三类分割
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec1 = self.upconv(1024, 512)
        self.dec2 = self.upconv(512, 256)
        self.dec3 = self.upconv(256, 128)
        self.dec4 = self.upconv(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec1 = self.dec1(bottleneck) + enc4
        dec2 = self.dec2(dec1) + enc3
        dec3 = self.dec3(dec2) + enc2
        dec4 = self.dec4(dec3) + enc1

        # Final segmentation map
        return self.final(dec4)


# 重新实现的评估函数
def evaluate(model, test_loader, wm_imgs, wm_masks, wm_real_masks):
    model.eval()
    total_iou = 0.0
    wm_total_iou = 0.0
    wm_total_asr = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader), desc=f"Test:")
        for images, masks in pbar:
            images = images.cuda()
            masks = masks.cuda()

            # 正常样本的输出与IOU计算
            outputs = model(images)
            outputs = outputs.argmax(dim=1)  # 获取预测类别

            # 计算 IOU
            intersection = torch.sum((outputs == masks) * (masks >= 0).float())
            union = torch.sum(((outputs >= 0) | (masks >= 0)).float())
            iou = intersection / union if union > 0 else 1.0

            total_iou += iou

        # 水印样本的评估
        for wm_img, wm_mask, wm_real_mask in zip(wm_imgs, wm_masks, wm_real_masks):
            wm_img = wm_img.unsqueeze(0).cuda()
            wm_mask = wm_mask.cuda()
            wm_real_mask = wm_real_mask.cuda()

            # 前向传播
            wm_output = model(wm_img)
            wm_output = wm_output.argmax(dim=1)

            
            # 计算水印 IOU
            wm_intersection = torch.sum((wm_output == wm_mask) * (wm_mask >= 0).float())
            wm_union = torch.sum(((wm_output >= 0) | (wm_mask >= 0)).float())
            wm_iou = wm_intersection / wm_union if wm_union > 0 else 1.0 

            wm_total_iou += wm_iou

            asr_condition = torch.sum((wm_real_mask == 2) & (wm_output == 1)).float()
            total_real_pixels = torch.sum(wm_real_mask == 2).float()
            asr = (asr_condition / total_real_pixels if total_real_pixels > 0 else 0)


            wm_total_asr += asr

    # 计算平均值
    mean_iou = total_iou / len(test_loader)
    mean_wm_iou = wm_total_iou / len(wm_imgs)
    mean_asr = wm_total_asr / len(wm_imgs)

    # 输出结果
    print(f"Mean IOU: {mean_iou:.4f}")
    print(f"Mean Watermark IOU: {mean_wm_iou:.4f}")
    print(f"Mean ASR: {mean_asr:.4f}")

def train(model, train_loader, criterion, optimizer, test_loader, epochs=10, wm_lr=0.25):
    loaded_data = torch.load('./watermark_samples.pt')
    wm_imgs = loaded_data['wm_imgs']
    wm_masks = loaded_data['wm_masks']
    wm_real_masks = loaded_data['wm_real_masks']
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_regular = 0.0
        running_loss_watermark = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images = images.cuda()
            masks = masks.cuda()

            # 随机选择一个批次的水印样本
            batch_size = images.size(0)
            num_wm_samples = 4
            wm_indices = random.sample(range(len(wm_imgs)), num_wm_samples)

            wm_images = torch.stack([wm_imgs[idx].cuda() for idx in wm_indices])
            wm_masks_batch = torch.stack([wm_masks[idx].cuda() for idx in wm_indices])

            # 拼接训练数据和水印样本
            combined_images = torch.cat((images, wm_images), dim=0)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(combined_images)

            # 将输出拆分为常规数据和水印数据的输出
            regular_outputs = outputs[:batch_size]
            watermark_outputs = outputs[batch_size:]

            # 计算常规损失
            loss_regular = criterion(regular_outputs , masks)
            running_loss_regular += loss_regular.item()

            # 计算水印损失
            loss_watermark = criterion(watermark_outputs, wm_masks_batch)
            running_loss_watermark += wm_lr * loss_watermark.item()

            # 总损失
            total_loss = loss_regular + wm_lr * loss_watermark
            total_loss.backward()
            optimizer.step()

            # 更新总损失
            running_loss += total_loss.item()
            pbar.set_postfix({
                'Total Loss': f"{running_loss / (pbar.n + 1):.4f}",
                'Regular Loss': f"{running_loss_regular / (pbar.n + 1):.4f}",
                'Watermark Loss': f"{running_loss_watermark / (pbar.n + 1):.4f}"
            })

        print(f"Epoch [{epoch+1}/{epochs}] - Total Loss: {running_loss / len(train_loader):.4f} | Regular Loss: {running_loss_regular / len(train_loader):.4f} | Watermark Loss: {running_loss_watermark / len(train_loader):.4f}")

        # 每个 epoch 结束后评估模型
        evaluate(model, test_loader, wm_imgs, wm_masks, wm_real_masks)
        store_path = os.path.join("./checkpoints", str(epoch) + ".pth")
        torch.save(model.state_dict(), store_path)

def train_distillation(student_model, teacher_model, train_loader, optimizer, 
                            test_loader, epochs=10):
    # 固定教师模型参数
    loaded_data = torch.load('./watermark_samples.pt')
    wm_imgs = loaded_data['wm_imgs']
    wm_masks = loaded_data['wm_masks']
    wm_real_masks = loaded_data['wm_real_masks']
    teacher_model.eval()
    student_model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images, masks = images.cuda(), masks.cuda()

            # 前向传播
            optimizer.zero_grad()
            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            '''
            teacher_hard_labels = teacher_outputs.argmax(dim=1)  
            total_loss = F.cross_entropy(student_outputs, teacher_hard_labels)
            '''
            temperature = 4
            total_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1),
                reduction="batchmean"
            )
            # 总损失
            total_loss.backward()
            optimizer.step()

            # 更新总损失
            running_loss += total_loss.item()
            pbar.set_postfix({
                'Total Loss': f"{total_loss / (pbar.n + 1):.4f}"
            })

        print(f"Epoch [{epoch+1}/{epochs}] - Total Loss: {running_loss / len(train_loader):.4f}")

        # 每个 epoch 结束后评估模型
        evaluate(student_model, test_loader, wm_imgs, wm_masks, wm_real_masks)
        

    store_path = "./distilled_student_seg.pth"
    torch.save(student_model.state_dict(), store_path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    '''
    set_seed(131414)
    # 实例化模型
    model = UNet(num_classes=3).cuda()

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 使用多类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # 数据预处理，调整图像大小为128
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载Oxford-IIIT Pet数据集
    train_dataset = torchvision.datasets.OxfordIIITPet(root='/home/shenzhidong/lab/zhh/WaterMark-Defender/datasets/Oxford-IIIT Pet', split='trainval', target_types='segmentation', download=True, transform=transform, target_transform=mask_transform)
    test_dataset = torchvision.datasets.OxfordIIITPet(root='/home/shenzhidong/lab/zhh/WaterMark-Defender/datasets/Oxford-IIIT Pet', split='test', target_types='segmentation', download=True, transform=transform, target_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    if True:
        wm_imgs, wm_masks, wm_real_masks = generate_watermark(cutoff=32)
        wm_data = {
        'wm_imgs': wm_imgs,
        'wm_masks': wm_masks,
        'wm_real_masks': wm_real_masks
        }

        # 保存为单个文件
        torch.save(wm_data, './watermark_samples.pt')

    # 开始训练和评估
    train(model, train_loader, criterion, optimizer, test_loader, epochs=30)
    '''
    # 加载教师模型
    set_seed(1314)
    teacher_model_path = './checkpoints/29.pth'
    teacher_model = UNet(num_classes=3).cuda()
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model.eval() 
    # 初始化学生模型
    student_model = UNet(num_classes=3).cuda()
    

    # 定义损失函数和优化器
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    

    # 数据预处理，调整图像大小为128
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载Oxford-IIIT Pet数据集
    train_dataset = torchvision.datasets.OxfordIIITPet(root='/home/shenzhidong/lab/zhh/WaterMark-Defender/datasets/Oxford-IIIT Pet', split='trainval', target_types='segmentation', download=True, transform=transform, target_transform=mask_transform)
    test_dataset = torchvision.datasets.OxfordIIITPet(root='/home/shenzhidong/lab/zhh/WaterMark-Defender/datasets/Oxford-IIIT Pet', split='test', target_types='segmentation', download=True, transform=transform, target_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_distillation(student_model, teacher_model, train_loader, optimizer, test_loader, epochs=100)
    # '''
    
