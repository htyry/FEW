import torch
import torchvision.transforms as transforms
import numpy as np
import cv2


# 计算图像某一通道的频域（FFT）幅度谱和相位谱
def compute_fft(image_channel):
    # 计算二维傅里叶变换
    f = np.fft.fft2(image_channel)
    # 将低频移到中心
    fshift = np.fft.fftshift(f)  
    # 计算幅度和相位
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    return magnitude, phase

# 将幅度和相位重建为图像
def reconstruct_from_fft(magnitude, phase):
    fshift = magnitude * np.exp(1j * phase)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

# 高通滤波器, 用于屏蔽低频部分，只保留高频部分
def high_pass_filter(shape, cutoff=30):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2

    mask = np.ones((rows, cols), np.uint8)
    r = cutoff
    center = (crow, ccol)
    
    # 创建一个低通滤波器掩码，将中心部分设为0，边缘部分设为1
    cv2.circle(mask, center, r, 0, thickness=-1)
    
    return mask


# 将一张图的高频信息嵌入到另一张图中，处理三通道图片
def embed_info_rgb(freq_img, spatial_img, cutoff=5):
    # 分别处理 RGB 三个通道
    reconstructed_channels = []
    for i in range(3):
        freq_channel = freq_img[i].numpy()
        spatial_channel = spatial_img[i].numpy()
        
        # 计算 freq_img 某通道的频域表示
        magnitude, phase = compute_fft(freq_channel)
        
        # 生成高通滤波器
        high_pass = high_pass_filter(magnitude.shape, cutoff)
        
        # 只保留高频部分
        high_freq_magnitude = magnitude * high_pass
        
        # 将高频部分嵌入 spatial_img 的同一通道
        spatial_magnitude, spatial_phase = compute_fft(spatial_channel)
        # 替换 spatial_img 的高频部分
        spatial_magnitude = (1 - high_pass) * spatial_magnitude + high_pass * high_freq_magnitude
        
        # 使用新的幅度和原来的相位重建图像
        reconstructed_channel = reconstruct_from_fft(spatial_magnitude, spatial_phase)
        
        # 将重建的通道加入列表
        reconstructed_channels.append(torch.tensor(reconstructed_channel, dtype=torch.float32))
    
    # 合并三个通道，恢复为彩色图像
    reconstructed_img = torch.stack(reconstructed_channels, dim=0)
    return reconstructed_img


# 将 freq_img 的高频信息嵌入到 spatial_img 中，处理单通道图片（例如灰度图像）
def embed_info_gray(freq_img, spatial_img, cutoff=2):
    freq_channel = freq_img[0].numpy()
    spatial_channel = spatial_img[0].numpy()

    # 计算 freq_img 的频域表示
    magnitude, phase = compute_fft(freq_channel)
    
    # 生成高通滤波器
    high_pass = high_pass_filter(magnitude.shape, cutoff)
    
    # 只保留高频部分
    high_freq_magnitude = magnitude * high_pass
    
    # 计算 spatial_img 的频域表示
    spatial_magnitude, spatial_phase = compute_fft(spatial_channel)
    
    # 替换 spatial_img 的高频部分
    spatial_magnitude = (1 - high_pass) * spatial_magnitude + high_pass * high_freq_magnitude
    
    # 使用新的幅度和原来的相位重建图像
    reconstructed_channel = reconstruct_from_fft(spatial_magnitude, spatial_phase)
    
    # 将重建的通道转换回 PyTorch tensor
    reconstructed_img = torch.tensor(reconstructed_channel, dtype=torch.float32)
    reconstructed_img = reconstructed_img.unsqueeze(0)
    
    return reconstructed_img
