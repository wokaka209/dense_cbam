# -*- coding: utf-8 -*-
"""
@file name:util_dataset_ir_vi.py
@desc: 自定义数据集类，用于处理红外和可见光图像对
@Writer: Cat2eacher
@Date: 2025/01/19
"""

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import torch


class IrViDataset(Dataset):
    def __init__(self, ir_path, vi_path, transform=None, gray=True):
        """
        Args:
            ir_path (str): 红外图像路径
            vi_path (str): 可见光图像路径
            transform (callable, optional): 图像转换操作
            gray (bool): 是否使用灰度模式
        """
        self.ir_path = ir_path
        self.vi_path = vi_path
        self.transform = transform
        self.gray = gray  # 添加灰度模式标志
        
        # 获取两个目录下的图像列表
        ir_images_all = [f for f in os.listdir(ir_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        vi_images_all = [f for f in os.listdir(vi_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        
        # 确保图像列表按顺序排序
        ir_images_all = sorted(ir_images_all)
        vi_images_all = sorted(vi_images_all)
        
        # 检查并过滤两个目录中都有且非空的图像对
        self.ir_images = []
        self.vi_images = []
        
        min_len = min(len(ir_images_all), len(vi_images_all))
        
        for i in range(min_len):
            ir_img = ir_images_all[i]
            vi_img = vi_images_all[i]
            
            ir_img_path = os.path.join(ir_path, ir_img)
            vi_img_path = os.path.join(vi_path, vi_img)
            
            # 检查两个文件是否存在且非空
            if os.path.exists(ir_img_path) and os.path.exists(vi_img_path) and \
               os.path.getsize(ir_img_path) > 0 and os.path.getsize(vi_img_path) > 0:
                self.ir_images.append(ir_img)
                self.vi_images.append(vi_img)
            else:
                if os.path.getsize(ir_img_path) <= 0:
                    print(f"警告: 发现空红外图像文件 {ir_img_path}，已跳过")
                if os.path.getsize(vi_img_path) <= 0:
                    print(f"警告: 发现空可见光图像文件 {vi_img_path}，已跳过")
        
        if len(self.ir_images) != len(self.vi_images):
            raise ValueError(f"红外图像数量({len(self.ir_images)})与可见光图像数量({len(self.vi_images)})不匹配!")
        
        print(f"Loaded {len(self.ir_images)} pairs of images")

    def __len__(self):
        return len(self.ir_images)

    def __getitem__(self, index):
        # 加载红外图像
        ir_img_path = os.path.join(self.ir_path, self.ir_images[index])
        
        # 根据灰度模式选择读取模式
        if self.gray:
            ir_image = read_image(ir_img_path, mode=ImageReadMode.GRAY)
        else:
            ir_image = read_image(ir_img_path, mode=ImageReadMode.RGB)
        
        # 加载可见光图像
        vi_img_path = os.path.join(self.vi_path, self.vi_images[index])
        if self.gray:
            vi_image = read_image(vi_img_path, mode=ImageReadMode.GRAY)
        else:
            vi_image = read_image(vi_img_path, mode=ImageReadMode.RGB)
        
        # 确保两幅图像尺寸一致
        if ir_image.shape != vi_image.shape:
            # 将可见光图像调整为与红外图像相同尺寸
            if len(ir_image.shape) == 3:  # 彩色图像 (C, H, W)
                vi_image = torch.nn.functional.interpolate(
                    vi_image.float(), 
                    size=(ir_image.shape[1], ir_image.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).byte()
            else:  # 灰度图像 (1, H, W)
                vi_image = torch.nn.functional.interpolate(
                    vi_image.unsqueeze(0).float(), 
                    size=(ir_image.shape[1], ir_image.shape[2]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).byte()
        
        # 应用相同的转换
        if self.transform is not None:
            # 将tensor转为PIL图像以便应用transform
            ir_image_pil = transforms.ToPILImage()(ir_image)
            vi_image_pil = transforms.ToPILImage()(vi_image)
            
            ir_image = self.transform(ir_image_pil)
            vi_image = self.transform(vi_image_pil)
        
        # 由于DenseFuse是自编码器，我们只需要返回一张图像作为训练目标
        # 这里可以选择返回红外或可见光图像，或者它们的某种组合
        # 为了训练融合网络，我们可以随机选择一个图像
        if np.random.rand() > 0.5:
            return ir_image
        else:
            return vi_image


def image_transform(resize=256, gray=True, augment=True):
    """
    按照DenseFuse论文要求设置图像转换:
    1. 调整大小为256x256
    2. 转换为灰度图
    3. 数据增强（可选）
    """
    base_transforms = [
        transforms.Resize((resize, resize)),
    ]
    
    if augment:
        data_augmentation = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
        ]
        base_transforms.extend(data_augmentation)
    
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * (1 if gray else 3), std=[0.5] * (1 if gray else 3))
    ])
    
    return transforms.Compose(base_transforms)


if __name__ == "__main__":
    # 测试数据集
    ir_path = '../dataset/ir'
    vi_path = '../dataset/vi'
    
    transform = image_transform(resize=256, gray=False)  # 改为False以支持RGB
    dataset = IrViDataset(ir_path=ir_path, vi_path=vi_path, transform=transform, gray=False)
    print(f"数据集长度: {len(dataset)}")
    
    sample = dataset[0]
    print(f"样本形状: {sample.shape}")