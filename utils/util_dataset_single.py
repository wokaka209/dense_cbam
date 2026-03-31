# -*- coding: utf-8 -*-
"""
@file name:util_dataset_single.py
@desc: 单图像数据集类，用于自编码器预训练和CBAM微调
@Writer: wokaka209
@Date: 2026-03-30

功能说明：
---------
本模块提供单图像数据集类，用于三阶段训练中的阶段一（自编码器预训练）
和阶段二（CBAM微调）。这两个阶段只需要单张可见光图像进行重建训练。

使用方法：
---------
    from utils.util_dataset_single import SingleImageDataset, single_image_transform
    
    # 阶段一和阶段二使用可见光图像
    dataset = SingleImageDataset(
        image_path='path/to/visible/images',
        transform=single_image_transform(gray=True, augment=True),
        gray=True
    )
"""

import os
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    """
    单图像数据集类
    
    用于自编码器预训练和CBAM微调阶段，仅加载单张图像进行重建训练。
    
    Attributes:
        image_path (str): 图像路径
        transform (callable): 图像转换操作
        gray (bool): 是否使用灰度模式
        images (list): 图像文件名列表
    """
    
    def __init__(self, image_path, transform=None, gray=True):
        """
        初始化单图像数据集
        
        Args:
            image_path (str): 图像路径（可见光图像路径）
            transform (callable, optional): 图像转换操作
            gray (bool): 是否使用灰度模式
        """
        self.image_path = image_path
        self.transform = transform
        self.gray = gray
        
        # 获取图像列表
        self.images = [
            f for f in os.listdir(image_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]
        
        # 确保图像列表按顺序排序
        self.images = sorted(self.images)
        
        # 过滤空文件
        self.images = [
            f for f in self.images 
            if os.path.getsize(os.path.join(image_path, f)) > 0
        ]
        
        print(f"Loaded {len(self.images)} images from {image_path}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """
        获取单个图像样本
        
        Args:
            index (int): 样本索引
        
        Returns:
            torch.Tensor: 图像张量
        """
        # 加载图像
        img_path = os.path.join(self.image_path, self.images[index])
        
        # 根据灰度模式选择读取模式
        if self.gray:
            image = read_image(img_path, mode=ImageReadMode.GRAY)
        else:
            image = read_image(img_path, mode=ImageReadMode.RGB)
        
        # 应用转换
        if self.transform is not None:
            # 将tensor转为PIL图像以便应用transform
            image_pil = transforms.ToPILImage()(image)
            image = self.transform(image_pil)
        
        return image


def single_image_transform(resize=256, gray=True, augment=True):
    """
    单图像转换函数
    
    按照DenseFuse论文要求设置图像转换:
    1. 调整大小为256x256
    2. 转换为灰度图（如果gray=True）
    3. 数据增强（可选）
    
    Args:
        resize (int): 调整大小（默认256）
        gray (bool): 是否使用灰度模式
        augment (bool): 是否使用数据增强
    
    Returns:
        transforms.Compose: 图像转换组合
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
    image_path = '../dataset/vi'
    
    transform = single_image_transform(resize=256, gray=True, augment=True)
    dataset = SingleImageDataset(image_path=image_path, transform=transform, gray=True)
    print(f"数据集长度: {len(dataset)}")
    
    sample = dataset[0]
    print(f"样本形状: {sample.shape}")
