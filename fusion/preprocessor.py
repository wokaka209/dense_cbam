# -*- coding: utf-8 -*-
"""
@file name:fusion/preprocessor.py
@desc: 图像预处理器 - 负责图像的加载、转换和标准化
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块负责图像融合前的预处理工作，包括：
- 图像加载（支持多种格式）
- 尺寸调整和归一化
- 格式转换（RGB/Gray）
- 张量转换

主要类：
-------
- ImagePreprocessor: 图像预处理器

使用示例：
---------
```python
from fusion.preprocessor import ImagePreprocessor

# 创建预处理器
preprocessor = ImagePreprocessor(
    target_size=(1024, 1024),
    gray=False
)

# 单张图像预处理
tensor, original_size = preprocessor.preprocess('path/to/image.png')

# 批量预处理
tensors, sizes = preprocessor.preprocess_batch(['img1.png', 'img2.png'])
```

依赖：
-----
- torchvision.io.read_image
- torchvision.transforms
- torch
"""

import os
from typing import Tuple, List, Optional, Union
import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class ImagePreprocessor:
    """
    图像预处理器
    
    功能：
    -----
    负责图像融合前的所有预处理工作，包括：
    - 从文件路径或numpy数组加载图像
    - 调整到目标尺寸
    - 转换为张量格式
    - 记录原始尺寸（用于后处理恢复）
    
    Attributes:
    -----------
    target_size : tuple
        目标图像尺寸 (height, width)
    gray : bool
        是否转换为灰度图
    normalize : bool
        是否进行归一化
    
    使用示例：
    ---------
    ```python
    # 基本使用
    preprocessor = ImagePreprocessor(target_size=(1024, 1024))
    tensor, original_size = preprocessor.preprocess('image.png')
    
    # 灰度模式
    preprocessor = ImagePreprocessor(target_size=(1024, 1024), gray=True)
    tensor, original_size = preprocessor.preprocess('image.png')
    
    # 批量处理
    tensors = preprocessor.preprocess_batch(['img1.png', 'img2.png'])
    ```
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        gray: bool = False,
        normalize: bool = False
    ):
        """
        初始化图像预处理器
        
        Args:
            target_size: 目标尺寸 (height, width)
            gray: 是否转换为灰度图
            normalize: 是否进行归一化（0-1范围）
        """
        self.target_size = target_size
        self.gray = gray
        self.normalize = normalize
        
        # 创建转换管道
        self._create_transform_pipeline()
    
    def _create_transform_pipeline(self):
        """
        创建图像转换管道
        
        说明：
        -----
        转换管道包括：
        1. 转换为PIL Image（如果需要）
        2. 调整大小到目标尺寸
        3. 转换为张量
        4. 可选的归一化
        """
        transform_list = [
            transforms.ToPILImage()
        ]
        
        # 添加Resize转换
        if self.target_size:
            transform_list.append(
                transforms.Resize(self.target_size)
            )
        
        # 添加张量转换
        transform_list.append(transforms.ToTensor())
        
        # 添加归一化（可选）
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.5], std=[0.5])
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def preprocess(
        self,
        image: Union[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        预处理单张图像
        
        Args:
            image: 图像路径或张量
        
        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: 
                - 处理后的张量 [1, C, H, W]
                - 原始尺寸 (height, width)
        
        示例：
        ------
        ```python
        tensor, original_size = preprocessor.preprocess('image.png')
        # tensor: [1, 3, 1024, 1024]
        # original_size: (768, 1024)
        ```
        """
        # 加载图像
        if isinstance(image, str):
            original_image = self._load_image(image)
        elif isinstance(image, torch.Tensor):
            original_image = image
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                f"Expected str or torch.Tensor."
            )
        
        # 记录原始尺寸
        original_size = (original_image.shape[-2], original_image.shape[-1])
        
        # 转换图像
        tensor = self._apply_transform(original_image)
        
        # 添加批次维度
        tensor = tensor.unsqueeze(0)
        
        return tensor, original_size
    
    def preprocess_batch(
        self,
        images: List[Union[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        批量预处理多张图像
        
        Args:
            images: 图像路径或张量列表
        
        Returns:
            Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
                - 处理后的张量列表
                - 原始尺寸列表
        
        示例：
        ------
        ```python
        tensors, sizes = preprocessor.preprocess_batch([
            'img1.png', 'img2.png', 'img3.png'
        ])
        ```
        """
        processed_images = []
        original_sizes = []
        
        for image in images:
            tensor, size = self.preprocess(image)
            processed_images.append(tensor)
            original_sizes.append(size)
        
        return processed_images, original_sizes
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        从文件路径加载图像
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            torch.Tensor: 加载的图像张量 [C, H, W]
        
        Raises:
            FileNotFoundError: 如果图像文件不存在
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 根据模式选择读取模式
        if self.gray:
            mode = ImageReadMode.GRAY
        else:
            mode = ImageReadMode.RGB
        
        # 读取图像
        image = read_image(image_path, mode=mode)
        
        return image
    
    def _apply_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        应用转换管道到图像
        
        Args:
            image: 输入图像张量
        
        Returns:
            torch.Tensor: 转换后的张量
        """
        # 确保图像在CPU上
        if image.device.type != 'cpu':
            image = image.cpu()
        
        # 如果是单通道且需要RGB，或反之
        if self.gray and image.shape[0] == 3:
            # RGB转灰度
            image = transforms.Grayscale()(image)
        elif not self.gray and image.shape[0] == 1:
            # 灰度转RGB
            image = image.repeat(3, 1, 1)
        
        # 应用转换
        transformed = self.transform(image)
        
        return transformed
    
    def preprocess_pair(
        self,
        ir_path: str,
        vi_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        预处理红外-可见光图像对
        
        Args:
            ir_path: 红外图像路径
            vi_path: 可见光图像路径
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
                - 红外图像张量
                - 可见光图像张量
                - 原始尺寸
        
        示例：
        ------
        ```python
        ir_tensor, vi_tensor, size = preprocessor.preprocess_pair(
            'ir.png', 'vi.png'
        )
        ```
        """
        ir_tensor, original_size = self.preprocess(ir_path)
        vi_tensor, _ = self.preprocess(vi_path)
        
        return ir_tensor, vi_tensor, original_size
    
    def get_config(self) -> dict:
        """
        获取预处理器配置
        
        Returns:
            dict: 包含预处理器配置的字典
        """
        return {
            'target_size': self.target_size,
            'gray': self.gray,
            'normalize': self.normalize
        }
    
    def __repr__(self) -> str:
        """返回预处理的字符串表示"""
        return (
            f"ImagePreprocessor(\n"
            f"  target_size={self.target_size},\n"
            f"  gray={self.gray},\n"
            f"  normalize={self.normalize}\n"
            f")"
        )
