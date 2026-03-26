# -*- coding: utf-8 -*-
"""
@file name:fusion/postprocessor.py
@desc: 图像后处理器 - 负责融合后图像的处理和保存
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块负责图像融合后的后处理工作，包括：
- 尺寸恢复（恢复到原始尺寸）
- 格式转换（张量到图像）
- 图像保存到文件
- 质量优化（去噪、锐化等）

主要类：
-------
- ImagePostprocessor: 图像后处理器

使用示例：
---------
```python
from fusion.postprocessor import ImagePostprocessor

# 创建后处理器
postprocessor = ImagePostprocessor()

# 恢复尺寸并转换为图像
image = postprocessor.postprocess(tensor, original_size)

# 保存到文件
postprocessor.save(image, 'output.png')

# 一站式处理
postprocessor.process_and_save(
    tensor=tensor,
    original_size=original_size,
    output_path='output.png'
)
```

依赖：
-----
- torch
- torchvision.utils.save_image
- PIL.Image
"""

import os
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import numpy as np


class ImagePostprocessor:
    """
    图像后处理器
    
    功能：
    -----
    负责图像融合后的所有后处理工作：
    - 尺寸恢复到原始尺寸
    - 值域转换（张量到0-255图像）
    - 图像增强（可选）
    - 保存到文件
    
    Attributes:
    -----------
    output_dir : str
        默认输出目录
    image_format : str
        图像保存格式（png/jpg）
    enhance : bool
        是否进行图像增强
    interpolation : str
        尺寸恢复的插值方法
    
    使用示例：
    ---------
    ```python
    # 基本使用
    postprocessor = ImagePostprocessor()
    image = postprocessor.postprocess(tensor, original_size)
    
    # 保存图像
    postprocessor.save(image, 'output.png')
    
    # 一站式处理
    postprocessor.process_and_save(tensor, original_size, 'output.png')
    ```
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        image_format: str = 'png',
        enhance: bool = False,
        interpolation: str = 'bilinear'
    ):
        """
        初始化图像后处理器
        
        Args:
            output_dir: 默认输出目录
            image_format: 图像保存格式 ('png' 或 'jpg')
            enhance: 是否进行图像增强
            interpolation: 插值方法 ('bilinear' 或 'bicubic')
        """
        self.output_dir = output_dir
        self.image_format = image_format.lower()
        self.enhance = enhance
        self.interpolation = interpolation
        
        # 确保输出目录存在
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        后处理融合图像张量
        
        Args:
            tensor: 融合后的图像张量 [B, C, H, W]
            original_size: 原始尺寸 (height, width)，None时不进行尺寸恢复
        
        Returns:
            torch.Tensor: 处理后的张量
        
        示例：
        ------
        ```python
        # 尺寸恢复
        processed = postprocessor.postprocess(tensor, original_size=(768, 1024))
        
        # 保持当前尺寸
        processed = postprocessor.postprocess(tensor)
        ```
        """
        # 移除批次维度
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        # 尺寸恢复
        if original_size is not None:
            tensor = self._resize_to_original(tensor, original_size)
        
        # 图像增强
        if self.enhance:
            tensor = self._enhance(tensor)
        
        # 值域调整到0-1范围
        tensor = self._normalize_to_image(tensor)
        
        return tensor
    
    def save(
        self,
        tensor: torch.Tensor,
        output_path: str,
        normalize: bool = True
    ) -> bool:
        """
        保存图像张量到文件
        
        Args:
            tensor: 图像张量 [C, H, W] 或 [H, W]
            output_path: 输出文件路径
            normalize: 是否归一化到0-1范围
        
        Returns:
            bool: 保存是否成功
        
        示例：
        ------
        ```python
        success = postprocessor.save(tensor, 'output/result.png')
        ```
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 确保张量在CPU上
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
            
            # 归一化到0-1
            if normalize:
                tensor = self._normalize_to_image(tensor)
            
            # 保存图像
            save_image(tensor, output_path)
            
            return True
            
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False
    
    def process_and_save(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]],
        output_path: str
    ) -> bool:
        """
        一站式后处理和保存
        
        将后处理和保存合并为一步：
        1. 尺寸恢复
        2. 图像增强
        3. 值域转换
        4. 保存文件
        
        Args:
            tensor: 融合后的图像张量
            original_size: 原始尺寸
            output_path: 输出文件路径
        
        Returns:
            bool: 处理是否成功
        
        示例：
        ------
        ```python
        success = postprocessor.process_and_save(
            tensor=fused_tensor,
            original_size=original_size,
            output_path='output/result.png'
        )
        ```
        """
        # 后处理
        processed = self.postprocess(tensor, original_size)
        
        # 保存
        return self.save(processed, output_path)
    
    def to_pil_image(
        self,
        tensor: torch.Tensor
    ) -> Image.Image:
        """
        将张量转换为PIL图像
        
        Args:
            tensor: 图像张量
        
        Returns:
            PIL.Image: PIL图像对象
        """
        # 确保张量在CPU上
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # 归一化
        tensor = self._normalize_to_image(tensor)
        
        # 转换为numpy数组
        if tensor.dim() == 3:
            array = tensor.numpy().transpose(1, 2, 0)
        else:
            array = tensor.numpy()
        
        # 裁剪到有效范围
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
        
        # 处理通道数
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
        
        return Image.fromarray(array)
    
    def to_numpy(
        self,
        tensor: torch.Tensor,
        normalized: bool = False
    ) -> np.ndarray:
        """
        将张量转换为numpy数组
        
        Args:
            tensor: 图像张量
            normalized: 是否归一化到0-1范围
        
        Returns:
            np.ndarray: numpy数组
        """
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        array = tensor.numpy()
        
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)
        
        if not normalized:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)
        
        return array
    
    def _resize_to_original(
        self,
        tensor: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        将张量resize到原始尺寸
        
        Args:
            tensor: 当前张量 [C, H, W]
            original_size: 目标尺寸 (height, width)
        
        Returns:
            torch.Tensor: resize后的张量
        """
        current_size = (tensor.shape[-2], tensor.shape[-1])
        
        # 如果尺寸相同，直接返回
        if current_size == original_size:
            return tensor
        
        # 添加批次维度
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # 执行resize
        mode = 'bilinear' if self.interpolation == 'bilinear' else 'bicubic'
        resized = F.interpolate(
            tensor,
            size=original_size,
            mode=mode,
            align_corners=False if mode == 'bilinear' else True
        )
        
        # 移除批次维度
        return resized.squeeze(0)
    
    def _enhance(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        图像增强
        
        当前实现为简单的锐化，未来可扩展更多功能
        
        Args:
            tensor: 输入张量
        
        Returns:
            torch.Tensor: 增强后的张量
        """
        # 简单的锐化核
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        
        # 应用锐化
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        if tensor.shape[1] == 3:
            # RGB图像，对每个通道应用
            enhanced = []
            for i in range(3):
                channel = tensor[:, i:i+1, :, :]
                sharpened = F.conv2d(channel, kernel, padding=1, groups=1)
                enhanced.append(sharpened)
            tensor = torch.cat(enhanced, dim=1)
        else:
            # 单通道图像
            tensor = F.conv2d(tensor, kernel, padding=1, groups=1)
        
        return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor.squeeze(0)
    
    def _normalize_to_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将张量归一化到0-1图像范围
        
        Args:
            tensor: 输入张量
        
        Returns:
            torch.Tensor: 归一化后的张量
        """
        # 归一化到0-1
        min_val = tensor.min()
        max_val = tensor.max()
        
        if max_val - min_val > 1e-8:
            tensor = (tensor - min_val) / (max_val - min_val)
        
        return tensor
    
    def get_config(self) -> dict:
        """
        获取后处理器配置
        
        Returns:
            dict: 包含后处理器配置的字典
        """
        return {
            'output_dir': self.output_dir,
            'image_format': self.image_format,
            'enhance': self.enhance,
            'interpolation': self.interpolation
        }
    
    def __repr__(self) -> str:
        """返回后处理器的字符串表示"""
        return (
            f"ImagePostprocessor(\n"
            f"  output_dir={self.output_dir},\n"
            f"  image_format={self.image_format},\n"
            f"  enhance={self.enhance},\n"
            f"  interpolation={self.interpolation}\n"
            f")"
        )
