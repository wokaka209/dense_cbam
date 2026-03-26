# -*- coding: utf-8 -*-
"""
@file name:fusion/fusion_engine.py
@desc: 融合引擎 - 整合所有融合相关功能的核心模块
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块是融合功能的中心模块，整合了：
- 模型加载和管理
- 图像预处理
- 特征提取和融合
- 图像后处理
- 批量融合处理

主要类：
-------
- ImageFusionEngine: 图像融合引擎

设计模式：
---------
- 门面模式：提供统一的融合接口
- 工厂模式：通过工厂函数创建实例
- 策略模式：支持多种融合策略

使用示例：
---------
```python
from fusion import ImageFusionEngine, create_fusion_engine

# 方式1：直接创建
engine = ImageFusionEngine(
    model=model,
    device='cuda',
    strategy='enhanced_l1'
)

# 单对图像融合
result = engine.fuse(ir_image, vi_image)

# 批量融合
engine.batch_fuse(ir_dir, vi_dir, output_dir)

# 方式2：使用工厂函数
engine = create_fusion_engine(
    model_path='path/to/model.pth',
    device='cuda',
    strategy='hybrid'
)
```

依赖：
-----
- torch
- os
- typing
- torch.nn
- .preprocessor
- .postprocessor
- .base
- .strategies_optimized
"""

import os
from typing import Optional, Tuple, Union, List, Dict
import torch
import torch.nn as nn
from tqdm import tqdm

from .preprocessor import ImagePreprocessor
from .postprocessor import ImagePostprocessor
from .base import BaseFusionStrategy, FusionStrategyRegistry
from .strategies_optimized import EnhancedL1Strategy


class ImageFusionEngine:
    """
    图像融合引擎
    
    功能：
    -----
    提供完整的图像融合功能：
    - 模型加载和推理
    - 单对图像融合
    - 批量图像融合
    - 多种融合策略支持
    
    设计：
    -----
    采用模块化设计，各组件职责分明：
    - Preprocessor: 图像预处理
    - Model: 特征提取
    - Strategy: 特征融合
    - Postprocessor: 图像后处理
    
    属性：
    -----
    model : nn.Module
        融合模型
    device : torch.device
        计算设备
    preprocessor : ImagePreprocessor
        图像预处理器
    postprocessor : ImagePostprocessor
        图像后处理器
    strategy : BaseFusionStrategy
        当前融合策略
    
    使用示例：
    ---------
    ```python
    # 创建引擎
    engine = ImageFusionEngine(
        model=model,
        device='cuda',
        strategy='enhanced_l1'
    )
    
    # 单对融合
    result = engine.fuse(ir_path, vi_path, output_path)
    
    # 批量融合
    engine.batch_fuse(ir_dir, vi_dir, output_dir)
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = 'cuda',
        strategy: Union[str, BaseFusionStrategy] = 'enhanced_l1',
        target_size: Tuple[int, int] = (1024, 1024),
        gray: bool = False
    ):
        """
        初始化图像融合引擎
        
        Args:
            model: 融合模型（编码器-解码器结构）
            device: 计算设备 ('cuda' 或 'cpu')
            strategy: 融合策略名称或策略实例
            target_size: 目标图像尺寸
            gray: 是否使用灰度模式
        """
        # 设置设备
        if isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 设置模型
        self.model = model.to(self.device)
        self.model.eval()
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(
            target_size=target_size,
            gray=gray
        )
        
        # 初始化后处理器
        self.postprocessor = ImagePostprocessor()
        
        # 设置融合策略
        self.set_strategy(strategy)
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_failed': 0
        }
    
    def set_strategy(self, strategy: Union[str, BaseFusionStrategy]):
        """
        设置融合策略
        
        Args:
            strategy: 策略名称或策略实例
                - 'enhanced_l1': 增强版L1融合
                - 'multi_scale': 多尺度融合
                - 'gradient': 梯度引导融合
                - 'hybrid': 混合融合
        
        示例：
        ------
        ```python
        # 使用字符串（自动选择优化版本）
        engine.set_strategy('hybrid')
        
        # 使用实例
        from fusion.strategies import MultiScaleStrategy
        engine.set_strategy(MultiScaleStrategy(scales=[1, 2, 4]))
        
        # 使用优化版本
        from fusion.strategies_optimized import HybridFusionStrategy
        engine.set_strategy(HybridFusionStrategy())
        ```
        """
        if isinstance(strategy, str):
            # 优先使用优化版本
            try:
                from .strategies_optimized import (
                    EnhancedL1Strategy as OptEnhancedL1,
                    MultiScaleStrategy as OptMultiScale,
                    GradientGuidedStrategy as OptGradient,
                    HybridFusionStrategy as OptHybrid
                )
                
                strategy_map = {
                    'enhanced_l1': OptEnhancedL1,
                    'multi_scale': OptMultiScale,
                    'gradient': OptGradient,
                    'hybrid': OptHybrid
                }
                
                if strategy in strategy_map:
                    self.strategy = strategy_map[strategy]()
                else:
                    self.strategy = FusionStrategyRegistry.create(strategy)
            except ImportError:
                self.strategy = FusionStrategyRegistry.create(strategy)
        elif isinstance(strategy, BaseFusionStrategy):
            self.strategy = strategy
        else:
            raise TypeError(
                f"strategy 必须是字符串或 BaseFusionStrategy 实例，"
                f"得到 {type(strategy)}"
            )
        
        print(f"✓ 融合策略: {self.strategy.name}")
    
    def fuse(
        self,
        ir_input: Union[str, torch.Tensor],
        vi_input: Union[str, torch.Tensor],
        output_path: Optional[str] = None,
        return_tensor: bool = False
    ) -> Optional[torch.Tensor]:
        """
        融合单对红外-可见光图像
        
        Args:
            ir_input: 红外图像（路径或张量）
            vi_input: 可见光图像（路径或张量）
            output_path: 输出路径（可选）
            return_tensor: 是否返回张量（默认False）
        
        Returns:
            torch.Tensor 或 None: 融合结果（如果return_tensor=True）
        
        示例：
        ------
        ```python
        # 从文件融合
        result = engine.fuse('ir.png', 'vi.png', 'output.png')
        
        # 融合并返回张量
        result = engine.fuse('ir.png', 'vi.png', return_tensor=True)
        
        # 直接使用张量
        ir_tensor = torch.randn(1, 3, 1024, 1024)
        vi_tensor = torch.randn(1, 3, 1024, 1024)
        result = engine.fuse(ir_tensor, vi_tensor)
        ```
        """
        try:
            # 预处理
            ir_tensor, original_size = self.preprocessor.preprocess(ir_input)
            vi_tensor, _ = self.preprocessor.preprocess(vi_input)
            
            # 移到设备
            ir_tensor = ir_tensor.to(self.device)
            vi_tensor = vi_tensor.to(self.device)
            
            # 特征提取
            ir_features = self.model.encoder(ir_tensor)
            vi_features = self.model.encoder(vi_tensor)
            
            # 特征融合
            fused_features = self.strategy.fuse(ir_features, vi_features)
            
            # 图像重建
            fused_tensor = self.model.decoder(fused_features)
            
            # 后处理
            fused_tensor = self.postprocessor.postprocess(
                fused_tensor,
                original_size if not isinstance(ir_input, str) else None
            )
            
            # 保存
            if output_path:
                self.postprocessor.process_and_save(
                    fused_tensor,
                    original_size,
                    output_path
                )
            
            # 更新统计
            self.stats['total_processed'] += 1
            
            if return_tensor:
                return fused_tensor
            
            return None
            
        except Exception as e:
            print(f"融合失败: {e}")
            self.stats['total_failed'] += 1
            return None
    
    def batch_fuse(
        self,
        ir_dir: str,
        vi_dir: str,
        output_dir: str,
        fusion_strategy: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        批量融合图像
        
        Args:
            ir_dir: 红外图像目录
            vi_dir: 可见光图像目录
            output_dir: 输出目录
            fusion_strategy: 临时融合策略（可选）
        
        Returns:
            Tuple[int, int]: (成功数量, 失败数量)
        
        示例：
        ------
        ```python
        processed, failed = engine.batch_fuse(
            ir_dir='data/ir',
            vi_dir='data/vi',
            output_dir='output'
        )
        print(f"成功: {processed}, 失败: {failed}")
        ```
        """
        # 临时更改策略
        if fusion_strategy:
            original_strategy = self.strategy
            self.set_strategy(fusion_strategy)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件列表
        ir_files = sorted([
            f for f in os.listdir(ir_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        vi_files = sorted([
            f for f in os.listdir(vi_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # 检查文件数量
        if len(ir_files) != len(vi_files):
            print(f"⚠️ 警告: 图像数量不匹配 ({len(ir_files)} vs {len(vi_files)})")
        
        print(f"开始批量融合: {len(ir_files)} 对图像")
        
        # 批量处理
        processed = 0
        failed = 0
        
        pbar = tqdm(zip(ir_files, vi_files), total=len(ir_files), desc="融合进度")
        
        for ir_file, vi_file in pbar:
            ir_path = os.path.join(ir_dir, ir_file)
            vi_path = os.path.join(vi_dir, vi_file)
            
            # 生成输出文件名
            base_name = os.path.splitext(ir_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            try:
                # 融合
                self.fuse(ir_path, vi_path, output_path)
                processed += 1
                
                pbar.set_postfix({
                    'processed': processed,
                    'failed': failed
                })
                
            except Exception as e:
                failed += 1
                print(f"\n融合失败 {ir_file}: {e}")
        
        # 恢复原始策略
        if fusion_strategy:
            self.set_strategy(original_strategy)
        
        # 打印统计
        self._print_batch_stats(processed, failed, output_dir)
        
        return processed, failed
    
    def get_stats(self) -> Dict:
        """
        获取融合统计信息
        
        Returns:
            dict: 统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'total_failed': 0
        }
    
    def _print_batch_stats(self, processed: int, failed: int, output_dir: str):
        """打印批量融合统计"""
        total = processed + failed
        success_rate = (processed / total * 100) if total > 0 else 0
        
        print(f"\n{'=' * 60}")
        print(f"批量融合完成！")
        print(f"成功处理: {processed}/{total} 对图像")
        print(f"失败数量: {failed}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"输出目录: {output_dir}")
        print(f"融合策略: {self.strategy.name}")
        print(f"{'=' * 60}")
    
    def __repr__(self) -> str:
        """返回引擎的字符串表示"""
        return (
            f"ImageFusionEngine(\n"
            f"  device={self.device},\n"
            f"  strategy={self.strategy.name},\n"
            f"  preprocessor={self.preprocessor},\n"
            f"  postprocessor={self.postprocessor}\n"
            f")"
        )


def create_fusion_engine(
    model_path: str,
    device: str = 'cuda',
    strategy: str = 'enhanced_l1',
    model_name: str = 'DenseFuse',
    input_nc: int = 3,
    output_nc: int = 3,
    gray: bool = False,
    target_size: Tuple[int, int] = (1024, 1024)
) -> ImageFusionEngine:
    """
    创建图像融合引擎的工厂函数
    
    功能：
    -----
    便捷地创建完整的融合引擎：
    1. 加载模型
    2. 设置设备
    3. 配置策略
    4. 返回完整配置的引擎
    
    Args:
        model_path: 模型权重文件路径
        device: 计算设备
        strategy: 融合策略
        model_name: 模型名称
        input_nc: 输入通道数
        output_nc: 输出通道数
        gray: 是否使用灰度模式
        target_size: 目标尺寸
    
    Returns:
        ImageFusionEngine: 配置好的融合引擎
    
    示例：
    ------
    ```python
    engine = create_fusion_engine(
        model_path='path/to/model.pth',
        device='cuda',
        strategy='hybrid',
        gray=False
    )
    
    # 开始融合
    engine.fuse('ir.png', 'vi.png', 'output.png')
    ```
    
    注意：
    ----
    模型必须具有 encoder 和 decoder 属性
    """
    from models import fuse_model
    
    # 创建设备
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = fuse_model(
        model_name=model_name,
        input_nc=input_nc,
        output_nc=output_nc
    )
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(
        model_path,
        map_location=device_obj,
        weights_only=False
    )
    
    # 加载编码器和解码器权重
    if 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    if 'decoder_state_dict' in checkpoint:
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    
    print(f"✓ 模型加载成功: {model_path}")
    
    # 选择融合策略（支持优化版本）
    strategy_obj = _create_strategy(strategy)
    
    print(f"✓ 融合策略: {strategy_obj.name}")
    
    # 创建引擎
    engine = ImageFusionEngine(
        model=model,
        device=device_obj,
        strategy=strategy_obj,
        target_size=target_size,
        gray=gray
    )
    
    return engine


def _create_strategy(strategy_name: str) -> BaseFusionStrategy:
    """
    创建融合策略（支持优化版本）
    
    Args:
        strategy_name: 策略名称
    
    Returns:
        BaseFusionStrategy: 融合策略实例
    """
    # 尝试从优化版本创建
    try:
        from .strategies_optimized import (
            EnhancedL1Strategy as OptEnhancedL1,
            MultiScaleStrategy as OptMultiScale,
            GradientGuidedStrategy as OptGradient,
            HybridFusionStrategy as OptHybrid
        )
        
        strategy_map = {
            'enhanced_l1': OptEnhancedL1,
            'multi_scale': OptMultiScale,
            'gradient': OptGradient,
            'hybrid': OptHybrid
        }
        
        if strategy_name in strategy_map:
            return strategy_map[strategy_name]()
    except ImportError:
        pass
    
    # 回退到原版本
    from .base import FusionStrategyRegistry
    return FusionStrategyRegistry.create(strategy_name)
