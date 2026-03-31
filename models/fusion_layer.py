# -*- coding: utf-8 -*-
"""
@file name:fusion_layer.py
@desc: 融合层模块，用于三阶段训练中的阶段三（端到端融合）
@Writer: wokaka209
@Date: 2026-03-30

功能说明：
---------
本模块提供多种融合策略，用于红外和可见光图像特征的融合。
支持加法融合、L1-norm融合等策略。

使用方法：
---------
    from models.fusion_layer import FusionLayer, get_fusion_layer
    
    # 创建融合层
    fusion_layer = get_fusion_layer(strategy='l1_norm')
    
    # 融合特征
    fused_features = fusion_layer(ir_features, vi_features)
"""

import torch
import torch.nn as nn


class AdditionFusion(nn.Module):
    """
    加法融合策略
    
    将红外和可见光特征直接相加进行融合。
    
    公式: F = feature_ir + feature_vis
    """
    
    def __init__(self):
        super().__init__()
        self.name = "addition"
        self.description = "加法融合策略"
    
    def forward(self, feature_ir, feature_vis):
        """
        加法融合
        
        Args:
            feature_ir (torch.Tensor): 红外图像特征
            feature_vis (torch.Tensor): 可见光图像特征
        
        Returns:
            torch.Tensor: 融合后的特征
        """
        return feature_ir + feature_vis


class L1NormFusion(nn.Module):
    """
    L1-norm融合策略
    
    基于特征的L1范数计算融合权重，实现自适应融合。
    
    公式:
        energy_ir = |feature_ir|
        energy_vis = |feature_vis|
        weight_ir = energy_ir / (energy_ir + energy_vis + epsilon)
        weight_vis = energy_vis / (energy_ir + energy_vis + epsilon)
        F = weight_ir * feature_ir + weight_vis * feature_vis
    """
    
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.name = "l1_norm"
        self.description = "L1-norm自适应融合策略"
        self.epsilon = epsilon
    
    def forward(self, feature_ir, feature_vis):
        """
        L1-norm融合
        
        Args:
            feature_ir (torch.Tensor): 红外图像特征
            feature_vis (torch.Tensor): 可见光图像特征
        
        Returns:
            torch.Tensor: 融合后的特征
        """
        # 计算L1范数（特征强度）
        l1_norm_ir = torch.sum(torch.abs(feature_ir), dim=1, keepdim=True)
        l1_norm_vis = torch.sum(torch.abs(feature_vis), dim=1, keepdim=True)
        
        # 计算融合权重
        total_energy = l1_norm_ir + l1_norm_vis + self.epsilon
        weight_ir = l1_norm_ir / total_energy
        weight_vis = l1_norm_vis / total_energy
        
        # 加权融合
        fused_features = weight_ir * feature_ir + weight_vis * feature_vis
        
        return fused_features


class WeightedAverageFusion(nn.Module):
    """
    加权平均融合策略
    
    使用可学习的权重参数进行融合。
    
    公式: F = alpha * feature_ir + (1 - alpha) * feature_vis
    """
    
    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.name = "weighted_average"
        self.description = "加权平均融合策略"
        
        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, feature_ir, feature_vis):
        """
        加权平均融合
        
        Args:
            feature_ir (torch.Tensor): 红外图像特征
            feature_vis (torch.Tensor): 可见光图像特征
        
        Returns:
            torch.Tensor: 融合后的特征
        """
        # 使用sigmoid确保权重在[0, 1]范围内
        weight = torch.sigmoid(self.alpha)
        
        fused_features = weight * feature_ir + (1 - weight) * feature_vis
        
        return fused_features


class FusionLayer(nn.Module):
    """
    融合层包装类
    
    支持多种融合策略的统一接口。
    
    Attributes:
        fusion_strategy (nn.Module): 融合策略模块
    """
    
    def __init__(self, strategy='l1_norm', **kwargs):
        """
        初始化融合层
        
        Args:
            strategy (str): 融合策略，可选 'addition', 'l1_norm', 'weighted_average'
            **kwargs: 融合策略的额外参数
        """
        super().__init__()
        
        self.strategy = strategy
        
        if strategy == 'addition':
            self.fusion_strategy = AdditionFusion()
        elif strategy == 'l1_norm':
            self.fusion_strategy = L1NormFusion(**kwargs)
        elif strategy == 'weighted_average':
            self.fusion_strategy = WeightedAverageFusion(**kwargs)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}. "
                           f"支持的策略: 'addition', 'l1_norm', 'weighted_average'")
    
    def forward(self, feature_ir, feature_vis):
        """
        融合特征
        
        Args:
            feature_ir (torch.Tensor): 红外图像特征
            feature_vis (torch.Tensor): 可见光图像特征
        
        Returns:
            torch.Tensor: 融合后的特征
        """
        return self.fusion_strategy(feature_ir, feature_vis)
    
    def get_strategy_name(self):
        """获取当前融合策略名称"""
        return self.fusion_strategy.name
    
    def get_strategy_description(self):
        """获取当前融合策略描述"""
        return self.fusion_strategy.description


def get_fusion_layer(strategy='l1_norm', **kwargs):
    """
    工厂函数：创建融合层
    
    Args:
        strategy (str): 融合策略，可选 'addition', 'l1_norm', 'weighted_average'
        **kwargs: 融合策略的额外参数
    
    Returns:
        FusionLayer: 融合层实例
    
    使用示例：
    ---------
    >>> fusion_layer = get_fusion_layer(strategy='l1_norm')
    >>> fused = fusion_layer(ir_features, vi_features)
    """
    return FusionLayer(strategy=strategy, **kwargs)


if __name__ == "__main__":
    # 测试融合层
    batch_size = 4
    channels = 64
    height = 32
    width = 32
    
    # 创建测试特征
    ir_features = torch.randn(batch_size, channels, height, width)
    vi_features = torch.randn(batch_size, channels, height, width)
    
    print(f"红外特征形状: {ir_features.shape}")
    print(f"可见光特征形状: {vi_features.shape}")
    
    # 测试加法融合
    print("\n测试加法融合...")
    addition_fusion = get_fusion_layer(strategy='addition')
    fused_addition = addition_fusion(ir_features, vi_features)
    print(f"融合后特征形状: {fused_addition.shape}")
    print(f"融合策略: {addition_fusion.get_strategy_name()}")
    
    # 测试L1-norm融合
    print("\n测试L1-norm融合...")
    l1_fusion = get_fusion_layer(strategy='l1_norm')
    fused_l1 = l1_fusion(ir_features, vi_features)
    print(f"融合后特征形状: {fused_l1.shape}")
    print(f"融合策略: {l1_fusion.get_strategy_name()}")
    
    # 测试加权平均融合
    print("\n测试加权平均融合...")
    weighted_fusion = get_fusion_layer(strategy='weighted_average', init_alpha=0.5)
    fused_weighted = weighted_fusion(ir_features, vi_features)
    print(f"融合后特征形状: {fused_weighted.shape}")
    print(f"融合策略: {weighted_fusion.get_strategy_name()}")
    print(f"可学习权重alpha: {weighted_fusion.fusion_strategy.alpha.item():.4f}")
    
    # 测试反向传播
    print("\n测试反向传播...")
    loss = fused_l1.sum()
    loss.backward()
    print(f"反向传播成功，梯度计算正常")
