# -*- coding: utf-8 -*-
"""
@file name:train/loss_weights.py
@desc: 损失权重管理器 - 包含自适应损失权重和EN/AG优化策略
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块提供损失函数的权重管理功能，用于训练过程中的动态权重调整。

主要损失项：
-----------
1. L1 Loss (λ1): 像素级重建，确保输出图像与目标图像一致
2. SSIM Loss (λ2): 结构相似性，保持图像整体结构
3. Gradient Loss (λ3): 边缘和纹理细节保留
4. TV Loss (λ4): Total Variation平滑损失

总损失函数：
-----------
Total_Loss = λ1 × L1_Loss + λ2 × (1 - SSIM) + λ3 × Gradient_Loss + λ4 × TV_Loss

使用示例：
---------
    from train.loss_weights import get_adaptive_loss_weights
    
    # 获取当前epoch的损失权重
    l1_w, ssim_w, grad_w, tv_w = get_adaptive_loss_weights(
        epoch=50,
        total_epochs=120,
        optimize_en_ag=True
    )
"""

import torch
import torch.nn.functional as F


def get_adaptive_loss_weights(epoch, total_epochs, use_balanced_loss=True, use_adaptive_weights=True, 
                              optimize_en_ag=False):
    """
    自适应损失权重计算
    
    功能：
    -----
    根据训练进度和优化目标，动态计算各损失项的权重。
    支持两种模式：
    1. 标准平衡模式：所有损失项归一化到同一量级
    2. EN/AG优化模式：增强梯度损失和TV损失的权重
    
    参数：
    -----
    epoch : int
        当前训练轮次
    total_epochs : int
        总训练轮次
    use_balanced_loss : bool
        是否使用平衡损失权重模式
        - True: 所有损失项归一化到同一量级（推荐）
        - False: 使用原始高权重模式
    use_adaptive_weights : bool
        是否使用自适应权重调整
        - True: 根据训练阶段动态调整权重
        - False: 使用固定权重
    optimize_en_ag : bool
        是否启用EN/AG优化模式
        - True: 增强梯度损失和TV损失的权重，专门优化EN和AG指标
        - False: 使用标准平衡模式
    
    返回：
    -----
    tuple: (l1_weight, ssim_weight, grad_weight, tv_weight)
        - l1_weight: L1损失权重
        - ssim_weight: SSIM损失权重
        - grad_weight: 梯度损失权重
        - tv_weight: TV损失权重
    
    EN/AG优化说明：
    -------------
    EN（边缘强度）：
    - 定义：EN = -Σ p_i log₂ p_i（图像信息熵）
    - 优化方法：增强Gradient Loss权重，保留更多边缘信息
    - 权重设置：从1.0提升到2.5-3.0
    
    AG（平均梯度）：
    - 定义：AG = (1/MN) Σ √((∂f/∂x)² + (∂f/∂y)²)/2
    - 优化方法：增强Gradient Loss + TV Loss
    - 权重设置：Gradient从1.0提升到2.5-3.0，TV从0.3提升到1.0-1.5
    
    TV损失的作用：
    -------------
    - 平滑性约束：减少图像中的噪声和不必要的细节
    - 边缘保护：与梯度损失协同工作
    - 噪声抑制：有效去除融合图像中的噪声和伪影
    - 计算公式：TV_Loss = Σ|x(i+1,j) - x(i,j)| + |x(i,j+1) - x(i,j)|
    
    EN/AG优化模式的权重配置：
    -------------------------
    前期（0-30%）: 
        - L1: 1.0, SSIM: 1.0, Gradient: 1.5, TV: 1.0
        - 目标：建立基础，同时引入梯度意识
    
    中期（30-60%）: 
        - L1: 0.8, SSIM: 0.9, Gradient: 2.5, TV: 1.2
        - 目标：强化梯度学习（EN/AG关键期）
    
    后期（60-100%）: 
        - L1: 0.6, SSIM: 0.8, Gradient: 3.0, TV: 1.5
        - 目标：最大化梯度主导
    
    标准平衡模式的权重配置：
    ------------------------
    前期（0-30%）: 
        - L1: 1.0, SSIM: 1.0, Gradient: 1.0, TV: 0.5
    
    中期（30-70%）: 
        - L1: 0.8, SSIM: 1.0, Gradient: 1.5, TV: 0.8
    
    后期（70-100%）: 
        - L1: 0.6, SSIM: 1.0, Gradient: 2.0, TV: 1.0
    
    使用示例：
    ---------
    ```python
    # EN/AG优化模式
    l1_w, ssim_w, grad_w, tv_w = get_adaptive_loss_weights(
        epoch=50,
        total_epochs=120,
        optimize_en_ag=True
    )
    print(f"Epoch 50 weights: L1={l1_w}, SSIM={ssim_w}, Grad={grad_w}, TV={tv_w}")
    
    # 计算总损失
    loss = (l1_w * l1_loss + 
            ssim_w * ssim_loss + 
            grad_w * grad_loss + 
            tv_w * tv_loss)
    ```
    """
    progress = epoch / total_epochs
    
    if optimize_en_ag:
        # 🎯 EN/AG优化模式：增强梯度相关损失权重
        if use_adaptive_weights:
            if progress < 0.3:
                # 前期：建立基础，同时引入梯度意识
                l1_weight = 1.0
                ssim_weight = 1.0
                grad_weight = 1.5  # ⬆️ 提前增加梯度权重
                tv_weight = 1.0   # ⬆️ 提前增加TV权重
            elif progress < 0.6:
                # 中期：强化梯度学习（EN/AG关键期）
                l1_weight = 0.8
                ssim_weight = 0.9
                grad_weight = 2.5  # ⬆️⬆️ 大幅增强梯度权重
                tv_weight = 1.2    # ⬆️ 增强TV权重
            else:
                # 后期：微调优化（保持梯度主导）
                l1_weight = 0.6
                ssim_weight = 0.8
                grad_weight = 3.0  # ⬆️⬆️⬆️ 最大化梯度权重
                tv_weight = 1.5    # ⬆️ 最大化TV权重
        else:
            # 固定权重（EN/AG优化版）
            l1_weight = 0.8
            ssim_weight = 0.9
            grad_weight = 2.5
            tv_weight = 1.2
    else:
        # 标准平衡模式
        if use_balanced_loss:
            # ✅ 平衡模式：所有损失项归一化到同一量级
            if use_adaptive_weights:
                if progress < 0.3:
                    # 前期：注重像素重建
                    l1_weight = 1.0
                    ssim_weight = 1.0
                    grad_weight = 1.0
                    tv_weight = 0.5  # ⬆️ 从0.3提升到0.5
                elif progress < 0.7:
                    # 中期：平衡优化
                    l1_weight = 0.8
                    ssim_weight = 1.0
                    grad_weight = 1.5  # ⬆️ 适度增加梯度权重
                    tv_weight = 0.8    # ⬆️ 适度增加TV权重
                else:
                    # 后期：注重梯度保持和TV平滑
                    l1_weight = 0.6
                    ssim_weight = 1.0
                    grad_weight = 2.0  # ⬆️ 重点提升梯度损失
                    tv_weight = 1.0    # ⬆️ 提升TV权重
            else:
                # 固定权重（非自适应）
                l1_weight = 0.8
                ssim_weight = 1.0
                grad_weight = 1.2
                tv_weight = 0.8
            
        else:
            # ❌ 非平衡模式：使用原始高权重
            if use_adaptive_weights:
                if progress < 0.3:
                    l1_weight = 1.0
                    ssim_weight = 500
                    grad_weight = 50
                    tv_weight = 30
                elif progress < 0.7:
                    l1_weight = 0.7
                    ssim_weight = 1200
                    grad_weight = 100
                    tv_weight = 60
                else:
                    l1_weight = 0.4
                    ssim_weight = 2000
                    grad_weight = 150
                    tv_weight = 90
            else:
                l1_weight = 0.7
                ssim_weight = 1200
                grad_weight = 100
                tv_weight = 60
    
    return l1_weight, ssim_weight, grad_weight, tv_weight


class LossWeightManager:
    """
    损失权重管理器类
    
    功能：
    -----
    提供更灵活的损失权重管理功能，支持：
    - 动态更新权重
    - 记录权重历史
    - 可视化权重变化
    
    使用示例：
    ---------
    ```python
    manager = LossWeightManager(total_epochs=120, optimize_en_ag=True)
    
    for epoch in range(num_epochs):
        weights = manager.get_weights(epoch)
        manager.update_history(weights)
        # 训练代码...
    
    # 获取权重历史
    history = manager.get_history()
    ```
    """
    
    def __init__(self, total_epochs, use_balanced_loss=True, use_adaptive_weights=True, optimize_en_ag=False):
        """
        初始化损失权重管理器
        
        Args:
            total_epochs (int): 总训练轮次
            use_balanced_loss (bool): 是否使用平衡损失权重模式
            use_adaptive_weights (bool): 是否使用自适应权重调整
            optimize_en_ag (bool): 是否启用EN/AG优化模式
        """
        self.total_epochs = total_epochs
        self.use_balanced_loss = use_balanced_loss
        self.use_adaptive_weights = use_adaptive_weights
        self.optimize_en_ag = optimize_en_ag
        self.history = []
    
    def get_weights(self, epoch):
        """
        获取指定epoch的损失权重
        
        Args:
            epoch (int): 当前epoch
        
        Returns:
            dict: 包含各损失权重的字典
        """
        l1, ssim, grad, tv = get_adaptive_loss_weights(
            epoch=epoch,
            total_epochs=self.total_epochs,
            use_balanced_loss=self.use_balanced_loss,
            use_adaptive_weights=self.use_adaptive_weights,
            optimize_en_ag=self.optimize_en_ag
        )
        
        return {
            'l1': l1,
            'ssim': ssim,
            'grad': grad,
            'tv': tv
        }
    
    def update_history(self, weights):
        """
        更新权重历史记录
        
        Args:
            weights (dict): 当前epoch的权重
        """
        self.history.append(weights.copy())
    
    def get_history(self):
        """
        获取权重历史记录
        
        Returns:
            list: 权重历史记录列表
        """
        return self.history
    
    def plot_weights(self, save_path=None):
        """
        可视化权重变化趋势
        
        Args:
            save_path (str): 保存路径，None时只显示
        
        注意：
        ----
        需要安装matplotlib库
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            epochs = range(len(self.history))
            l1_weights = [h['l1'] for h in self.history]
            ssim_weights = [h['ssim'] for h in self.history]
            grad_weights = [h['grad'] for h in self.history]
            tv_weights = [h['tv'] for h in self.history]
            
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, l1_weights, label='L1', linewidth=2)
            plt.plot(epochs, ssim_weights, label='SSIM', linewidth=2)
            plt.plot(epochs, grad_weights, label='Gradient', linewidth=2)
            plt.plot(epochs, tv_weights, label='TV', linewidth=2)
            
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.title('Loss Weights Over Training', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"权重曲线已保存至: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️ 警告：matplotlib未安装，无法生成权重曲线图")
