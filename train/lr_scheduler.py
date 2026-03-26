# -*- coding: utf-8 -*-
"""
@file name:train/lr_scheduler.py
@desc: 学习率调度器模块 - 包含预热、自适应调整等功能
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块提供多种学习率调度策略，用于优化训练过程：
1. WarmupScheduler：学习率预热调度器
2. LearningRateOptimizer：学习率优化器（包含重新升温和自适应调整）

典型使用场景：
-----------
- 训练初期使用预热策略稳定训练
- 训练中期遇到收敛停滞时重新升温
- 根据loss趋势自适应调整学习率

使用示例：
---------
    # 预热调度器
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, target_lr=2e-4)
    warmup_scheduler.step()
    
    # 学习率优化器
    optimizer = LearningRateOptimizer.re_warmup_and_optimize(
        optimizer=optimizer,
        current_lr=1e-5,
        target_lr=1e-4,
        epochs_to_recover=10
    )
"""

import torch


class WarmupScheduler:
    """
    学习率预热调度器
    
    功能：
    -----
    在训练初期逐步增加学习率，从0线性增长到目标学习率。
    这有助于：
    - 稳定训练初期的不稳定性
    - 避免大学习率导致的梯度爆炸
    - 帮助模型在初期找到更好的局部最优
    
    使用方法：
    ---------
    ```python
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        target_lr=2e-4
    )
    
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()
    ```
    
    Attributes:
    -----------
    optimizer : torch.optim.Optimizer
        优化器实例，用于更新学习率
    warmup_epochs : int
        预热的epoch数
    target_lr : float
        目标学习率（预热结束后的学习率）
    current_epoch : int
        当前epoch计数
        
    Example:
    --------
    ```python
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    warmup = WarmupScheduler(optimizer, warmup_epochs=5, target_lr=2e-4)
    
    for epoch in range(100):
        if epoch < 5:
            warmup.step()
            current_lr = warmup.get_lr()
        else:
            # 使用其他调度器
            pass
    ```
    """
    
    def __init__(self, optimizer, warmup_epochs, target_lr):
        """
        初始化预热调度器
        
        Args:
            optimizer (torch.optim.Optimizer): PyTorch优化器实例
            warmup_epochs (int): 预热epoch数
            target_lr (float): 目标学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        """
        执行一步预热
        
        说明：
        -----
        在预热期间，学习率从0线性增长到目标学习率：
        lr = target_lr * (current_epoch + 1) / warmup_epochs
        
        每调用一次step()，current_epoch增加1
        """
        if self.current_epoch < self.warmup_epochs:
            # 线性预热：lr = target_lr * (epoch + 1) / warmup_epochs
            lr = self.target_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_lr(self):
        """
        获取当前学习率
        
        Returns:
            float: 当前优化器的学习率
        """
        return self.optimizer.param_groups[0]['lr']


class LearningRateOptimizer:
    """
    学习率优化器 - 针对训练中期的收敛停滞问题
    
    功能：
    -----
    提供两种学习率优化策略：
    1. re_warmup_and_optimize：重新升温策略
    2. adaptive_lr_adjustment：自适应学习率调整
    
    解决问题：
    ---------
    - 学习率过低导致收敛停滞（常见于训练50+ epoch后）
    - 需要重新激活模型的学习能力
    - 在继续训练时恢复学习率
    
    适用场景：
    ---------
    - 继续训练时发现loss不再下降
    - 学习率已衰减到极低值（如0.000009）
    - 需要在后期提升EN/AG指标
    
    使用示例：
    ---------
    ```python
    # 场景1：检测到loss停滞，重新升温
    optimizer, warmup_epochs = LearningRateOptimizer.re_warmup_and_optimize(
        optimizer=optimizer,
        current_lr=optimizer.param_groups[0]['lr'],
        target_lr=2e-4 * 0.6,  # 恢复到初始lr的60%
        epochs_to_recover=10
    )
    
    # 场景2：根据loss趋势自适应调整
    new_lr = LearningRateOptimizer.adaptive_lr_adjustment(
        optimizer=optimizer,
        current_lr=optimizer.param_groups[0]['lr'],
        loss_trend=-0.02,  # loss下降2%
        threshold=0.05
    )
    ```
    """
    
    @staticmethod
    def re_warmup_and_optimize(optimizer, current_lr, target_lr, epochs_to_recover=10):
        """
        重新升温并优化学习率
        
        功能：
        -----
        当检测到训练收敛停滞时，临时提高学习率帮助模型跳出局部最优。
        这是一种"跳出局部最优"策略，在保持一定预热效果的同时，
        给模型提供更强的学习动力。
        
        参数：
        -----
        optimizer : torch.optim.Optimizer
            优化器实例
        current_lr : float
            当前学习率（通常很低）
        target_lr : float
            目标学习率（建议恢复到初始学习率的50-80%）
        epochs_to_recover : int
            升温周期（通常5-10个epoch即可）
        
        返回：
        -----
        tuple: (optimizer, warmup_epochs)
            - optimizer: 学习率已更新的优化器
            - warmup_epochs: 重新预热的epoch数
        
        算法原理：
        ---------
        1. 直接将学习率设置为目标值（跳到较高的学习率）
        2. 返回新的预热周期，使学习率从高到低过渡
        3. 这样既有足够的探索能力，又有稳定的收敛过程
        
        使用建议：
        ---------
        - 目标学习率通常设为初始学习率的 50%-80%
        - 升温周期设为 5-10 个epoch
        - 建议在连续 10+ 个epoch loss无明显下降时使用
        
        示例：
        ------
        ```python
        # 当学习率过低时（如0.000009），调用此函数恢复
        optimizer, recovery_epochs = LearningRateOptimizer.re_warmup_and_optimize(
            optimizer=optimizer,
            current_lr=optimizer.param_groups[0]['lr'],
            target_lr=2e-4 * 0.6,  # 恢复到初始lr的60%
            epochs_to_recover=8
        )
        ```
        
        打印输出：
        --------
        函数会打印详细的优化信息，包括：
        - 当前学习率
        - 目标学习率
        - 升温周期
        - 升温策略
        """
        print(f"\n" + "=" * 60)
        print("🔄 学习率优化：重新升温策略")
        print("=" * 60)
        print(f"当前学习率: {current_lr:.8f}")
        print(f"目标学习率: {target_lr:.8f}")
        print(f"升温周期: {epochs_to_recover} epochs")
        print(f"升温策略: 线性预热")
        print("=" * 60 + "\n")
        
        # 重新设置学习率为目标值
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr
        
        return optimizer, epochs_to_recover
    
    @staticmethod
    def adaptive_lr_adjustment(optimizer, current_lr, loss_trend, threshold=0.05):
        """
        自适应学习率调整 - 根据loss趋势动态调整
        
        功能：
        -----
        根据训练过程中loss的变化趋势，智能地调整学习率：
        - 如果loss停滞或上升，增加学习率
        - 如果loss快速下降，适当降低学习率以稳定收敛
        - 如果loss正常下降，保持当前学习率
        
        参数：
        -----
        optimizer : torch.optim.Optimizer
            优化器实例
        current_lr : float
            当前学习率
        loss_trend : float
            loss变化率（负值表示下降，正值表示上升或停滞）
            计算方式：(current_loss - previous_loss) / previous_loss
        threshold : float
            调整阈值（默认0.05，即loss变化小于5%时视为停滞）
        
        返回：
        -----
        float: 调整后的学习率
        
        调整策略详解：
        -------------
        1. Loss停滞或上升 (loss_trend > -threshold):
           → 提高学习率20-50%，帮助跳出局部最优
           → 条件：当前学习率 < 1e-3（避免过高）
        
        2. Loss正常下降 (-threshold <= loss_trend <= -0.1):
           → 保持当前学习率
           → 这是理想的训练状态
        
        3. Loss快速下降 (loss_trend < -0.1):
           → 降低学习率20%，以更稳定的方式收敛
           → 条件：当前学习率 > 1e-5（避免过低）
        
        算法公式：
        ---------
        - 提升学习率: new_lr = current_lr × 1.3
        - 降低学习率: new_lr = current_lr × 0.8
        
        使用建议：
        ---------
        - threshold 设置为 0.03-0.05 效果较好
        - loss_trend 应使用移动平均（如最近5个epoch的平均）
        - 避免在学习初期（< 10 epochs）使用此策略
        
        示例：
        ------
        ```python
        # 假设最近5个epoch的loss变化
        loss_history = [0.5, 0.48, 0.47, 0.46, 0.45]
        loss_trend = (loss_history[-1] - loss_history[0]) / loss_history[0]  # -0.1
        
        new_lr = LearningRateOptimizer.adaptive_lr_adjustment(
            optimizer=optimizer,
            current_lr=optimizer.param_groups[0]['lr'],
            loss_trend=loss_trend,
            threshold=0.05
        )
        ```
        
        注意事项：
        ---------
        - 此函数只调整学习率，不会重置optimizer的状态
        - 调整后会打印详细的调整信息
        - 建议配合学习率预热使用
        """
        if loss_trend > -threshold:
            # Loss停滞或上升，增加学习率
            if current_lr < 1e-3:  # 避免学习率过高
                new_lr = current_lr * 1.3  # 提高30%
                print(f"\n📈 Loss停滞，学习率提升: {current_lr:.6f} → {new_lr:.6f}")
            else:
                new_lr = current_lr  # 学习率已经很高，不继续提升
        elif loss_trend < -0.1:
            # Loss快速下降，适当降低学习率
            if current_lr > 1e-5:  # 避免学习率过低
                new_lr = current_lr * 0.8  # 降低20%
                print(f"\n📉 Loss快速下降，学习率微调: {current_lr:.6f} → {new_lr:.6f}")
            else:
                new_lr = current_lr  # 学习率已经很低，不继续降低
        else:
            new_lr = current_lr  # 保持不变
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr


def create_cosine_annealing_scheduler(optimizer, T_0=20, T_mult=2, eta_min=1e-6):
    """
    创建余弦退火学习率调度器
    
    功能：
    -----
    创建一个余弦退火学习率调度器，具有周期重启功能。
    
    参数：
    -----
    optimizer : torch.optim.Optimizer
        优化器实例
    T_0 : int
        第一个周期的长度（默认20）
    T_mult : int
        周期的倍增因子（默认2）
        - T_0=20, T_mult=2: 周期长度为 20, 40, 80, ...
    eta_min : float
        最小学习率（默认1e-6）
    
    返回：
    -----
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    
    算法原理：
    ---------
    余弦退火：lr(t) = eta_min + (eta_max - eta_min) × (1 + cos(t/T_max)) / 2
    
    其中 t 是当前epoch，T_max 是周期长度。
    
    特点：
    -----
    - 学习率先上升后下降，呈余弦波形
    - 周期性重启时会恢复到较高的学习率
    - 帮助模型在训练后期继续优化
    
    使用示例：
    ---------
    ```python
    lr_scheduler = create_cosine_annealing_scheduler(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    for epoch in range(num_epochs):
        # 训练代码...
        lr_scheduler.step()
    ```
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )
