# -*- coding: utf-8 -*-
"""
@file name:train/__init__.py
@desc: 训练模块包初始化
@Writer: wokaka209
@Date: 2026-03-26

模块说明：
---------
本包包含红外可见光图像融合模型的训练相关功能，包括：
- 学习率调度（预热、余弦退火、自适应调整）
- 损失权重管理（自适应权重、EN/AG优化）
- 训练器（训练循环、验证、保存checkpoint）
- 回调函数（checkpoint、早停、指标记录）

包结构：
-------
train/
├── __init__.py          # 包初始化
├── lr_scheduler.py      # 学习率调度器
├── loss_weights.py     # 损失权重管理器
├── trainer.py          # 训练器
└── callbacks.py        # 回调函数

使用示例：
---------
```python
# 方式1：使用训练器
from train import Trainer, WarmupScheduler, LearningRateOptimizer
from train.loss_weights import get_adaptive_loss_weights

trainer = Trainer(model, optimizer, train_loader, device)
trainer.train(num_epochs=120)

# 方式2：使用损失权重
l1_w, ssim_w, grad_w, tv_w = get_adaptive_loss_weights(
    epoch=50,
    total_epochs=120,
    optimize_en_ag=True
)

# 方式3：使用回调
from train.callbacks import CheckpointCallback, EarlyStoppingCallback
callbacks = [CheckpointCallback('./checkpoints'), EarlyStoppingCallback(patience=10)]
```
"""

from .lr_scheduler import (
    WarmupScheduler, 
    LearningRateOptimizer,
    create_cosine_annealing_scheduler
)
from .loss_weights import get_adaptive_loss_weights, LossWeightManager
from .trainer import Trainer, create_trainer, create_run_directory
from .callbacks import (
    CheckpointCallback, 
    EarlyStoppingCallback, 
    MetricsLoggerCallback
)

__all__ = [
    # 学习率调度
    'WarmupScheduler',
    'LearningRateOptimizer',
    'create_cosine_annealing_scheduler',
    
    # 损失权重
    'get_adaptive_loss_weights',
    'LossWeightManager',
    
    # 训练器
    'Trainer',
    'create_trainer',
    'create_run_directory',
    
    # 回调函数
    'CheckpointCallback',
    'EarlyStoppingCallback',
    'MetricsLoggerCallback',
]

__version__ = '1.0.0'
__author__ = 'wokaka209'
__email__ = '1325536985@qq.com'

