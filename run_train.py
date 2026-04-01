# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc: 两阶段训练主程序入口 - 实现自编码器预训练和CBAM微调
@Writer: wokaka209
@Date: 2026-03-31

功能说明：
---------
本脚本实现两阶段训练流程：
1. 阶段一：自编码器预训练（仅可见光图像，不使用CBAM，损失函数：L1 + SSIM）
2. 阶段二：CBAM微调（红外+可见光图像对，冻结主干权重，损失函数：L1 + SSIM + Gradient + TV）

注意：阶段三已禁用
-----------------
由于使用手动融合策略（addition, l1_norm, hybrid），融合层无参数需要训练，
因此删除了阶段三的训练流程。推理时使用阶段二模型 + 手动融合策略即可。

使用方法：
---------
    # 完整两阶段训练
    python run_train.py --train_all_stages
    
    # 单独训练某个阶段
    python run_train.py --stage 1
    python run_train.py --stage 2 --resume_stage1 ./checkpoints/stage1/best.pth
    
    # 从配置加载训练
    python run_train.py --config train_configs.json --stage 2

融合策略说明：
-----------
推理时支持以下融合策略（在fusion_configs.json中配置）：
- weighted_average: 加权平均融合
- l1_norm: L1范数自适应融合
- hybrid: 混合融合策略（推荐）

损失函数配置：
-----------
每个阶段的损失函数在train_configs.json中独立配置：
- L1 Loss (Pixel Loss): 像素级重建损失
- SSIM Loss: 结构相似性损失
- Gradient Loss: 梯度损失
- TV Loss: 全变分损失

模块结构：
---------
train/
├── __init__.py          # 包初始化
├── lr_scheduler.py      # 学习率调度器
├── loss_weights.py     # 损失权重管理器
├── trainer.py          # 训练器
└── callbacks.py        # 回调函数

models/
├── DenseFuse.py                    # 基础DenseFuse模型
├── DenseFuse_with_fusion.py        # 带融合层的DenseFuse模型
├── fusion_layer.py                # 融合层模块
└── attention_modules.py            # 注意力模块

utils/
├── util_dataset_ir_vi.py    # 红外-可见光数据集
└── util_dataset_single.py   # 单图像数据集
"""

import os
import sys
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入训练模块
from train import Trainer, create_run_directory
from utils.util_dataset_single import SingleImageDataset, single_image_transform
from utils.util_dataset_ir_vi import IrViDataset, image_transform
from utils.util_loss import msssim, gradient_loss, tv_loss
from utils.util_device import device_on
from models import fuse_model, fuse_model_with_fusion_layer
from configs_loader import ConfigLoader, TrainingConfig


def load_config(config_path: str = None):
    """
    从JSON配置文件加载训练配置
    
    Args:
        config_path: 配置文件路径（可选，默认使用train_configs.json）
    
    Returns:
        TrainingConfig: 训练配置对象
    
    功能说明：
    ---------
    从外部JSON配置文件加载训练参数，替代原有的命令行参数解析。
    支持配置文件的热重载和配置验证。
    
    使用示例：
    ---------
    ```python
    # 使用默认配置
    config = load_config()
    
    # 使用自定义配置
    config = load_config('custom_train_config.json')
    
    # 访问配置
    batch_size = config.training['batch_size']
    stage1_epochs = config.stage1['epochs']
    ```
    """
    try:
        config = TrainingConfig(config_path)
        
        if not config.validate():
            raise ValueError("配置文件验证失败")
        
        print("[OK] 配置文件加载成功")
        print(f"[OK] 配置文件路径: {config_path or 'train_configs.json'}")
        
        return config
        
    except FileNotFoundError as e:
        print(f"[ERROR] 配置文件不存在: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON格式错误: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] 配置加载失败: {e}")
        raise


def print_config(config, train_stage: int = 1, train_all: bool = False):
    """
    打印训练配置
    
    Args:
        config: 训练配置对象或字典
        train_stage: 当前训练阶段 (1、2 或 3)
        train_all: 是否执行完整三阶段训练
    """
    print("=" * 100)
    print("三阶段训练配置")
    print("=" * 100)
    
    if isinstance(config, TrainingConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    print("\n[数据集配置]:")
    print(f"  - 红外图像: {ConfigLoader.get_value(config_dict, 'dataset', 'ir_path')}")
    print(f"  - 可见光图像: {ConfigLoader.get_value(config_dict, 'dataset', 'vi_path')}")
    print(f"  - 灰度模式: {ConfigLoader.get_value(config_dict, 'dataset', 'gray')}")
    
    print("\n[训练阶段]:")
    if train_all:
        print(f"  - 执行完整两阶段训练")
        print(f"  - 阶段一: {ConfigLoader.get_value(config_dict, 'stage1', 'epochs')} epochs, lr={ConfigLoader.get_value(config_dict, 'stage1', 'learning_rate')}")
        print(f"  - 阶段二: {ConfigLoader.get_value(config_dict, 'stage2', 'epochs')} epochs, lr={ConfigLoader.get_value(config_dict, 'stage2', 'learning_rate')}")
        print(f"  - 阶段三: 已禁用（使用手动融合策略）")
    else:
        print(f"  - 当前阶段: {train_stage}")
        if train_stage == 1:
            print(f"  - 训练轮数: {ConfigLoader.get_value(config_dict, 'stage1', 'epochs')}")
            print(f"  - 学习率: {ConfigLoader.get_value(config_dict, 'stage1', 'learning_rate')}")
        elif train_stage == 2:
            print(f"  - 训练轮数: {ConfigLoader.get_value(config_dict, 'stage2', 'epochs')}")
            print(f"  - 学习率: {ConfigLoader.get_value(config_dict, 'stage2', 'learning_rate')}")
            print(f"  - 恢复路径: {ConfigLoader.get_value(config_dict, 'stage2', 'resume_stage1_path')}")
    
    print("\n[训练参数]:")
    print(f"  - 设备: {ConfigLoader.get_value(config_dict, 'training', 'device')}")
    print(f"  - 批量大小: {ConfigLoader.get_value(config_dict, 'training', 'batch_size')}")
    print(f"  - 预热轮数: {ConfigLoader.get_value(config_dict, 'optimizer', 'warmup_epochs')}")
    
    print("\n[优化参数]:")
    print(f"  - 自适应权重: {ConfigLoader.get_value(config_dict, 'loss_function', 'use_adaptive_weights')}")
    print(f"  - EN/AG优化: {ConfigLoader.get_value(config_dict, 'loss_function', 'optimize_en_ag')}")
    print(f"  - 平衡损失: {ConfigLoader.get_value(config_dict, 'loss_function', 'use_balanced_loss')}")
    print(f"  - 学习率衰减: {ConfigLoader.get_value(config_dict, 'optimizer', 'use_lr_decay')}")
    print(f"  - 梯度裁剪: {ConfigLoader.get_value(config_dict, 'optimizer', 'use_gradient_clipping')}")
    
    print("\n[损失函数权值]:")
    print(f"  - 权值模式: {'自适应调整' if ConfigLoader.get_value(config_dict, 'loss_function', 'use_adaptive_weights') else '手动设置'}")
    print(f"  - L1 Loss默认权值: {ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'l1_weight')}")
    print(f"  - SSIM Loss默认权值: {ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'ssim_weight')}")
    print(f"  - Gradient Loss默认权值: {ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'grad_weight')}")
    print(f"  - TV Loss默认权值: {ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'tv_weight')}")
    
    print("=" * 100)


def train_stage1(config, resume_path: str = ''):
    """
    阶段一：自编码器预训练
    
    Args:
        config: 训练配置对象
        resume_path: 恢复训练路径（可选）
    
    输入：单张可见光图像
    模型：编码器（不含CBAM）+ 解码器
    目标：重建损失
    可训练参数：全部参数
    """
    print("\n" + "=" * 60)
    print("阶段一：自编码器预训练")
    print("=" * 60)
    
    config_dict = config.to_dict()
    
    base_dir = ConfigLoader.get_value(config_dict, 'training', 'base_dir', default='./runs')
    ir_path = ConfigLoader.get_value(config_dict, 'dataset', 'ir_path')
    vi_path = ConfigLoader.get_value(config_dict, 'dataset', 'vi_path')
    gray = ConfigLoader.get_value(config_dict, 'dataset', 'gray', default=False)
    resize = ConfigLoader.get_value(config_dict, 'dataset', 'resize', default=[256, 256])
    device = ConfigLoader.get_value(config_dict, 'training', 'device')
    batch_size = ConfigLoader.get_value(config_dict, 'training', 'batch_size')
    num_workers = ConfigLoader.get_value(config_dict, 'training', 'num_workers', default=8)
    stage1_epochs = ConfigLoader.get_value(config_dict, 'stage1', 'epochs')
    stage1_lr = ConfigLoader.get_value(config_dict, 'stage1', 'learning_rate')
    warmup_epochs = ConfigLoader.get_value(config_dict, 'optimizer', 'warmup_epochs', default=5)
    use_gradient_clipping = ConfigLoader.get_value(config_dict, 'optimizer', 'use_gradient_clipping', default=True)
    use_adaptive_weights = ConfigLoader.get_value(config_dict, 'loss_function', 'use_adaptive_weights', default=True)
    optimize_en_ag = ConfigLoader.get_value(config_dict, 'loss_function', 'optimize_en_ag', default=True)
    use_balanced_loss = ConfigLoader.get_value(config_dict, 'loss_function', 'use_balanced_loss', default=True)
    use_lr_decay = ConfigLoader.get_value(config_dict, 'optimizer', 'use_lr_decay', default=True)
    l1_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'l1_weight', default=1.0)
    ssim_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'ssim_weight', default=100.0)
    grad_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'grad_weight', default=5.0)
    tv_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'tv_weight', default=0.1)
    
    stage1_dir = os.path.join(base_dir, 'stage1_autoencoder')
    os.makedirs(stage1_dir, exist_ok=True)
    checkpoint_dir = os.path.join(stage1_dir, 'checkpoints')
    log_dir = os.path.join(stage1_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n[加载数据集]（阶段一：仅可见光图像）...")
    dataset = SingleImageDataset(
        image_path=vi_path,
        transform=single_image_transform(resize=resize[0], gray=gray, augment=True),
        gray=gray
    )
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    print(f"[OK] 数据集加载完成: {len(dataset)} 样本")
    
    print("\n[初始化模型]（阶段一：不含CBAM）...")
    model = fuse_model(
        model_name="DenseFuse",
        input_nc=1 if gray else 3,
        output_nc=1 if gray else 3,
        use_attention=False
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] 模型加载完成: {total_params:,} 参数")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage1_lr,
        weight_decay=1e-4
    )
    print(f"[OK] 优化器创建完成: AdamW(lr={stage1_lr})")
    
    l1_loss_fn = torch.nn.L1Loss().to(device)
    print("[OK] 损失函数加载完成")
    
    stage1_loss_config = ConfigLoader.get_value(config_dict, 'stage1', 'loss_config', default={})
    
    l1_loss_enabled = ConfigLoader.get_value(stage1_loss_config, 'l1_loss', 'enabled', default=True)
    ssim_loss_enabled = ConfigLoader.get_value(stage1_loss_config, 'ssim_loss', 'enabled', default=True)
    grad_loss_enabled = ConfigLoader.get_value(stage1_loss_config, 'gradient_loss', 'enabled', default=False)
    tv_loss_enabled = ConfigLoader.get_value(stage1_loss_config, 'tv_loss', 'enabled', default=False)
    
    l1_loss_weight = ConfigLoader.get_value(stage1_loss_config, 'l1_loss', 'weight', default=1.0)
    ssim_loss_weight = ConfigLoader.get_value(stage1_loss_config, 'ssim_loss', 'weight', default=100.0)
    grad_loss_weight = ConfigLoader.get_value(stage1_loss_config, 'gradient_loss', 'weight', default=5.0)
    tv_loss_weight = ConfigLoader.get_value(stage1_loss_config, 'tv_loss', 'weight', default=0.1)
    
    print("\n[损失函数配置]（阶段一）:")
    print(f"  L1 Loss: {'启用' if l1_loss_enabled else '禁用'} (weight={l1_loss_weight})")
    print(f"  SSIM Loss: {'启用' if ssim_loss_enabled else '禁用'} (weight={ssim_loss_weight})")
    print(f"  Gradient Loss: {'启用' if grad_loss_enabled else '禁用'} (weight={grad_loss_weight})")
    print(f"  TV Loss: {'启用' if tv_loss_enabled else '禁用'} (weight={tv_loss_weight})")
    
    grad_loss_fn_used = gradient_loss if grad_loss_enabled else None
    tv_loss_fn_used = tv_loss if tv_loss_enabled else None
    
    print("\n[创建训练器]...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        loss_fn=l1_loss_fn if l1_loss_enabled else None,
        ssim_loss_fn=msssim if ssim_loss_enabled else None,
        grad_loss_fn=grad_loss_fn_used,
        tv_loss_fn=tv_loss_fn_used,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        warmup_epochs=warmup_epochs,
        initial_lr=stage1_lr,
        use_gradient_clipping=use_gradient_clipping
    )
    
    init_epoch = 0
    if resume_path:
        print(f"\n[恢复训练]: {resume_path}")
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(
                    resume_path,
                    map_location=device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                init_epoch = checkpoint['epoch'] + 1
                print(f"[OK] 模型加载成功")
                print(f"[OK] 从epoch {init_epoch} 继续训练")
            except Exception as e:
                print(f"[ERROR] 错误: 加载模型失败 - {e}")
                return None
        else:
            print(f"[ERROR] 错误: 模型文件不存在")
            return None
    
    print("\n" + "=" * 60)
    print("开始训练 - 阶段一")
    print("=" * 60 + "\n")
    
    results = trainer.train(
        num_epochs=stage1_epochs,
        init_epoch=init_epoch,
        use_adaptive_weights=use_adaptive_weights,
        optimize_en_ag=optimize_en_ag,
        use_balanced_loss=use_balanced_loss,
        use_lr_decay=use_lr_decay,
        l1_weight=l1_weight if not use_adaptive_weights else None,
        ssim_weight=ssim_weight if not use_adaptive_weights else None,
        grad_weight=grad_weight if not use_adaptive_weights else None,
        tv_weight=tv_weight if not use_adaptive_weights else None
    )
    
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    print(f"\n[OK] 阶段一最佳模型: {best_model_path}")
    
    return {
        'model_path': best_model_path,
        'best_loss': results['best_loss'],
        'training_time': results['training_time']
    }


def train_stage2(config, resume_stage1_path: str = '', resume_path: str = ''):
    """
    阶段二：CBAM微调
    
    Args:
        config: 训练配置对象
        resume_stage1_path: 阶段一模型路径（用于加载权重）
        resume_path: 恢复训练路径（可选）
    
    输入：单张可见光图像
    模型：编码器（含CBAM）+ 解码器
    目标：重建损失
    可训练参数：仅CBAM模块（冻结主干权重）
    """
    print("\n" + "=" * 60)
    print("阶段二：CBAM微调")
    print("=" * 60)
    
    config_dict = config.to_dict()
    
    if not resume_stage1_path:
        resume_stage1_path = ConfigLoader.get_value(config_dict, 'stage2', 'resume_stage1_path')
    
    if not os.path.exists(resume_stage1_path):
        print(f"[ERROR] 错误: 阶段一模型文件不存在: {resume_stage1_path}")
        return None
    
    base_dir = ConfigLoader.get_value(config_dict, 'training', 'base_dir', default='./runs')
    vi_path = ConfigLoader.get_value(config_dict, 'dataset', 'vi_path')
    resize = ConfigLoader.get_value(config_dict, 'dataset', 'resize', default=[256, 256])
    ir_path = ConfigLoader.get_value(config_dict, 'dataset', 'ir_path')
    gray = ConfigLoader.get_value(config_dict, 'dataset', 'gray', default=False)
    
    device = ConfigLoader.get_value(config_dict, 'training', 'device')
    batch_size = ConfigLoader.get_value(config_dict, 'training', 'batch_size')
    num_workers = ConfigLoader.get_value(config_dict, 'training', 'num_workers', default=8)
    stage2_epochs = ConfigLoader.get_value(config_dict, 'stage2', 'epochs')
    stage2_lr = ConfigLoader.get_value(config_dict, 'stage2', 'learning_rate')
    warmup_epochs = ConfigLoader.get_value(config_dict, 'optimizer', 'warmup_epochs', default=5)
    use_gradient_clipping = ConfigLoader.get_value(config_dict, 'optimizer', 'use_gradient_clipping', default=True)
    use_adaptive_weights = ConfigLoader.get_value(config_dict, 'loss_function', 'use_adaptive_weights', default=True)
    optimize_en_ag = ConfigLoader.get_value(config_dict, 'loss_function', 'optimize_en_ag', default=True)
    use_balanced_loss = ConfigLoader.get_value(config_dict, 'loss_function', 'use_balanced_loss', default=True)
    use_lr_decay = ConfigLoader.get_value(config_dict, 'optimizer', 'use_lr_decay', default=True)
    l1_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'l1_weight', default=1.0)
    ssim_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'ssim_weight', default=100.0)
    grad_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'grad_weight', default=5.0)
    tv_weight = ConfigLoader.get_value(config_dict, 'loss_function', 'weights', 'tv_weight', default=0.1)
    
    stage2_dir = os.path.join(base_dir, 'stage2_cbam')
    os.makedirs(stage2_dir, exist_ok=True)
    checkpoint_dir = os.path.join(stage2_dir, 'checkpoints')
    log_dir = os.path.join(stage2_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n[加载数据集]（阶段二：红外+可见光图像对，CBAM微调）...")
    dataset = IrViDataset(
        ir_path=ir_path,
        vi_path=vi_path,
        transform=image_transform(resize=resize[0], gray=gray, augment=True),
        gray=gray
    )
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    print(f"[OK] 数据集加载完成: {len(dataset)} 样本")
    
    print("\n[初始化模型]（阶段二：含CBAM，双输入）...")
    model = fuse_model_with_fusion_layer(
        model_name="DenseFuse",
        input_nc=1 if gray else 3,
        output_nc=1 if gray else 3,
        use_attention=True,
        fusion_strategy='addition'  # 阶段二使用简单的加法融合
    )
    model = model.to(device)
    
    print(f"\n[加载阶段一权重]: {resume_stage1_path}")
    try:
        checkpoint = torch.load(resume_stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        print("[OK] 阶段一权重加载成功")
    except Exception as e:
        print(f"[ERROR] 错误: 加载阶段一权重失败 - {e}")
        return None
    
    print("\n[冻结主干权重]，仅训练CBAM模块...")
    model.freeze_backbone()
    model.unfreeze_cbam()
    
    trainable_params = model.get_trainable_params()
    cbam_params = model.get_cbam_params()
    print(f"[OK] 可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"[OK] CBAM参数数量: {sum(p.numel() for p in cbam_params):,}")
    
    optimizer = torch.optim.AdamW(
        cbam_params,
        lr=stage2_lr,
        weight_decay=1e-4
    )
    print(f"[OK] 优化器创建完成: AdamW(lr={stage2_lr})")
    
    l1_loss_fn = torch.nn.L1Loss().to(device)
    print("[OK] 损失函数加载完成")
    
    stage2_loss_config = ConfigLoader.get_value(config_dict, 'stage2', 'loss_config', default={})
    
    l1_loss_enabled = ConfigLoader.get_value(stage2_loss_config, 'l1_loss', 'enabled', default=True)
    ssim_loss_enabled = ConfigLoader.get_value(stage2_loss_config, 'ssim_loss', 'enabled', default=True)
    grad_loss_enabled = ConfigLoader.get_value(stage2_loss_config, 'gradient_loss', 'enabled', default=False)
    tv_loss_enabled = ConfigLoader.get_value(stage2_loss_config, 'tv_loss', 'enabled', default=False)
    
    l1_loss_weight = ConfigLoader.get_value(stage2_loss_config, 'l1_loss', 'weight', default=1.0)
    ssim_loss_weight = ConfigLoader.get_value(stage2_loss_config, 'ssim_loss', 'weight', default=100.0)
    grad_loss_weight = ConfigLoader.get_value(stage2_loss_config, 'gradient_loss', 'weight', default=5.0)
    tv_loss_weight = ConfigLoader.get_value(stage2_loss_config, 'tv_loss', 'weight', default=0.1)
    
    print("\n[损失函数配置]（阶段二）:")
    print(f"  L1 Loss: {'启用' if l1_loss_enabled else '禁用'} (weight={l1_loss_weight})")
    print(f"  SSIM Loss: {'启用' if ssim_loss_enabled else '禁用'} (weight={ssim_loss_weight})")
    print(f"  Gradient Loss: {'启用' if grad_loss_enabled else '禁用'} (weight={grad_loss_weight})")
    print(f"  TV Loss: {'启用' if tv_loss_enabled else '禁用'} (weight={tv_loss_weight})")
    
    grad_loss_fn_used = gradient_loss if grad_loss_enabled else None
    tv_loss_fn_used = tv_loss if tv_loss_enabled else None
    
    print("\n[创建训练器]...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        loss_fn=l1_loss_fn if l1_loss_enabled else None,
        ssim_loss_fn=msssim if ssim_loss_enabled else None,
        grad_loss_fn=grad_loss_fn_used,
        tv_loss_fn=tv_loss_fn_used,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        warmup_epochs=warmup_epochs,
        initial_lr=stage2_lr,
        use_gradient_clipping=use_gradient_clipping
    )
    
    init_epoch = 0
    if resume_path:
        print(f"\n[恢复训练]: {resume_path}")
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(
                    resume_path,
                    map_location=device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                init_epoch = checkpoint['epoch'] + 1
                print(f"[OK] 模型加载成功")
                print(f"[OK] 从epoch {init_epoch} 继续训练")
            except Exception as e:
                print(f"[ERROR] 错误: 加载模型失败 - {e}")
                return None
        else:
            print(f"[ERROR] 错误: 模型文件不存在")
            return None
    
    print("\n" + "=" * 60)
    print("开始训练 - 阶段二")
    print("=" * 60 + "\n")
    
    results = trainer.train(
        num_epochs=stage2_epochs,
        init_epoch=init_epoch,
        use_adaptive_weights=use_adaptive_weights,
        optimize_en_ag=optimize_en_ag,
        use_balanced_loss=use_balanced_loss,
        use_lr_decay=use_lr_decay,
        l1_weight=l1_weight if not use_adaptive_weights else None,
        ssim_weight=ssim_weight if not use_adaptive_weights else None,
        grad_weight=grad_weight if not use_adaptive_weights else None,
        tv_weight=tv_weight if not use_adaptive_weights else None
    )
    
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    print(f"\n[OK] 阶段二最佳模型: {best_model_path}")
    
    return {
        'model_path': best_model_path,
        'best_loss': results['best_loss'],
        'training_time': results['training_time']
    }


def train_stage3(config, resume_stage2_path: str = '', resume_path: str = ''):
    """
    阶段三：融合层训练
    
    Args:
        config: 训练配置对象
        resume_stage2_path: 阶段二模型路径（用于加载权重）
        resume_path: 恢复训练路径（可选）
    
    输入：红外+可见光图像对
    模型：编码器（含CBAM）+ 融合层 + 解码器
    目标：融合损失
    可训练参数：Decoder + Fusion Layer（冻结Encoder和CBAM）
    """
    print("\n" + "=" * 60)
    print("阶段三：融合层训练")
    print("=" * 60)
    
    config_dict = config.to_dict()
    
    if not resume_stage2_path:
        resume_stage2_path = ConfigLoader.get_value(config_dict, 'stage3', 'resume_stage2_path')
    
    if not os.path.exists(resume_stage2_path):
        print(f"[ERROR] 错误: 阶段二模型文件不存在: {resume_stage2_path}")
        return None
    
    base_dir = ConfigLoader.get_value(config_dict, 'training', 'base_dir', default='./runs')
    ir_path = ConfigLoader.get_value(config_dict, 'dataset', 'ir_path')
    vi_path = ConfigLoader.get_value(config_dict, 'dataset', 'vi_path')
    gray = ConfigLoader.get_value(config_dict, 'dataset', 'gray', default=False)
    resize = ConfigLoader.get_value(config_dict, 'dataset', 'resize', default=[256, 256])
    device = ConfigLoader.get_value(config_dict, 'training', 'device')
    batch_size = ConfigLoader.get_value(config_dict, 'training', 'batch_size')
    num_workers = ConfigLoader.get_value(config_dict, 'training', 'num_workers', default=8)
    stage3_epochs = ConfigLoader.get_value(config_dict, 'stage3', 'epochs')
    stage3_lr = ConfigLoader.get_value(config_dict, 'stage3', 'learning_rate')
    warmup_epochs = ConfigLoader.get_value(config_dict, 'optimizer', 'warmup_epochs', default=5)
    use_gradient_clipping = ConfigLoader.get_value(config_dict, 'optimizer', 'use_gradient_clipping', default=True)
    use_adaptive_weights = ConfigLoader.get_value(config_dict, 'loss_function', 'use_adaptive_weights', default=False)
    optimize_en_ag = ConfigLoader.get_value(config_dict, 'loss_function', 'optimize_en_ag', default=True)
    use_balanced_loss = ConfigLoader.get_value(config_dict, 'loss_function', 'use_balanced_loss', default=False)
    use_lr_decay = ConfigLoader.get_value(config_dict, 'optimizer', 'use_lr_decay', default=True)
    
    fusion_strategy = ConfigLoader.get_value(config_dict, 'stage3', 'fusion_config', 'strategy', default='l1_norm')
    
    stage3_dir = os.path.join(base_dir, 'stage3_fusion')
    os.makedirs(stage3_dir, exist_ok=True)
    checkpoint_dir = os.path.join(stage3_dir, 'checkpoints')
    log_dir = os.path.join(stage3_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n[加载数据集]（阶段三：红外+可见光图像对）...")
    dataset = IrViDataset(
        ir_path=ir_path,
        vi_path=vi_path,
        transform=image_transform(resize=resize[0], gray=gray, augment=True),
        gray=gray
    )
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    print(f"[OK] 数据集加载完成: {len(dataset)} 样本")
    
    print(f"\n[初始化模型]（阶段三：Decoder + Fusion Layer）...")
    model = fuse_model_with_fusion_layer(
        model_name="DenseFuse",
        input_nc=1 if gray else 3,
        output_nc=1 if gray else 3,
        use_attention=True,
        fusion_strategy=fusion_strategy
    )
    model = model.to(device)
    
    print(f"\n[加载阶段二权重]: {resume_stage2_path}")
    try:
        checkpoint = torch.load(resume_stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        print("[OK] 阶段二权重加载成功")
    except Exception as e:
        print(f"[ERROR] 错误: 加载阶段二权重失败 - {e}")
        return None
    
    print("\n[冻结Encoder和CBAM，仅训练Decoder + Fusion Layer]...")
    model.freeze_encoder()
    model.freeze_cbam()
    model.unfreeze_decoder()
    model.unfreeze_fusion_layer()
    
    trainable_params = model.get_trainable_params()
    decoder_params = model.get_decoder_params()
    fusion_params = model.get_fusion_params()
    print(f"[OK] 可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"[OK] Decoder参数数量: {sum(p.numel() for p in decoder_params):,}")
    print(f"[OK] Fusion Layer参数数量: {sum(p.numel() for p in fusion_params):,}")
    
    all_trainable_params = list(decoder_params) + list(fusion_params)
    optimizer = torch.optim.AdamW(
        all_trainable_params,
        lr=stage3_lr,
        weight_decay=1e-4
    )
    print(f"[OK] 优化器创建完成: AdamW(lr={stage3_lr})")
    
    l1_loss_fn = torch.nn.L1Loss().to(device)
    print("[OK] 损失函数加载完成")
    
    stage3_loss_config = ConfigLoader.get_value(config_dict, 'stage3', 'loss_config', default={})
    
    l1_loss_enabled = ConfigLoader.get_value(stage3_loss_config, 'l1_loss', 'enabled', default=True)
    ssim_loss_enabled = ConfigLoader.get_value(stage3_loss_config, 'ssim_loss', 'enabled', default=True)
    grad_loss_enabled = ConfigLoader.get_value(stage3_loss_config, 'gradient_loss', 'enabled', default=True)
    tv_loss_enabled = ConfigLoader.get_value(stage3_loss_config, 'tv_loss', 'enabled', default=True)
    
    l1_loss_weight = ConfigLoader.get_value(stage3_loss_config, 'l1_loss', 'weight', default=1.0)
    ssim_loss_weight = ConfigLoader.get_value(stage3_loss_config, 'ssim_loss', 'weight', default=100.0)
    grad_loss_weight = ConfigLoader.get_value(stage3_loss_config, 'gradient_loss', 'weight', default=5.0)
    tv_loss_weight = ConfigLoader.get_value(stage3_loss_config, 'tv_loss', 'weight', default=0.1)
    
    print(f"\n[融合策略配置]: {fusion_strategy}")
    print("\n[损失函数配置]（阶段三）:")
    print(f"  L1 Loss: {'启用' if l1_loss_enabled else '禁用'} (weight={l1_loss_weight})")
    print(f"  SSIM Loss: {'启用' if ssim_loss_enabled else '禁用'} (weight={ssim_loss_weight})")
    print(f"  Gradient Loss: {'启用' if grad_loss_enabled else '禁用'} (weight={grad_loss_weight})")
    print(f"  TV Loss: {'启用' if tv_loss_enabled else '禁用'} (weight={tv_loss_weight})")
    
    grad_loss_fn_used = gradient_loss if grad_loss_enabled else None
    tv_loss_fn_used = tv_loss if tv_loss_enabled else None
    
    print("\n[创建训练器]...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        loss_fn=l1_loss_fn if l1_loss_enabled else None,
        ssim_loss_fn=msssim if ssim_loss_enabled else None,
        grad_loss_fn=grad_loss_fn_used,
        tv_loss_fn=tv_loss_fn_used,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        warmup_epochs=warmup_epochs,
        initial_lr=stage3_lr,
        use_gradient_clipping=use_gradient_clipping
    )
    
    init_epoch = 0
    if resume_path:
        print(f"\n[恢复训练]: {resume_path}")
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(
                    resume_path,
                    map_location=device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                init_epoch = checkpoint['epoch'] + 1
                print(f"[OK] 模型加载成功")
                print(f"[OK] 从epoch {init_epoch} 继续训练")
            except Exception as e:
                print(f"[ERROR] 错误: 加载模型失败 - {e}")
                return None
        else:
            print(f"[ERROR] 错误: 模型文件不存在")
            return None
    
    print("\n" + "=" * 60)
    print("开始训练 - 阶段三")
    print("=" * 60 + "\n")
    
    results = trainer.train(
        num_epochs=stage3_epochs,
        init_epoch=init_epoch,
        use_adaptive_weights=use_adaptive_weights,
        optimize_en_ag=optimize_en_ag,
        use_balanced_loss=use_balanced_loss,
        use_lr_decay=use_lr_decay,
        l1_weight=l1_loss_weight if not use_adaptive_weights else None,
        ssim_weight=ssim_loss_weight if not use_adaptive_weights else None,
        grad_weight=grad_loss_weight if not use_adaptive_weights else None,
        tv_weight=tv_loss_weight if not use_adaptive_weights else None
    )
    
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    print(f"\n[OK] 阶段三最佳模型: {best_model_path}")
    
    return {
        'model_path': best_model_path,
        'best_loss': results['best_loss'],
        'training_time': results['training_time']
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='两阶段红外可见光图像融合训练')
    parser.add_argument('--train_all_stages', action='store_true', default=False,
                       help='执行完整的两阶段训练')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='选择训练阶段 (1: 自编码器预训练, 2: CBAM微调)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选，默认使用train_configs.json）')
    parser.add_argument('--resume_stage1', type=str, default='',
                       help='阶段一模型路径（用于阶段二）')
    parser.add_argument('--resume_path', type=str, default='',
                       help='恢复训练的模型路径（可选）')
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print_config(config, train_stage=args.stage, train_all=args.train_all_stages)
        
        base_dir = ConfigLoader.get_value(config.to_dict(), 'training', 'base_dir', default='./runs')
        os.makedirs(base_dir, exist_ok=True)
        
        if args.train_all_stages:
            print("\n[开始完整三阶段训练]...")
            
            stage1_result = train_stage1(config)
            if stage1_result is None:
                print("\n[ERROR] 阶段一训练失败")
                return
            
            stage2_result = train_stage2(config, resume_stage1_path=stage1_result['model_path'])
            if stage2_result is None:
                print("\n[ERROR] 阶段二训练失败")
                return
            
            print("\n" + "=" * 60)
            print("两阶段训练完成（阶段三已禁用）")
            print("=" * 60)
            print(f"[OK] 阶段一最佳损失: {stage1_result['best_loss']:.6f}")
            print(f"[OK] 阶段二最佳损失: {stage2_result['best_loss']:.6f}")
            print(f"[OK] 总训练时间: {stage1_result['training_time'] + stage2_result['training_time']:.2f}秒")
            print(f"[OK] 最终模型: {stage2_result['model_path']}")
            print(f"[INFO] 使用手动融合策略（hybrid/l1_norm/weighted_average）")
            print("=" * 60)
            
        else:
            print(f"\n[开始阶段 {args.stage} 训练]...")
            
            if args.stage == 1:
                result = train_stage1(config, resume_path=args.resume_path)
            elif args.stage == 2:
                resume_stage1 = args.resume_stage1 if args.resume_stage1 else ''
                result = train_stage2(config, resume_stage1_path=resume_stage1, resume_path=args.resume_path)
            else:
                print("\n[ERROR] 阶段三已禁用（使用手动融合策略，无需训练）")
                print("[INFO] 请使用阶段二的模型进行推理")
                return
            
            if result is None:
                print(f"\n[ERROR] 阶段 {args.stage} 训练失败")
                return
            
            print("\n" + "=" * 60)
            print(f"阶段 {args.stage} 训练完成")
            print("=" * 60)
            print(f"[OK] 最佳损失: {result['best_loss']:.6f}")
            print(f"[OK] 训练时间: {result['training_time']:.2f}秒")
            print(f"[OK] 模型保存: {result['model_path']}")
            print("=" * 60)
    
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
