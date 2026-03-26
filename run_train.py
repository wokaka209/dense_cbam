# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc: 训练主程序入口 - 模块化版本
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本脚本使用模块化的训练代码，提供清晰的结构和良好的可维护性。

使用方法：
---------
    python run_train.py
    
    # 或带参数
    python run_train.py --num_epochs 120 --optimize_en_ag

模块结构：
---------
train/
├── __init__.py          # 包初始化
├── lr_scheduler.py      # 学习率调度器
├── loss_weights.py     # 损失权重管理器
├── trainer.py          # 训练器
└── callbacks.py        # 回调函数
"""

import os
import sys
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入训练模块
from train import Trainer, create_run_directory
from utils.util_dataset_ir_vi import IrViDataset, image_transform
from utils.util_loss import msssim, gradient_loss, tv_loss
from utils.util_device import device_on
from utils.utils import weights_init
from models import fuse_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='红外可见光图像融合训练 - 模块化版本'
    )
    
    # 数据集参数
    parser.add_argument('--ir_path', type=str, 
                       default='E:/whx_Graduation project/baseline_project/dataset/ir',
                       help='红外图像路径')
    parser.add_argument('--vi_path', type=str,
                       default='E:/whx_Graduation project/baseline_project/dataset/vi',
                       help='可见光图像路径')
    parser.add_argument('--gray', action='store_true', default=False,
                       help='是否使用灰度模式')
    
    # 训练参数
    parser.add_argument('--device', type=str, default=device_on(),
                       help='训练设备')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=120,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='初始学习率')
    parser.add_argument('--resume_path', type=str, default='',
                       help='恢复训练的模型路径')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载线程数')
    
    # 优化参数
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='使用注意力机制')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='学习率预热轮数')
    parser.add_argument('--use_gradient_clipping', type=bool, default=True,
                       help='梯度裁剪')
    
    # 损失函数参数
    parser.add_argument('--use_adaptive_weights', type=bool, default=True,
                       help='自适应权重')
    parser.add_argument('--optimize_en_ag', action='store_true', default=True,
                       help='EN/AG优化模式')
    parser.add_argument('--use_balanced_loss', type=bool, default=True,
                       help='平衡损失模式')
    
    # 学习率参数
    parser.add_argument('--use_lr_decay', type=bool, default=True,
                       help='学习率衰减')
    
    # 输出参数
    parser.add_argument('--output', action='store_true', default=True,
                       help='显示输出')
    
    return parser.parse_args()


def print_config(args):
    """打印训练配置"""
    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    
    print("\n📁 数据集配置:")
    print(f"  - 红外图像: {args.ir_path}")
    print(f"  - 可见光图像: {args.vi_path}")
    print(f"  - 灰度模式: {args.gray}")
    
    print("\n⚙️ 训练参数:")
    print(f"  - 设备: {args.device}")
    print(f"  - 批量大小: {args.batch_size}")
    print(f"  - 训练轮数: {args.num_epochs}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 预热轮数: {args.warmup_epochs}")
    
    print("\n🔧 优化参数:")
    print(f"  - 使用注意力机制: {args.use_attention}")
    print(f"  - 自适应权重: {args.use_adaptive_weights}")
    print(f"  - EN/AG优化: {args.optimize_en_ag}")
    print(f"  - 平衡损失: {args.use_balanced_loss}")
    print(f"  - 学习率衰减: {args.use_lr_decay}")
    print(f"  - 梯度裁剪: {args.use_gradient_clipping}")
    
    print("\n📊 损失函数组合:")
    print("  Total_Loss = λ1×L1 + λ2×(1-SSIM) + λ3×Grad + λ4×TV")
    print(f"  - EN/AG优化模式: {'启用' if args.optimize_en_ag else '禁用'}")
    
    print("=" * 60)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 打印配置
    if args.output:
        print_config(args)
    
    # 创建运行目录
    run_dir, checkpoint_dir, log_dir = create_run_directory(args)
    
    # 加载数据集
    print("\n📦 加载数据集...")
    dataset = IrViDataset(
        ir_path=args.ir_path,
        vi_path=args.vi_path,
        transform=image_transform(gray=args.gray, augment=True),
        gray=args.gray
    )
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"✓ 数据集加载完成: {len(dataset)} 样本")
    
    # 初始化模型
    print("\n🏗️ 初始化模型...")
    model = fuse_model(
        model_name="DenseFuse",
        input_nc=1 if args.gray else 3,
        output_nc=1 if args.gray else 3,
        use_attention=args.use_attention
    )
    model = model.to(args.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载完成: {total_params:,} 参数")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    print(f"✓ 优化器创建完成: AdamW(lr={args.lr})")
    
    # 损失函数
    l1_loss_fn = torch.nn.L1Loss().to(args.device)
    print("✓ 损失函数加载完成")
    
    # 创建训练器
    print("\n🚀 创建训练器...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=args.device,
        loss_fn=l1_loss_fn,
        ssim_loss_fn=msssim,
        grad_loss_fn=gradient_loss,
        tv_loss_fn=tv_loss,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        warmup_epochs=args.warmup_epochs,
        initial_lr=args.lr,
        use_gradient_clipping=args.use_gradient_clipping
    )
    
    # 处理恢复训练
    init_epoch = 0
    if args.resume_path:
        print(f"\n📥 恢复训练: {args.resume_path}")
        
        if not os.path.exists(args.resume_path):
            print(f"❌ 错误: 模型文件不存在")
            return
        
        try:
            checkpoint = torch.load(
                args.resume_path,
                map_location=args.device,
                weights_only=False
            )
            
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch'] + 1
            
            print(f"✓ 模型加载成功")
            print(f"✓ 从epoch {init_epoch} 继续训练")
            
        except Exception as e:
            print(f"❌ 错误: 加载模型失败 - {e}")
            return
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60 + "\n")
    
    results = trainer.train(
        num_epochs=args.num_epochs,
        init_epoch=init_epoch,
        use_adaptive_weights=args.use_adaptive_weights,
        optimize_en_ag=args.optimize_en_ag,
        use_balanced_loss=args.use_balanced_loss,
        use_lr_decay=args.use_lr_decay
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"✓ 最佳损失: {results['best_loss']:.6f}")
    print(f"✓ 训练时间: {results['training_time']:.2f}秒")
    print(f"✓ 训练轮数: {results['num_epochs']}")
    print(f"✓ 模型保存: {checkpoint_dir}")
    print(f"✓ 日志记录: {log_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
