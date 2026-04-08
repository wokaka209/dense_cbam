# -*- coding: utf-8 -*-
"""
@file name:train_ir_vi_optimized.py
@desc: 优化版训练脚本 - 专门解决EN、AG、MI、Qabf指标低的问题
@Writer: wokaka209
@Date: 2026-03-13
"""

import os
import time
import argparse
from torch.utils.data import DataLoader
from utils.utils import create_run_directory, weights_init
from utils.util_dataset_ir_vi import IrViDataset, image_transform
from utils.util_train import train_epoch
from utils.util_loss import *
from models import fuse_model
from utils.util_device import device_on
from torch.utils.tensorboard import SummaryWriter
import torch
import glob
import shutil

# 混合精度训练支持
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast


def cleanup_old_checkpoints(checkpoint_dir, keep_best=True, keep_last=True):
    """
    清理旧的checkpoint文件，只保留best.pth和last.pth
    
    Args:
        checkpoint_dir: checkpoint保存目录
        keep_best: 是否保留best.pth
        keep_last: 是否保留last.pth
    
    功能：
        - 删除所有epoch*.pth格式的旧文件
        - 保留best.pth（最佳模型）
        - 保留last.pth（最新模型）
        - 处理文件权限和不存在的情况
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # 要保留的文件列表
    files_to_keep = []
    if keep_best:
        files_to_keep.append('best.pth')
    if keep_last:
        files_to_keep.append('last.pth')
    
    # 获取所有.pth文件
    all_pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    
    deleted_count = 0
    for filepath in all_pth_files:
        filename = os.path.basename(filepath)
        
        # 跳过要保留的文件
        if filename in files_to_keep:
            continue
        
        # 删除所有其他.pth文件（包括epoch*.pth和其他文件）
        try:
            os.remove(filepath)
            deleted_count += 1
        except PermissionError:
            print(f'Warning: 文件权限不足，无法删除: {filename}')
        except OSError as e:
            print(f'Warning: 删除文件失败: {filename}, 错误: {e}')
    
    if deleted_count > 0:
        print(f'[Cleanup] 已清理 {deleted_count} 个旧checkpoint文件')


def parse_args():
    parser = argparse.ArgumentParser(description="优化版红外和可见光图像融合模型训练参数")
    
    # 数据集相关参数
    parser.add_argument('--ir_path', default='E:/whx_Graduation project/baseline_project/dataset/ir', type=str, help='红外图像路径')
    parser.add_argument('--vi_path', default='E:/whx_Graduation project/baseline_project/dataset/vi', type=str, help='可见光图像路径')
    parser.add_argument('--gray', action='store_true', default=False, help='是否使用灰度模式')
    
    # 训练相关参数（优化版）
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=35, help='number of epochs to train for（优化版：25）')
    parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率（优化版：2e-4）')
    parser.add_argument('--resume_path', default='', type=str, help='导入已训练好的模型路径')
    parser.add_argument('--num_workers', type=int, default=4, help='载入数据集所调用的cpu线程数')
    
    # 优化参数
    parser.add_argument('--cbam_scheme', type=int, default=0, choices=[0, 1, 2], 
                        help='CBAM实施方案选择: 0=不使用CBAM, 1=DenseBlock输出后CBAM, 2=融合层输入前CBAM')
    parser.add_argument('--reduction_ratio', type=int, default=2, choices=[2, 8, 16, 32, 64, 128], 
                        help='CBAM通道压缩比例: 16, 32, 64, 128等')
    parser.add_argument('--use_color_aware', action='store_true', default=True, 
                        help='是否使用颜色感知CBAM（解决泛黄问题）')
    parser.add_argument('--color_preservation_weight', type=float, default=0.5, 
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='颜色保护权重（0.0-1.0），推荐0.5')
    # CBAM消融实验参数（仅在CBAM方案不为0时有效）
    parser.add_argument('--use_channel_attention', action='store_true', default=True,
                        help='是否启用通道注意力（仅在CBAM方案不为0时有效）')
    parser.add_argument('--use_spatial_attention', action='store_true', default=True,
                        help='是否启用空间注意力（仅在CBAM方案不为0时有效）')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help='是否使用混合精度训练')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='学习率预热epoch数')
    
    # 多尺度梯度损失参数（仅在启用多尺度梯度时有效）
    parser.add_argument('--use_multiscale_gradient', action='store_true', default=True, 
                        help='是否启用多尺度梯度损失函数')
    parser.add_argument('--gradient_scales', type=int, default=4, choices=[2, 3, 4, 5], 
                        help='多尺度梯度计算的尺度数量（仅在启用多尺度梯度时有效）')
    parser.add_argument('--gradient_weight', type=float, default=1.5, 
                        help='多尺度梯度损失的权重系数（仅在启用多尺度梯度时有效）')
    parser.add_argument('--gradient_alpha', type=float, default=1.5, 
                        help='水平梯度权重（方向感知参数，仅在启用多尺度梯度时有效）')
    parser.add_argument('--gradient_beta', type=float, default=1.0, 
                        help='垂直梯度权重（方向感知参数，仅在启用多尺度梯度时有效）')
    # 梯度方向消融实验参数（仅在启用多尺度梯度时有效）
    parser.add_argument('--gradient_direction', type=str, default='bidirectional', choices=['single', 'bidirectional', 'none'],
                        help='梯度方向选择: single=单方向梯度损失, bidirectional=双向梯度损失（仅在启用多尺度梯度时有效）')
    
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    
    args = parser.parse_args()

    if args.output:
        print("==========优化版训练参数==========")
        print("----------数据集相关参数----------")
        print(f'ir_path: {args.ir_path}')
        print(f'vi_path: {args.vi_path}')
        print(f'gray_images: {args.gray}')

        print("----------训练相关参数----------")
        print(f'device: {args.device}')
        print(f'batch_size: {args.batch_size}')
        print(f'num_epochs: {args.num_epochs}（优化版：80个epoch）')
        print(f'num_workers: {args.num_workers}')
        print(f'initial learning rate: {args.lr}')
        print(f'resume_path: {args.resume_path}')
        
        print("----------优化选项----------")
        print(f'cbam_scheme: {args.cbam_scheme}')
        if args.cbam_scheme == 0:
            print('  └─ 不使用CBAM注意力机制')
        elif args.cbam_scheme == 1:
            print('  └─ 方案1：DenseBlock输出后应用CBAM')
        elif args.cbam_scheme == 2:
            print('  └─ 方案2：融合层输入前应用CBAM')
        print(f'reduction_ratio: {args.reduction_ratio}')
        print(f'use_color_aware: {args.use_color_aware}')
        if args.use_color_aware:
            print(f'  └─ 颜色保护权重: {args.color_preservation_weight}')
        print(f'use_mixed_precision: {args.use_mixed_precision}')
        print(f'warmup_epochs: {args.warmup_epochs}')
        # CBAM消融实验参数
        print(f'use_channel_attention: {args.use_channel_attention}')
        print(f'use_spatial_attention: {args.use_spatial_attention}')
        
        print("----------多尺度梯度损失参数----------")
        print(f'use_multiscale_gradient: {args.use_multiscale_gradient}')
        if args.use_multiscale_gradient:
            print(f'gradient_scales: {args.gradient_scales}')
            print(f'gradient_weight: {args.gradient_weight}')
            print(f'gradient_alpha: {args.gradient_alpha}')
            print(f'gradient_beta: {args.gradient_beta}')
            print(f'gradient_direction: {args.gradient_direction}')
    return args


class WarmupScheduler:
    """学习率预热调度器"""
    def __init__(self, optimizer, warmup_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.target_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def get_adaptive_loss_weights(epoch, total_epochs, use_multiscale_gradient=False):
    """
    自适应损失权重 - 专门优化EN、AG、MI指标
    
    策略：
    - 前期（0-30%）：注重像素重建（MSE权重高）
    - 中期（30-70%）：平衡像素和结构
    - 后期（70-100%）：注重结构保持和细节（SSIM和梯度权重高）
    """
    progress = epoch / total_epochs
    
    if progress < 0.3:
        # 前期：注重像素重建
        mse_weight = 1.0
        ssim_weight = 100
        gradient_weight = 1.0 if use_multiscale_gradient else 0.0
    elif progress < 0.7:
        # 中期：平衡
        mse_weight = 1.0
        ssim_weight = 100
        gradient_weight = 1.5 if use_multiscale_gradient else 0.0
    elif progress < 1.0:
        # 后期：注重结构保持和细节
        mse_weight = 1.0
        ssim_weight = 100
        gradient_weight = 2.0 if use_multiscale_gradient else 0.0
    
    return mse_weight, ssim_weight, gradient_weight


if __name__ == "__main__":
    print("==================优化版模型训练==================")
    args = parse_args()
    run_dir, checkpoint_dir, log_dir = create_run_directory(args)
    print("==================优化版模型训练==================")
    
    # ----------------------------------------------------#
    #           数据集（启用数据增强）
    # ----------------------------------------------------#
    dataset = IrViDataset(
        ir_path=args.ir_path, 
        vi_path=args.vi_path, 
        transform=image_transform(gray=args.gray, augment=True),  # 启用数据增强
        gray=args.gray
    )
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    print('训练数据载入完成...')

    # ----------------------------------------------------#
    #           device
    # ----------------------------------------------------#
    device = args.device
    print("设备就绪...")
    
    # ----------------------------------------------------#
    #           网络模型（支持两种CBAM方案）
    # ----------------------------------------------------#
    model_name = "DenseFuse"
    in_channel = 1 if args.gray else 3
    out_channel = 1 if args.gray else 3
    
    # 导入新的模型类
    from models.DenseFuse import DenseFuse_train
    
    model_train = DenseFuse_train(
        input_nc=in_channel, 
        output_nc=out_channel, 
        cbam_scheme=args.cbam_scheme,
        reduction_ratio=args.reduction_ratio,
        use_color_aware=args.use_color_aware,
        color_preservation_weight=args.color_preservation_weight,
        use_channel_attention=args.use_channel_attention,
        use_spatial_attention=args.use_spatial_attention
    )
    model_train.to(device)
    print(f'模型参数量: {sum(p.numel() for p in model_train.parameters()):,}')

    # ----------------------------------------------------#
    #           训练过程
    # ----------------------------------------------------#
    writer = SummaryWriter(log_dir)
    print('Tensorboard 构建完成，进入路径：' + log_dir)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # 导入测试图像
    test_image = next(iter(train_loader)).to(args.device)
    print('测试数据载入完成...')

    # 训练设置
    num_epochs = args.num_epochs
    
    # 损失函数
    mse_loss = torch.nn.MSELoss().to(device)
    ssim_loss = msssim
    
    # 多尺度梯度损失函数
    if args.use_multiscale_gradient:
        from utils.util_loss import MultiScaleGradientLoss
        # 根据梯度方向参数调整alpha和beta
        gradient_alpha = args.gradient_alpha
        gradient_beta = args.gradient_beta
        if args.gradient_direction == 'single':
            # 单方向梯度损失：只使用水平梯度
            gradient_alpha = 1.0
            gradient_beta = 0.0
            print(f'[梯度方向] 单方向梯度损失 (alpha={gradient_alpha}, beta={gradient_beta})')
        else:
            # 双向梯度损失：使用默认的alpha和beta
            print(f'[梯度方向] 双向梯度损失 (alpha={gradient_alpha}, beta={gradient_beta})')
        
        gradient_loss = MultiScaleGradientLoss(
            scales=args.gradient_scales,
            alpha=gradient_alpha,
            beta=gradient_beta,
            size_average=True
        ).to(device)
        print(f'[启用] 多尺度梯度损失函数已启用 (scales={args.gradient_scales}, weight={args.gradient_weight})')
    else:
        gradient_loss = None
        print('✗ 多尺度梯度损失函数已禁用')
    
    # 优化版学习率策略：预热 + 余弦退火
    optimizer = torch.optim.AdamW(model_train.parameters(), args.lr, weight_decay=1e-4)
    
    # 学习率预热
    warmup_scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.lr)
    
    # 余弦退火调度器（在预热后使用）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=40,  # 周期长度
        T_mult=1,  # 周期倍增
        eta_min=1e-6  # 最小学习率
    )

    # 是否预训练（继续训练功能）
    if args.resume_path:
        print('='*60)
        print('【继续训练模式】')
        print('='*60)
        
        # 检查文件是否存在
        if not os.path.exists(args.resume_path):
            print(f'❌ 错误：模型权重文件不存在: {args.resume_path}')
            print('请检查文件路径是否正确！')
            exit(1)
        
        try:
            print(f'正在加载预训练模型: {args.resume_path}')
            checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)
            
            # 检查checkpoint文件格式是否正确
            required_keys = ['epoch', 'model', 'encoder_state_dict', 'decoder_state_dict', 'optimizer', 'lr', 'best_loss']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                print(f'❌ 错误：checkpoint文件缺少必要的键: {missing_keys}')
                print('checkpoint文件格式不正确，请使用正确的训练权重文件！')
                exit(1)
            
            # 加载模型权重
            print('正在加载模型权重...')
            model_train.load_state_dict(checkpoint['model'])
            
            # 加载优化器状态（保留训练状态）
            print('正在恢复优化器状态...')
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 恢复学习率调度器状态
            print('正在恢复学习率调度器状态...')
            lr_scheduler.load_state_dict(checkpoint['lr'])
            
            # 恢复训练进度
            init_epoch = checkpoint['epoch'] + 1  # 从下一个epoch继续
            best_loss = checkpoint['best_loss']
            
            print(f'[成功] 模型权重加载成功')
            print(f'[成功] 优化器状态恢复成功')
            print(f'[成功] 学习率调度器状态恢复成功')
            print(f'[继续] 将从epoch {init_epoch} 继续训练（上次训练到epoch {checkpoint["epoch"]}）')
            print(f'[上次] 上次最佳loss: {best_loss:.6f}')
            
        except Exception as e:
            print(f'❌ 错误：加载模型权重文件时发生异常')
            print(f'异常信息: {str(e)}')
            print('请检查checkpoint文件是否损坏或格式是否正确！')
            exit(1)
    else:
        print('='*60)
        print('【从头开始训练】')
        print('='*60)
        weights_init(model_train)
        init_epoch = 0
        best_loss = float('inf')
    
    print('='*60)
    print('网络模型及优化器构建完成...')
    print('='*60)
    
    # 训练开始前清理旧checkpoint文件
    print('='*60)
    print('【Checkpoint管理】')
    print('='*60)
    print(f'Checkpoint目录: {checkpoint_dir}')
    print('[策略] 保存策略：每个epoch保存last.pth，loss改进时保存best.pth')
    print('[清理] 自动清理：每个epoch清理旧epoch*.pth文件')
    cleanup_old_checkpoints(checkpoint_dir)
    print('='*60)
    
    start_time = time.time()
    
    # 训练循环
    print('='*60)
    if init_epoch > 0:
        print(f'【继续训练】将从epoch {init_epoch} 训练到epoch {num_epochs}')
        print(f'剩余训练epoch数: {num_epochs - init_epoch}')
    else:
        print(f'【从头训练】将从epoch 1 训练到epoch {num_epochs}')
    
    # 混合精度训练初始化
    if args.use_mixed_precision and torch.cuda.is_available():
        scaler = GradScaler()
        print(f'[启用] 混合精度训练已启用 (GradScaler)')
    else:
        scaler = None
        if args.use_mixed_precision and not torch.cuda.is_available():
            print('[警告] 混合精度训练需要CUDA设备，当前设备不支持，已禁用')
        else:
            print('✗ 混合精度训练已禁用')
    
    print('='*60)
    
    for epoch in range(init_epoch, num_epochs):
        # 学习率预热
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = warmup_scheduler.get_lr()
        else:
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
        
        # 获取自适应损失权重
        mse_weight, ssim_weight, gradient_weight = get_adaptive_loss_weights(
            epoch, num_epochs, args.use_multiscale_gradient
        )
        
        # 更新损失函数字典
        criterion = {
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss,
            "lambda": ssim_weight,
        }
        
        # 如果启用了多尺度梯度损失，添加到损失函数字典
        if args.use_multiscale_gradient and gradient_loss is not None:
            criterion["gradient_loss"] = gradient_loss
            criterion["gradient_weight"] = gradient_weight * args.gradient_weight
        
        # =====================train============================
        model_train.train()
        train_epoch_loss = {"mse_loss": [], "ssim_loss": [], "total_loss": []}
        
        # 如果启用了多尺度梯度损失，初始化梯度损失记录
        if args.use_multiscale_gradient:
            train_epoch_loss["gradient_loss"] = []
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch_idx, image_batch in enumerate(pbar, start=1):
            optimizer.zero_grad()
            inputs = image_batch.to(device)
            labels = image_batch.data.clone().to(device)
            
            # 混合精度训练前向传播
            if scaler is not None:
                with autocast():
                    # 前向传播
                    outputs = model_train(inputs)
                    
                    # 计算损失（使用自适应权重）
                    pixel_loss_value = criterion["mse_loss"](outputs, labels)
                    ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
                    
                    # 基础损失
                    loss = mse_weight * pixel_loss_value + ssim_weight * ssim_loss_value
                    
                    # 如果启用了多尺度梯度损失，添加梯度损失
                    if args.use_multiscale_gradient and "gradient_loss" in criterion:
                        gradient_loss_value = criterion["gradient_loss"](outputs, labels)
                        loss += criterion["gradient_weight"] * gradient_loss_value
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通精度训练
                # 前向传播
                outputs = model_train(inputs)
                
                # 计算损失（使用自适应权重）
                pixel_loss_value = criterion["mse_loss"](outputs, labels)
                ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
                
                # 基础损失
                loss = mse_weight * pixel_loss_value + ssim_weight * ssim_loss_value
                
                # 如果启用了多尺度梯度损失，添加梯度损失
                if args.use_multiscale_gradient and "gradient_loss" in criterion:
                    gradient_loss_value = criterion["gradient_loss"](outputs, labels)
                    loss += criterion["gradient_weight"] * gradient_loss_value
                
                # 反向传播
                loss.backward()
                optimizer.step()
            
            # 记录损失值
            train_epoch_loss["mse_loss"].append(pixel_loss_value.item())
            train_epoch_loss["ssim_loss"].append(ssim_loss_value.item())
            train_epoch_loss["total_loss"].append(loss.item())
            
            # 如果启用了多尺度梯度损失，记录梯度损失值
            if args.use_multiscale_gradient and "gradient_loss" in criterion:
                train_epoch_loss["gradient_loss"].append(gradient_loss_value.item())
            
            # 显示训练进度（包含继续训练信息）
            mode_str = "继续训练" if init_epoch > 0 else "从头训练"
            pbar.set_description(f'Epoch [{epoch + 1}/{num_epochs}] ({mode_str})')
            
            # 构建进度显示信息
            postfix_info = {
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}',
                'mse_w': f'{mse_weight:.1f}',
                'ssim_w': f'{ssim_weight:.0f}'
            }
            
            # 如果启用了多尺度梯度损失，添加梯度损失信息
            if args.use_multiscale_gradient and "gradient_loss" in criterion:
                postfix_info['grad_w'] = f'{criterion["gradient_weight"]:.0f}'
                postfix_info['grad_loss'] = f'{gradient_loss_value.item():.4f}'
            
            pbar.set_postfix(postfix_info)
        
        # 计算平均损失
        train_loss = {
            "mse_loss": sum(train_epoch_loss["mse_loss"]) / len(train_epoch_loss["mse_loss"]),
            "ssim_loss": sum(train_epoch_loss["ssim_loss"]) / len(train_epoch_loss["ssim_loss"]),
            "total_loss": sum(train_epoch_loss["total_loss"]) / len(train_epoch_loss["total_loss"]),
        }
        
        # 如果启用了多尺度梯度损失，记录梯度损失
        if args.use_multiscale_gradient and "gradient_loss" in train_epoch_loss:
            train_loss["gradient_loss"] = sum(train_epoch_loss["gradient_loss"]) / len(train_epoch_loss["gradient_loss"])
        
        # =====================valid============================
        # 无验证集，替换成在tensorboard中测试
        with torch.no_grad():
            rebuild_img = model_train(test_image)
            import torchvision
            img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
            img_grid_rebuild = torchvision.utils.make_grid(rebuild_img, normalize=True, nrow=4)
            writer.add_image('Real image', img_grid_real, global_step=1)
            writer.add_image('Rebuild image', img_grid_rebuild, global_step=epoch)
        
        # 记录损失值
        for loss_name, loss_value in train_loss.items():
            writer.add_scalar(loss_name, loss_value, global_step=epoch)
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
        writer.add_scalar('mse_weight', mse_weight, global_step=epoch)
        writer.add_scalar('ssim_weight', ssim_weight, global_step=epoch)
        
        # 如果启用了多尺度梯度损失，记录梯度权重
        if args.use_multiscale_gradient:
            writer.add_scalar('gradient_weight', gradient_weight * args.gradient_weight, global_step=epoch)
        
        # =====================checkpoint=======================
        # 确保checkpoint目录存在
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # 准备checkpoint数据
        checkpoint = {
            'epoch': epoch,
            'model': model_train.state_dict(),
            'encoder_state_dict': model_train.encoder.state_dict(),
            'decoder_state_dict': model_train.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'cbam_scheme': args.cbam_scheme,  # 保存CBAM方案信息
            'reduction_ratio': args.reduction_ratio,  # 保存reduction_ratio参数
            'use_color_aware': args.use_color_aware,  # 保存颜色感知CBAM信息
            'color_preservation_weight': args.color_preservation_weight,  # 保存颜色保护权重
        }
        
        # 保存last.pth（每个epoch都保存）
        last_save_path = os.path.join(checkpoint_dir, 'last.pth')
        torch.save(checkpoint, last_save_path)
        
        # 如果loss改进，保存best.pth
        if train_loss["total_loss"] < best_loss:
            best_loss = train_loss["total_loss"]
            checkpoint['best_loss'] = best_loss
            best_save_path = os.path.join(checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_save_path)
            print(f'[保存] 保存最佳模型: best.pth (loss: {best_loss:.6f})')
        
        # 清理旧的epoch*.pth文件（只保留best.pth和last.pth）
        cleanup_old_checkpoints(checkpoint_dir)

    writer.close()
    end_time = time.time()
    
    # 训练结束前进行最终清理
    cleanup_old_checkpoints(checkpoint_dir)
    
    print('='*60)
    print('训练完成！')
    print('='*60)
    print(f'训练耗时：{end_time - start_time:.2f}秒')
    print(f'训练epoch数：{num_epochs - init_epoch}')
    print(f'Best loss: {best_loss:.6f}')
    print()
    print('【最终Checkpoint文件】')
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print(f'  [最佳] best.pth - 最佳模型 (loss: {best_loss:.6f})')
    else:
        print(f'  [未生成] best.pth - 未生成')
    if os.path.exists(os.path.join(checkpoint_dir, 'last.pth')):
        last_info = torch.load(os.path.join(checkpoint_dir, 'last.pth'), map_location='cpu')
        print(f'  [最新] last.pth - 最新模型 (epoch: {last_info["epoch"]}, loss: {last_info["best_loss"]:.6f})')
    else:
        print(f'  [未生成] last.pth - 未生成')
    print()
    
    if init_epoch > 0:
        print(f'继续训练模式：从epoch {init_epoch} 训练到epoch {num_epochs}')
    else:
        print(f'从头训练模式：从epoch 1 训练到epoch {num_epochs}')
    
    print(f'预期EN提升：20-30%（相比基线模型）')
    print('='*60)