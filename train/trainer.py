# -*- coding: utf-8 -*-
"""
@file name:train/trainer.py
@desc: 训练器模块 - 包含完整的训练逻辑和回调函数
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块提供完整的模型训练功能，包括：
- 训练循环管理
- 损失计算和记录
- 学习率调度
- 模型保存和恢复
- 训练监控和回调

主要组件：
---------
1. Trainer: 主训练器类
2. 训练循环：epoch循环和batch循环
3. 验证和可视化
4. Checkpoint保存

使用示例：
---------
    from train import Trainer
    from train.loss_weights import get_adaptive_loss_weights
    from train.lr_scheduler import WarmupScheduler, LearningRateOptimizer
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        loss_fn=l1_loss_fn
    )
    
    trainer.train(num_epochs=120, init_epoch=0)
"""

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision


class Trainer:
    """
    模型训练器
    
    功能：
    -----
    提供完整的训练流程管理，包括：
    - 训练循环（epoch + batch）
    - 损失计算和记录
    - 学习率调度（预热 + 余弦退火）
    - 模型checkpoint保存
    - Tensorboard可视化
    - 训练监控（loss停滞检测）
    
    使用方法：
    ---------
    ```python
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        loss_fn=l1_loss,
        val_loader=val_loader,  # 可选
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    )
    
    # 开始训练
    trainer.train(num_epochs=120, init_epoch=0)
    ```
    
    Attributes:
    -----------
    model : torch.nn.Module
        待训练的模型
    optimizer : torch.optim.Optimizer
        优化器
    train_loader : DataLoader
        训练数据加载器
    device : torch.device
        训练设备
    loss_fn : callable
        损失函数
    checkpoint_dir : str
        checkpoint保存路径
    log_dir : str
        Tensorboard日志路径
    writer : SummaryWriter
        Tensorboard writer
    """
    
    def __init__(self, model, optimizer, train_loader, device, loss_fn, 
                 val_loader=None, checkpoint_dir='./checkpoints', log_dir='./logs',
                 ssim_loss_fn=None, grad_loss_fn=None, tv_loss_fn=None,
                 warmup_epochs=5, initial_lr=2e-4, use_gradient_clipping=True):
        """
        初始化训练器
        
        Args:
            model (torch.nn.Module): 待训练的模型
            optimizer (torch.optim.Optimizer): 优化器
            train_loader (DataLoader): 训练数据加载器
            device (torch.device): 训练设备
            loss_fn (callable): 主损失函数
            val_loader (DataLoader, optional): 验证数据加载器
            checkpoint_dir (str): checkpoint保存路径
            log_dir (str): Tensorboard日志路径
            ssim_loss_fn (callable, optional): SSIM损失函数
            grad_loss_fn (callable, optional): 梯度损失函数
            tv_loss_fn (callable, optional): TV损失函数
            warmup_epochs (int): 学习率预热epoch数
            initial_lr (float): 初始学习率
            use_gradient_clipping (bool): 是否使用梯度裁剪
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.loss_fn = loss_fn
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.use_gradient_clipping = use_gradient_clipping
        
        # 损失函数
        self.ssim_loss_fn = ssim_loss_fn
        self.grad_loss_fn = grad_loss_fn
        self.tv_loss_fn = tv_loss_fn
        
        # 学习率
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # 训练监控
        self.loss_history = []
        self.stagnation_counter = 0
        self.last_lr_recovery_epoch = 0
    
    def train(self, num_epochs, init_epoch=0, use_adaptive_weights=True, optimize_en_ag=False,
              use_balanced_loss=True, use_lr_decay=True, use_mixed_precision=False):
        """
        执行训练流程
        
        Args:
            num_epochs (int): 总训练epoch数
            init_epoch (int): 初始epoch（用于继续训练）
            use_adaptive_weights (bool): 是否使用自适应权重
            optimize_en_ag (bool): 是否启用EN/AG优化模式
            use_balanced_loss (bool): 是否使用平衡损失
            use_lr_decay (bool): 是否使用学习率衰减
            use_mixed_precision (bool): 是否使用混合精度训练
        
        Returns:
            dict: 训练结果统计
        """
        from train.lr_scheduler import WarmupScheduler, LearningRateOptimizer
        from train.loss_weights import get_adaptive_loss_weights
        
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        # 初始化调度器
        warmup_scheduler = WarmupScheduler(
            self.optimizer, 
            self.warmup_epochs, 
            self.initial_lr
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 恢复调度器状态（如果是继续训练）
        if init_epoch > 0:
            # 重新初始化warmup
            warmup_scheduler = WarmupScheduler(
                self.optimizer,
                self.warmup_epochs,
                self.initial_lr
            )
        
        start_time = time.time()
        best_loss = float('inf')
        
        # 获取测试图像
        test_image = next(iter(self.train_loader)).to(self.device)
        
        for epoch in range(init_epoch, num_epochs):
            # ==================== 学习率调度 ====================
            if epoch < self.warmup_epochs:
                warmup_scheduler.step()
                current_lr = warmup_scheduler.get_lr()
            else:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]
            
            # 学习率衰减
            if use_lr_decay and epoch > self.warmup_epochs and epoch > 0 and epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.8
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n学习率衰减至: {current_lr:.6f} (Epoch {epoch + 1})")
            
            # 获取损失权重
            l1_weight, ssim_weight, grad_weight, tv_weight = get_adaptive_loss_weights(
                epoch=epoch,
                total_epochs=num_epochs,
                use_balanced_loss=use_balanced_loss,
                use_adaptive_weights=use_adaptive_weights,
                optimize_en_ag=optimize_en_ag
            )
            
            # ==================== 训练循环 ====================
            self.model.train()
            train_epoch_loss = {
                'l1_loss': [],
                'ssim_loss': [],
                'grad_loss': [],
                'tv_loss': [],
                'total_loss': []
            }
            
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            for batch_idx, image_batch in enumerate(pbar, start=1):
                self.optimizer.zero_grad()
                inputs = image_batch.to(self.device)
                labels = image_batch.data.clone().to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                pixel_loss = self.loss_fn(outputs, labels)
                
                # SSIM损失
                if self.ssim_loss_fn is not None:
                    ssim_loss = 1 - self.ssim_loss_fn(outputs, labels, normalize=True)
                else:
                    ssim_loss = torch.tensor(0.0)
                
                # 梯度损失
                if self.grad_loss_fn is not None:
                    grad_loss = self.grad_loss_fn(outputs, labels)
                else:
                    grad_loss = torch.tensor(0.0)
                
                # TV损失
                if self.tv_loss_fn is not None:
                    tv_loss = self.tv_loss_fn(outputs)
                else:
                    tv_loss = torch.tensor(0.0)
                
                # 总损失
                total_loss = (l1_weight * pixel_loss + 
                            ssim_weight * ssim_loss + 
                            grad_weight * grad_loss + 
                            tv_weight * tv_loss)
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                train_epoch_loss['l1_loss'].append(pixel_loss.item())
                train_epoch_loss['ssim_loss'].append(ssim_loss.item())
                train_epoch_loss['grad_loss'].append(grad_loss.item())
                train_epoch_loss['tv_loss'].append(tv_loss.item())
                train_epoch_loss['total_loss'].append(total_loss.item())
                
                # 更新进度条
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            # 计算平均损失
            avg_losses = {
                key: sum(values) / len(values) 
                for key, values in train_epoch_loss.items()
            }
            
            # ==================== 验证和可视化 ====================
            with torch.no_grad():
                rebuild_img = self.model(test_image)
                img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
                img_grid_rebuild = torchvision.utils.make_grid(rebuild_img, normalize=True, nrow=4)
                self.writer.add_image('Real image', img_grid_real, global_step=epoch)
                self.writer.add_image('Rebuild image', img_grid_rebuild, global_step=epoch)
            
            # 记录到Tensorboard
            for loss_name, loss_value in avg_losses.items():
                self.writer.add_scalar(loss_name, loss_value, global_step=epoch)
            self.writer.add_scalar('learning_rate', current_lr, global_step=epoch)
            self.writer.add_scalar('l1_weight', l1_weight, global_step=epoch)
            self.writer.add_scalar('ssim_weight', ssim_weight, global_step=epoch)
            self.writer.add_scalar('grad_weight', grad_weight, global_step=epoch)
            self.writer.add_scalar('tv_weight', tv_weight, global_step=epoch)
            
            # ==================== 停滞检测和恢复 ====================
            self.loss_history.append(avg_losses['total_loss'])
            
            if len(self.loss_history) > 5:
                recent_change = (self.loss_history[-1] - self.loss_history[-5]) / self.loss_history[-5]
                
                if abs(recent_change) < 0.01:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                
                # 自动重新升温
                if (self.stagnation_counter >= 10 and 
                    (epoch - self.last_lr_recovery_epoch) > 20):
                    
                    print(f"\n⚠️ 检测到loss停滞，触发重新升温...")
                    
                    target_lr = self.initial_lr * 0.6
                    if current_lr < target_lr:
                        self.optimizer, recovery_epochs = LearningRateOptimizer.re_warmup_and_optimize(
                            self.optimizer,
                            current_lr,
                            target_lr,
                            epochs_to_recover=10
                        )
                        
                        warmup_scheduler = WarmupScheduler(
                            self.optimizer, 
                            recovery_epochs, 
                            target_lr
                        )
                        self.last_lr_recovery_epoch = epoch
                        self.stagnation_counter = 0
            
            # ==================== 保存Checkpoint ====================
            if avg_losses['total_loss'] < best_loss:
                best_loss = avg_losses['total_loss']
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=self.optimizer.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict(),
                    best_loss=best_loss
                )
        
        # 训练结束
        self.writer.close()
        end_time = time.time()
        
        print("=" * 60)
        print("训练完成！")
        print(f"训练耗时: {end_time - start_time:.2f}秒")
        print(f"最佳损失: {best_loss:.6f}")
        print("=" * 60)
        
        return {
            'best_loss': best_loss,
            'training_time': end_time - start_time,
            'num_epochs': num_epochs - init_epoch
        }
    
    def save_checkpoint(self, epoch, optimizer, lr_scheduler, best_loss):
        """
        保存模型checkpoint
        
        Args:
            epoch (int): 当前epoch
            optimizer (dict): 优化器状态
            lr_scheduler (dict): 学习率调度器状态
            best_loss (float): 最佳损失
        """
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer': optimizer,
            'lr': lr_scheduler,
            'best_loss': best_loss,
        }
        
        checkpoint_name = f'epoch{epoch:03d}-loss{best_loss:.3f}.pth'
        save_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, save_path)
        print(f"保存checkpoint: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载模型checkpoint
        
        Args:
            checkpoint_path (str): checkpoint文件路径
        
        Returns:
            dict: checkpoint内容
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        return checkpoint


def create_trainer(model, train_loader, device, args):
    """
    工厂函数：创建训练器实例
    
    Args:
        model (torch.nn.Module): 模型
        train_loader (DataLoader): 数据加载器
        device (torch.device): 设备
        args (Namespace): 配置参数
    
    Returns:
        Trainer: 训练器实例
    """
    from utils.util_loss import L1Loss as L1Loss
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    
    # 损失函数
    l1_loss_fn = torch.nn.L1Loss().to(device)
    
    # 获取损失函数
    from utils.util_loss import msssim, gradient_loss, tv_loss
    
    # 创建checkpoint目录
    run_dir, checkpoint_dir, log_dir = create_run_directory(args)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
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
    
    return trainer


def create_run_directory(args):
    """
    创建运行目录
    
    说明：
    -----
    本函数是对 utils/utils.py 中 create_run_directory 的包装，
    保持API一致性。
    
    Args:
        args (Namespace): 配置参数，应包含以下属性：
            - gray: 是否使用灰度模式
            - num_epochs: 训练轮数
            - base_dir: 可选，基础目录
    
    Returns:
        tuple: (run_dir, checkpoint_dir, log_dir)
    
    使用示例：
    ---------
    ```python
    from train.trainer import create_run_directory
    
    run_dir, checkpoint_dir, log_dir = create_run_directory(args)
    ```
    """
    # 使用 utils 中的实现
    from utils.utils import create_run_directory as utils_create_run_directory
    return utils_create_run_directory(args)
