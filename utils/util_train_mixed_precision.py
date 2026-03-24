# Author: wokaka209
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from .utils import get_lr


def train_epoch_mixed_precision(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches, scaler):
    """使用混合精度训练一个epoch
    Args:
        model: 待训练的模型
        device: 计算设备
        train_dataloader: 训练数据加载器
        criterion: 损失函数字典，包含'mse_loss'和'ssim_loss'
        optimizer: 优化器
        epoch: 当前epoch
        num_epochs: 总epoch数
        scaler: GradScaler实例用于混合精度训练
    Returns:
        包含平均损失值的字典
    """
    model.train()
    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }

    # 动态权重调整：前期注重像素重建，后期注重结构保持
    mse_weight = 1.0
    ssim_weight = 1000 * (1 + 0.5 * (epoch / num_Epoches))

    # 创建进度条
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_idx, image_batch in enumerate(pbar, start=1):
        # 清空梯度
        optimizer.zero_grad()
        
        # 载入批量图像
        inputs = image_batch.to(device)
        # 复制图像作为标签
        labels = image_batch.data.clone().to(device)
        
        # 使用自动混合精度
        with autocast():
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            pixel_loss_value = criterion["mse_loss"](outputs, labels)
            ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
            loss = mse_weight * pixel_loss_value + ssim_weight * ssim_loss_value
        
        # 反向传播（使用scaler进行缩放）
        scaler.scale(loss).backward()
        
        # 参数更新（使用scaler进行反缩放）
        scaler.step(optimizer)
        
        # 更新scaler
        scaler.update()
        
        # 记录损失值
        train_epoch_loss["mse_loss"].append(pixel_loss_value.item())
        train_epoch_loss["ssim_loss"].append(ssim_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        pbar.set_postfix({
            'pixel_loss': f'{pixel_loss_value.item():.4f}',
            'ssim_loss': f'{ssim_loss_value.item():.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })

    return {"mse_loss": np.mean(train_epoch_loss["mse_loss"]),
            "ssim_loss": np.mean(train_epoch_loss["ssim_loss"]),
            "total_loss": np.mean(train_epoch_loss["total_loss"]),
            }