# -*- coding: utf-8 -*-
"""
@file name:train/callbacks.py
@desc: 训练回调函数模块
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块提供训练过程中的回调函数，用于扩展训练行为：
- Checkpoint保存
- 早停（Early Stopping）
- 学习率调整
- 自定义监控

使用示例：
---------
    from train.callbacks import CheckpointCallback, EarlyStoppingCallback
    
    callbacks = [
        CheckpointCallback(save_dir='./checkpoints'),
        EarlyStoppingCallback(patience=10),
    ]
    
    for epoch in range(num_epochs):
        # 训练代码...
        for callback in callbacks:
            callback.on_epoch_end(epoch, metrics={'loss': loss_value})
"""

import os
import torch


class CheckpointCallback:
    """
    Checkpoint保存回调
    
    功能：
    -----
    在每个epoch结束时保存模型checkpoint
    
    使用示例：
    ---------
    ```python
    callback = CheckpointCallback(
        save_dir='./checkpoints',
        save_best_only=True,
        monitor='loss',
        mode='min'
    )
    
    for epoch in range(num_epochs):
        # 训练...
        callback.on_epoch_end(epoch, model, optimizer, loss=0.5)
    ```
    """
    
    def __init__(self, save_dir, save_best_only=True, monitor='loss', mode='min'):
        """
        初始化Checkpoint回调
        
        Args:
            save_dir (str): checkpoint保存目录
            save_best_only (bool): 是否只保存最佳模型
            monitor (str): 监控指标名
            mode (str): 'min'或'max'
        """
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        os.makedirs(save_dir, exist_ok=True)
        
        if mode == 'min':
            self.best_value = float('inf')
            self.compare_fn = lambda new, best: new < best
        else:
            self.best_value = float('-inf')
            self.compare_fn = lambda new, best: new > best
    
    def on_epoch_end(self, epoch, model, optimizer, **metrics):
        """
        Epoch结束时的回调
        
        Args:
            epoch (int): 当前epoch
            model (torch.nn.Module): 模型
            optimizer (torch.optim.Optimizer): 优化器
            **metrics: 其他指标
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            return
        
        # 检查是否应该保存
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.compare_fn(current_value, self.best_value):
            should_save = True
            self.best_value = current_value
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_value': self.best_value,
                **metrics
            }
            
            filename = f'checkpoint_epoch{epoch:03d}.pth'
            filepath = os.path.join(self.save_dir, filename)
            torch.save(checkpoint, filepath)
            
            print(f"Checkpoint已保存: {filepath}")


class EarlyStoppingCallback:
    """
    早停回调
    
    功能：
    -----
    当验证指标不再改善时，停止训练以避免过拟合
    
    使用示例：
    ---------
    ```python
    callback = EarlyStoppingCallback(
        patience=10,
        min_delta=0.001,
        monitor='val_loss'
    )
    
    for epoch in range(num_epochs):
        # 训练...
        val_loss = evaluate(model, val_loader)
        
        should_stop = callback.on_epoch_end(epoch, val_loss=val_loss)
        if should_stop:
            print("早停触发，停止训练")
            break
    ```
    """
    
    def __init__(self, patience=10, min_delta=0.001, monitor='loss'):
        """
        初始化早停回调
        
        Args:
            patience (int): 容忍epoch数
            min_delta (float): 最小改善量
            monitor (str): 监控指标名
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def on_epoch_end(self, epoch, **metrics):
        """
        Epoch结束时的回调
        
        Args:
            epoch (int): 当前epoch
            **metrics: 指标字典
        
        Returns:
            bool: 是否应该停止训练
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            return False
        
        if self.best_value is None:
            self.best_value = current_value
        elif current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"早停触发：连续{self.patience}个epoch无改善")
        
        return self.should_stop
    
    def reset(self):
        """
        重置早停状态
        """
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class MetricsLoggerCallback:
    """
    指标记录回调
    
    功能：
    -----
    记录训练过程中的各种指标到文件
    
    使用示例：
    ---------
    ```python
    logger = MetricsLoggerCallback(log_file='training_metrics.csv')
    
    for epoch in range(num_epochs):
        # 训练...
        metrics = {'loss': 0.5, 'accuracy': 0.95}
        logger.on_epoch_end(epoch, **metrics)
    ```
    """
    
    def __init__(self, log_file='training_metrics.csv'):
        """
        初始化指标记录器
        
        Args:
            log_file (str): 日志文件路径
        """
        self.log_file = log_file
        self.initialized = False
    
    def on_epoch_end(self, epoch, **metrics):
        """
        Epoch结束时的回调
        
        Args:
            epoch (int): 当前epoch
            **metrics: 指标字典
        """
        import csv
        
        row = {'epoch': epoch, **metrics}
        
        # 写入CSV
        mode = 'a' if self.initialized else 'w'
        
        with open(self.log_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            if not self.initialized:
                writer.writeheader()
                self.initialized = True
            
            writer.writerow(row)
