# 三阶段训练详细文档

## 📋 概述

本文档详细说明 `run_train.py` 实现的三阶段训练流程，这是红外可见光图像融合模型的完整训练方案。

## 🎯 三阶段训练流程

### 阶段一：自编码器预训练（AutoEncoder Pretraining）

**目标**: 训练编码器和解码器的基础重建能力

**特点**:
- 输入：单张可见光图像
- 模型：编码器（不含CBAM）+ 解码器
- 优化目标：重建损失（L1 + SSIM + Gradient + TV）
- 可训练参数：全部参数

**关键配置**:
```bash
--stage1_epochs: 80        # 阶段一训练轮数
--stage1_lr: 2e-4          # 阶段一学习率
--use_attention: False      # 阶段一不使用注意力机制
```

### 阶段二：CBAM微调（CBAM Fine-tuning）

**目标**: 在预训练的基础上引入CBAM注意力机制

**特点**:
- 输入：单张可见光图像
- 模型：编码器（含CBAM）+ 解码器
- 优化目标：重建损失
- 可训练参数：仅CBAM模块（主干权重冻结）

**关键配置**:
```bash
--stage2_epochs: 40        # 阶段二训练轮数
--stage2_lr: 1e-4          # 阶段二学习率
--use_attention: True      # 阶段二启用注意力机制
--resume_stage1: ./checkpoints/stage1/best.pth  # 加载阶段一权重
```

**训练策略**:
```python
model.freeze_backbone()     # 冻结主干权重
model.unfreeze_cbam()       # 解冻CBAM模块
optimizer = AdamW(cbam_params, lr=1e-4)  # 仅优化CBAM参数
```

### 阶段三：端到端融合（End-to-End Fusion）

**目标**: 完成红外-可见光图像融合训练

**特点**:
- 输入：红外+可见光图像对
- 模型：编码器（含CBAM）+ 融合层 + 解码器
- 优化目标：融合损失
- 可训练参数：全部参数或仅融合层

**关键配置**:
```bash
--stage3_epochs: 60         # 阶段三训练轮数
--stage3_lr: 1e-4          # 阶段三学习率
--fusion_strategy: l1_norm  # 融合策略（addition/l1_norm/weighted_average）
--end_to_end_finetune: False # 是否端到端微调
--resume_stage2: ./checkpoints/stage2/best.pth  # 加载阶段二权重
```

## 🔧 完整参数说明

### 训练阶段选择参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_all_stages` | bool | False | 执行完整的三阶段训练 |
| `--stage` | int | 1 | 选择训练阶段 (1/2/3) |

### 数据集参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ir_path` | str | M3FD路径 | 红外图像路径 |
| `--vi_path` | str | M3FD路径 | 可见光图像路径 |
| `--gray` | bool | False | 是否使用灰度模式 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | auto | 训练设备（cpu/cuda） |
| `--batch_size` | int | 16 | 批量大小 |
| `--num_workers` | int | 0 | 数据加载线程数 |

### 各阶段参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stage1_epochs` | int | 80 | 阶段一训练轮数 |
| `--stage1_lr` | float | 2e-4 | 阶段一学习率 |
| `--stage2_epochs` | int | 40 | 阶段二训练轮数 |
| `--stage2_lr` | float | 1e-4 | 阶段二学习率 |
| `--stage3_epochs` | int | 60 | 阶段三训练轮数 |
| `--stage3_lr` | float | 1e-4 | 阶段三学习率 |
| `--fusion_strategy` | str | l1_norm | 融合策略 |
| `--end_to_end_finetune` | bool | False | 阶段三是否端到端微调 |

### 恢复训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resume_stage1` | str | ./checkpoints/stage1/best.pth | 阶段一模型路径 |
| `--resume_stage2` | str | ./checkpoints/stage2/best.pth | 阶段二模型路径 |
| `--resume_path` | str | None | 当前阶段恢复路径 |

### 优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warmup_epochs` | int | 5 | 学习率预热轮数 |
| `--use_gradient_clipping` | bool | True | 梯度裁剪 |
| `--use_adaptive_weights` | bool | True | 自适应权重 |
| `--optimize_en_ag` | bool | True | EN/AG优化模式 |
| `--use_balanced_loss` | bool | True | 平衡损失模式 |
| `--use_lr_decay` | bool | True | 学习率衰减 |

### 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | bool | True | 显示输出 |
| `--base_dir` | str | ./runs | 基础输出目录 |

## 📖 使用方法

### 1. 完整三阶段训练

```bash
python run_train.py --train_all_stages
```

**执行流程**:
1. 自动执行阶段一训练（80 epochs）
2. 自动执行阶段二训练（40 epochs）
3. 自动执行阶段三训练（60 epochs）
4. 保存所有阶段的模型checkpoint

### 2. 单阶段训练

```bash
# 训练阶段一
python run_train.py --stage 1

# 训练阶段二（需要阶段一模型）
python run_train.py --stage 2 --resume_stage1 ./checkpoints/stage1/best.pth

# 训练阶段三（需要阶段二模型）
python run_train.py --stage 3 --resume_stage2 ./checkpoints/stage2/best.pth
```

### 3. 自定义参数训练

```bash
python run_train.py --train_all_stages \
    --stage1_epochs 100 \
    --stage2_epochs 50 \
    --stage3_epochs 80 \
    --batch_size 32 \
    --fusion_strategy weighted_average \
    --end_to_end_finetune
```

## 🏗️ 模块结构

### train/ 模块

```
train/
├── __init__.py          # 包初始化
├── lr_scheduler.py      # 学习率调度器
├── loss_weights.py      # 损失权重管理器
├── trainer.py          # 训练器
└── callbacks.py        # 回调函数
```

### 各模块功能

#### trainer.py - 训练器

**核心功能**:
- 完整的训练循环管理
- 多损失函数组合（L1 + SSIM + Gradient + TV）
- 学习率调度（预热 + 余弦退火 + 衰减）
- Checkpoint自动保存
- Tensorboard可视化
- 训练停滞检测和自动恢复

**Trainer 类**:
```python
class Trainer:
    def __init__(self, model, optimizer, train_loader, device, loss_fn,
                 val_loader=None, checkpoint_dir='./checkpoints', log_dir='./logs',
                 ssim_loss_fn=None, grad_loss_fn=None, tv_loss_fn=None,
                 warmup_epochs=5, initial_lr=2e-4, use_gradient_clipping=True)
    
    def train(self, num_epochs, init_epoch=0, use_adaptive_weights=True, 
              optimize_en_ag=False, use_balanced_loss=True, use_lr_decay=True)
```

#### lr_scheduler.py - 学习率调度

**核心功能**:
- **WarmupScheduler**: 学习率预热（0 → target_lr）
- **LearningRateOptimizer**: 学习率优化（重新升温 + 自适应调整）

**使用示例**:
```python
# 预热调度器
warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, target_lr=2e-4)

# 学习率优化器
optimizer = LearningRateOptimizer.re_warmup_and_optimize(
    optimizer, current_lr=1e-5, target_lr=1e-4, epochs_to_recover=10
)
```

#### loss_weights.py - 损失权重管理

**核心功能**:
- 自适应损失权重计算
- EN/AG优化模式
- 平衡损失模式

**损失函数**:
```
Total_Loss = λ1 × L1_Loss + λ2 × (1 - SSIM) + λ3 × Gradient_Loss + λ4 × TV_Loss
```

**自适应权重策略**:

| 训练阶段 | L1权重 | SSIM权重 | 梯度权重 | TV权重 |
|---------|--------|----------|---------|--------|
| 前期 (0-30%) | 1.0 | 1.0 | 1.5 | 1.0 |
| 中期 (30-70%) | 0.7 | 1.2 | 2.0 | 1.0 |
| 后期 (70-100%) | 0.4 | 2.0 | 2.5 | 1.0 |

#### callbacks.py - 回调函数

**核心功能**:
- **CheckpointCallback**: 自动保存模型checkpoint
- **EarlyStoppingCallback**: 早停机制
- **MetricsLoggerCallback**: 指标记录

## 🔍 训练监控

### Tensorboard可视化

训练过程自动记录以下指标：
- 各损失项（l1_loss, ssim_loss, grad_loss, tv_loss, total_loss）
- 学习率
- 损失权重
- 训练图像对比

**启动Tensorboard**:
```bash
tensorboard --logdir=./runs/stage1/logs
```

### 训练停滞检测

当检测到loss停滞（5个epoch内变化<1%）时：
1. 记录停滞次数
2. 停滞达到10次且距离上次恢复>20个epoch时触发重新升温
3. 学习率恢复至初始值的60%
4. 重新进行预热

## 📊 输出结构

```
runs/
├── stage1_autoencoder/
│   ├── checkpoints/
│   │   ├── epoch000-loss0.xxx.pth    # 训练过程checkpoint
│   │   ├── best.pth                   # 最佳模型
│   │   └── stage1_final.pth          # 最终模型
│   └── logs/
│       └── events.*                  # Tensorboard日志
├── stage2_cbam/
│   └── ...
└── stage3_fusion/
    └── ...
```

## ⚙️ 高级配置

### 1. 自定义融合策略

```bash
# 使用加权平均融合
python run_train.py --stage 3 --fusion_strategy weighted_average

# 使用加法融合
python run_train.py --stage 3 --fusion_strategy addition

# 使用L1范数融合（默认）
python run_train.py --stage 3 --fusion_strategy l1_norm
```

### 2. 端到端微调

```bash
# 启用端到端微调（解冻所有参数）
python run_train.py --stage 3 --end_to_end_finetune
```

### 3. 自适应权重控制

```bash
# 禁用自适应权重
python run_train.py --stage 1 --use_adaptive_weights False

# 启用EN/AG优化
python run_train.py --stage 1 --optimize_en_ag True --use_balanced_loss True
```

## 🐛 故障排除

### 1. 恢复训练失败

**问题**: 模型文件不存在或格式错误

**解决方案**:
```bash
# 检查checkpoint是否存在
ls ./checkpoints/stage1/

# 使用正确的恢复路径
python run_train.py --stage 2 --resume_stage1 ./checkpoints/stage1/best.pth
```

### 2. GPU内存不足

**问题**: 批次大小过大导致OOM

**解决方案**:
```bash
# 减小批次大小
python run_train.py --train_all_stages --batch_size 8

# 使用CPU训练
python run_train.py --train_all_stages --device cpu
```

### 3. 学习率设置不当

**问题**: loss不下降或震荡

**解决方案**:
```bash
# 降低学习率
python run_train.py --stage 1 --stage1_lr 1e-4

# 增加预热轮数
python run_train.py --stage 1 --warmup_epochs 10
```

## 📝 示例脚本

### 完整训练脚本

```bash
#!/bin/bash

# 完整三阶段训练
python run_train.py --train_all_stages \
    --ir_path ./data/test/ir \
    --vi_path ./data/test/vi \
    --batch_size 16 \
    --stage1_epochs 80 --stage1_lr 2e-4 \
    --stage2_epochs 40 --stage2_lr 1e-4 \
    --stage3_epochs 60 --stage3_lr 1e-4 \
    --fusion_strategy l1_norm \
    --warmup_epochs 5 \
    --use_adaptive_weights True \
    --optimize_en_ag True \
    --use_balanced_loss True \
    --use_gradient_clipping True
```

### 仅训练融合层

```bash
# 阶段三，仅训练融合层
python run_train.py --stage 3 \
    --resume_stage2 ./checkpoints/stage2/best.pth \
    --end_to_end_finetune False \
    --stage3_epochs 60
```

### 端到端微调

```bash
# 阶段三，端到端微调所有参数
python run_train.py --stage 3 \
    --resume_stage2 ./checkpoints/stage2/best.pth \
    --end_to_end_finetune True \
    --stage3_epochs 30 \
    --stage3_lr 5e-5  # 使用较小的学习率
```

## 📚 参考文献

- DenseFuse: A Fusion Approach to Infrared and Visible Images (IEEE TIP 2019)
- CBAM: Convolutional Block Attention Module (ECCV 2018)
- Warmup: Accelerating Training Process (2018)

## 🔗 相关文档

- [训练流程图](../docs/TRAINING_FLOWCHART.md)
- [三阶段训练总结](../docs/THREE_STAGE_TRAINING_SUMMARY.md)
- [性能优化指南](../docs/PERFORMANCE_OPTIMIZATION.md)
