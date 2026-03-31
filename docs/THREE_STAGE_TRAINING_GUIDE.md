# 三阶段训练使用指南

## 概述

项目已实现完整的三阶段训练流程，包括自编码器预训练、CBAM微调和融合层训练。所有配置通过 `train_configs.json` 统一管理。

## 三阶段训练流程

### 阶段1: 自编码器预训练

**目标**: 学习图像重建的基础能力

**特点**:
- 输入：仅可见光图像
- 模型：编码器（不含CBAM）+ 解码器
- 可训练参数：全部参数
- 损失函数：**L1 Loss + SSIM Loss**
- 禁用：Gradient Loss, TV Loss

**配置位置**: `train_configs.json` → `stage1`

```json
{
    "stage1": {
        "name": "autoencoder_pretraining",
        "use_attention": false,
        "trainable_params": "all",
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 100.0},
            "gradient_loss": {"enabled": false, "weight": 0.0},
            "tv_loss": {"enabled": false, "weight": 0.0}
        }
    }
}
```

---

### 阶段2: CBAM微调

**目标**: 增强模型的特征提取能力

**特点**:
- 输入：仅可见光图像
- 模型：编码器（含CBAM）+ 解码器
- 可训练参数：仅CBAM模块（冻结主干权重）
- 损失函数：**L1 Loss + SSIM Loss**
- 禁用：Gradient Loss, TV Loss

**配置位置**: `train_configs.json` → `stage2`

```json
{
    "stage2": {
        "name": "cbam_finetuning",
        "use_attention": true,
        "trainable_params": "cbam_only",
        "resume_from_stage1": true,
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 100.0},
            "gradient_loss": {"enabled": false, "weight": 0.0},
            "tv_loss": {"enabled": false, "weight": 0.0}
        }
    }
}
```

---

### 阶段3: 融合层训练

**目标**: 训练融合层，实现红外-可见光图像融合

**特点**:
- 输入：红外+可见光图像对
- 模型：编码器（含CBAM）+ 融合层 + 解码器
- 可训练参数：Decoder + Fusion Layer（冻结Encoder和CBAM）
- 损失函数：**L1 Loss + SSIM Loss + Gradient Loss + TV Loss**

**融合策略配置**:

在 `train_configs.json` 中配置融合策略：

```json
{
    "stage3": {
        "fusion_config": {
            "strategy": "l1_norm",
            "available_strategies": [
                "addition",
                "l1_norm",
                "weighted_average"
            ]
        }
    }
}
```

**可用融合策略**:

1. **addition**: 加法融合
   - 简单直接的特征融合方式
   - 适合快速实验

2. **l1_norm**: L1范数加权融合（推荐）
   - 根据特征强度自适应加权
   - 保留更多细节信息
   - 适合大多数场景

3. **weighted_average**: 加权平均融合
   - 使用可学习的权重融合特征
   - 需要更多训练数据
   - 可能获得更好的平衡

**配置位置**: `train_configs.json` → `stage3`

```json
{
    "stage3": {
        "name": "fusion_training",
        "use_attention": true,
        "trainable_params": "decoder_and_fusion",
        "resume_from_stage2": true,
        "fusion_config": {
            "strategy": "l1_norm"
        },
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 100.0},
            "gradient_loss": {"enabled": true, "weight": 5.0},
            "tv_loss": {"enabled": true, "weight": 0.1}
        }
    }
}
```

---

## 使用方法

### 1. 完整三阶段训练

```bash
python run_train.py --train_all_stages
```

这将自动执行：
1. 阶段1: 自编码器预训练（30 epochs）
2. 阶段2: CBAM微调（40 epochs）
3. 阶段3: 融合层训练（50 epochs）

### 2. 单阶段训练

#### 阶段1训练
```bash
python run_train.py --stage 1
```

#### 阶段2训练
```bash
python run_train.py --stage 2 --resume_stage1 ./runs/stage1_autoencoder/checkpoints/best.pth
```

#### 阶段3训练
```bash
python run_train.py --stage 3 --resume_stage2 ./runs/stage2_cbam/checkpoints/best.pth
```

### 3. 自定义配置训练

#### 使用自定义配置文件
```bash
python run_train.py --config my_custom_config.json --stage 3
```

#### 修改融合策略
在 `train_configs.json` 中修改：

```json
{
    "stage3": {
        "fusion_config": {
            "strategy": "weighted_average"  // 改为weighted_average
        }
    }
}
```

---

## 配置参数说明

### 全局参数

```json
{
    "training": {
        "device": "cuda",           // 计算设备
        "batch_size": 16,           // 批量大小
        "num_workers": 8,            // 数据加载线程数
        "base_dir": "./runs"         // 输出目录
    }
}
```

### 优化器参数

```json
{
    "optimizer": {
        "type": "AdamW",
        "weight_decay": 1e-4,
        "use_lr_decay": true,        // 学习率衰减
        "warmup_epochs": 5,          // 预热轮数
        "use_gradient_clipping": true // 梯度裁剪
    }
}
```

### 损失函数权重

```json
{
    "loss_function": {
        "use_adaptive_weights": false,
        "manual_weights_mode": true,
        "weights": {
            "l1_weight": 1.0,
            "ssim_weight": 100.0,
            "grad_weight": 5.0,
            "tv_weight": 0.1
        }
    }
}
```

---

## 损失函数说明

### L1 Loss (Pixel Loss)
- **作用**: 像素级重建损失
- **权重范围**: 0.1 - 10.0
- **调整建议**: 
  - 图像模糊 → 增加权重
  - 图像过锐 → 减小权重

### SSIM Loss (Structural Similarity)
- **作用**: 结构相似性损失
- **权重范围**: 10.0 - 1000.0
- **调整建议**:
  - 结构失真 → 增加权重
  - 过度平滑 → 减小权重

### Gradient Loss
- **作用**: 边缘和细节保持
- **权重范围**: 1.0 - 10.0
- **调整建议**:
  - 边缘模糊 → 增加权重
  - 边缘过锐 → 减小权重

### TV Loss (Total Variation)
- **作用**: 图像平滑
- **权重范围**: 0.01 - 1.0
- **调整建议**:
  - 噪声过多 → 增加权重
  - 过度平滑 → 减小权重

---

## 测试验证

运行测试脚本验证配置：

```bash
python test/test_three_stage_config.py
```

测试内容包括：
1. 阶段1配置验证（无CBAM，L1+SSIM）
2. 阶段2配置验证（CBAM，L1+SSIM）
3. 阶段3配置验证（融合层，L1+SSIM+Gradient+TV）
4. 融合策略验证

---

## 故障排查

### 问题1: 阶段3训练失败

**可能原因**:
- 阶段2模型文件不存在
- 数据集路径错误
- 融合策略配置错误

**解决方案**:
```bash
# 检查阶段2模型是否存在
ls ./runs/stage2_cbam/checkpoints/best.pth

# 检查数据集路径
python -c "from configs_loader import TrainingConfig; print(TrainingConfig().to_dict()['dataset'])"

# 修改融合策略
# 在train_configs.json中尝试不同的融合策略
```

### 问题2: 融合效果不佳

**可能原因**:
- 融合策略不适合当前数据
- 损失函数权重配置不当
- 训练轮数不足

**解决方案**:
1. 尝试不同的融合策略
2. 调整损失函数权重
3. 增加训练轮数
4. 使用数据增强

### 问题3: 内存不足

**解决方案**:
```json
{
    "training": {
        "batch_size": 8  // 减小批量大小
    }
}
```

---

## 融合策略对比

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| addition | 简单直接 | 快速实验 |
| l1_norm | 自适应加权 | 大多数场景（推荐） |
| weighted_average | 可学习权重 | 数据充足时 |

---

## 联系与支持

如有问题，请查看：
- `test/test_three_stage_config.py`: 测试脚本
- `run_train.py`: 训练脚本源码
- `train_configs.json`: 训练配置文件
- `LOSS_CONFIG_GUIDE.md`: 损失函数配置指南
