# 损失函数配置使用指南

## 概述

项目已支持通过配置文件灵活配置阶段一和阶段二的损失函数。用户可以在 `train_configs.json` 中独立设置每个损失函数的启用状态和权重。

## 功能特性

### 支持的损失函数

1. **L1 Loss (Pixel Loss)**: 像素级重建损失
2. **SSIM Loss**: 结构相似性损失
3. **Gradient Loss**: 梯度损失
4. **TV Loss (Total Variation)**: 全变分损失

### 配置位置

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 500.0},
            "gradient_loss": {"enabled": true, "weight": 5.0},
            "tv_loss": {"enabled": true, "weight": 0.1}
        }
    },
    "stage2": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 500.0},
            "gradient_loss": {"enabled": true, "weight": 5.0},
            "tv_loss": {"enabled": true, "weight": 0.1}
        }
    }
}
```

---

## 使用方法

### 1. 启用/禁用损失函数

在配置文件中设置 `enabled` 字段：

- `true`: 启用该损失函数
- `false`: 禁用该损失函数

**示例：禁用Gradient Loss和TV Loss**

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 500.0},
            "gradient_loss": {"enabled": false, "weight": 5.0},
            "tv_loss": {"enabled": false, "weight": 0.1}
        }
    }
}
```

### 2. 调整损失函数权重

在配置文件中设置 `weight` 字段：

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 2.0},
            "ssim_loss": {"enabled": true, "weight": 200.0}
        }
    }
}
```

---

## 推荐配置

### 阶段一（自编码器预训练）

#### 方案1：仅L1 + SSIM（轻量级）
```json
{
    "l1_loss": {"enabled": true, "weight": 1.0},
    "ssim_loss": {"enabled": true, "weight": 100.0},
    "gradient_loss": {"enabled": false, "weight": 0.0},
    "tv_loss": {"enabled": false, "weight": 0.0}
}
```

**适用场景**: 快速训练，验证模型基础性能

#### 方案2：L1 + SSIM + Gradient（平衡型）
```json
{
    "l1_loss": {"enabled": true, "weight": 1.0},
    "ssim_loss": {"enabled": true, "weight": 100.0},
    "gradient_loss": {"enabled": true, "weight": 5.0},
    "tv_loss": {"enabled": false, "weight": 0.0}
}
```

**适用场景**: 需要更好的边缘保持

#### 方案3：L1 + SSIM + Gradient + TV（完整型）
```json
{
    "l1_loss": {"enabled": true, "weight": 1.0},
    "ssim_loss": {"enabled": true, "weight": 500.0},
    "gradient_loss": {"enabled": true, "weight": 5.0},
    "tv_loss": {"enabled": true, "weight": 0.1}
}
```

**适用场景**: 需要平滑输出，减少噪声

### 阶段二（CBAM微调）

阶段二的损失函数配置与阶段一类似，建议使用相同的配置方案。

---

## 权重调整建议

### 权重范围

- **L1 Loss**: 0.1 - 10.0
- **SSIM Loss**: 10.0 - 1000.0
- **Gradient Loss**: 1.0 - 10.0
- **TV Loss**: 0.01 - 1.0

### 调整原则

1. **如果重建图像模糊**
   - 增加 L1 Loss 权重
   - 减小 TV Loss 权重

2. **如果丢失细节**
   - 增加 SSIM Loss 权重
   - 增加 Gradient Loss 权重

3. **如果噪声太多**
   - 增加 TV Loss 权重
   - 减小 L1 Loss 权重

4. **如果结构失真**
   - 增加 SSIM Loss 权重
   - 减小 L1 Loss 权重

---

## 测试验证

运行测试脚本验证配置：

```bash
python test/test_loss_config.py
```

测试包括：
1. 配置文件加载
2. 损失配置结构验证
3. 损失函数计算
4. 不同损失函数组合
5. 配置修改验证

---

## 常见问题

### Q: 为什么SSIM Loss计算结果为NaN？

A: SSIM对输入值范围敏感，确保输入张量值在[0, 1]范围内。可以在计算前添加：

```python
pred = torch.clamp(pred, 0, 1)
target = torch.clamp(target, 0, 1)
```

### Q: 如何选择最佳的损失函数组合？

A: 建议从简单配置开始，逐步增加复杂度：
1. 先使用 L1 Loss 验证模型基础性能
2. 添加 SSIM Loss 改善结构保持
3. 根据需要添加 Gradient Loss 和 TV Loss

### Q: 权重设置过大或过小会有什么影响？

A:
- **权重过大**: 训练不稳定，可能发散
- **权重过小**: 该损失项几乎不起作用

建议从默认值开始，逐步微调。

---

## 配置示例

### 示例1：快速训练配置

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 50.0},
            "gradient_loss": {"enabled": false, "weight": 0.0},
            "tv_loss": {"enabled": false, "weight": 0.0}
        }
    }
}
```

### 示例2：高质量训练配置

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 500.0},
            "gradient_loss": {"enabled": true, "weight": 5.0},
            "tv_loss": {"enabled": true, "weight": 0.1}
        }
    }
}
```

### 示例3：边缘保持训练配置

```json
{
    "stage1": {
        "loss_config": {
            "l1_loss": {"enabled": true, "weight": 1.0},
            "ssim_loss": {"enabled": true, "weight": 100.0},
            "gradient_loss": {"enabled": true, "weight": 10.0},
            "tv_loss": {"enabled": false, "weight": 0.0}
        }
    }
}
```

---

## 联系与支持

如有问题，请查看：
- `test/test_loss_config.py`: 测试脚本
- `run_train.py`: 训练脚本源码
- `train_configs.json`: 训练配置文件
