# 三阶段训练流程实现总结

## 📋 概述

根据 `docs/TRAINING_FLOWCHART.md` 中定义的训练阶段流程和规范，已成功对训练程序 `run_train.py` 进行系统性修改，实现完整的三阶段训练流程，并确保与 `train/` 文件夹中的所有组件正确集成。

## ✅ 完成的修改

### 1. 新增文件

#### 1.1 单图像数据集类
**文件**: `utils/util_dataset_single.py`

**功能**:
- 提供 `SingleImageDataset` 类，用于加载单张图像（阶段一和阶段二）
- 支持 `single_image_transform` 函数，提供图像预处理和数据增强
- 适用于自编码器预训练和CBAM微调阶段

**关键特性**:
- 自动过滤空文件
- 支持灰度和彩色模式
- 支持数据增强（颜色抖动、翻转等）

#### 1.2 融合层模块
**文件**: `models/fusion_layer.py`

**功能**:
- 提供多种融合策略用于阶段三（端到端融合）
- 支持加法融合、L1-norm融合、加权平均融合

**融合策略**:
1. **AdditionFusion**: 直接相加融合
   - 公式: `F = feature_ir + feature_vis`
   
2. **L1NormFusion**: 基于L1范数的自适应融合
   - 公式: `weight = |feature| / (|feature_ir| + |feature_vis| + epsilon)`
   - `F = weight_ir * feature_ir + weight_vis * feature_vis`
   
3. **WeightedAverageFusion**: 可学习权重的加权平均
   - 公式: `F = alpha * feature_ir + (1 - alpha) * feature_vis`
   - alpha为可学习参数

#### 1.3 带融合层的DenseFuse模型
**文件**: `models/DenseFuse_with_fusion.py`

**功能**:
- 实现 `DenseFuseWithFusion` 类，用于阶段三（端到端融合）
- 支持两个共享权重的编码器、一个融合层和一个解码器
- 支持参数冻结/解冻功能

**关键方法**:
- `freeze_backbone()`: 冻结编码器和解码器的主干网络
- `unfreeze_backbone()`: 解冻主干网络
- `freeze_cbam()`: 冻结CBAM模块
- `unfreeze_cbam()`: 解冻CBAM模块
- `freeze_fusion_layer()`: 冻结融合层
- `unfreeze_fusion_layer()`: 解冻融合层
- `get_trainable_params()`: 获取所有可训练参数
- `get_backbone_params()`: 获取主干网络参数
- `get_cbam_params()`: 获取CBAM参数
- `get_fusion_params()`: 获取融合层参数

#### 1.4 三阶段训练测试脚本
**文件**: `test_three_stage_training.py`

**功能**:
- 测试三阶段训练流程的所有关键组件
- 验证数据集加载、模型初始化、参数冻结、融合层、权重传递等功能

**测试结果**: 5/5 通过 (100.0%)

### 2. 修改的文件

#### 2.1 DenseFuse模型增强
**文件**: `models/DenseFuse.py`

**新增功能**:
- 添加参数冻结/解冻方法
- 添加参数分组获取方法

**新增方法**:
```python
def freeze_backbone(self)
def unfreeze_backbone(self)
def freeze_cbam(self)
def unfreeze_cbam(self)
def freeze_decoder(self)
def unfreeze_decoder(self)
def get_trainable_params(self)
def get_backbone_params(self)
def get_cbam_params(self)
def get_encoder_params(self)
def get_decoder_params(self)
```

#### 2.2 模型包初始化
**文件**: `models/__init__.py`

**新增功能**:
- 导出融合层相关类和函数
- 添加 `fuse_model_with_fusion_layer()` 工厂函数

#### 2.3 三阶段训练主程序
**文件**: `run_train.py`

**完全重写**: 实现完整的三阶段训练流程

**新增功能**:
1. **阶段一：自编码器预训练**
   - 输入：单张可见光图像
   - 模型：编码器（不含CBAM）+ 解码器
   - 目标：重建损失
   - 可训练参数：全部参数

2. **阶段二：CBAM微调**
   - 输入：单张可见光图像
   - 模型：编码器（含CBAM）+ 解码器
   - 目标：重建损失
   - 可训练参数：仅CBAM模块（冻结主干权重）

3. **阶段三：端到端融合**
   - 输入：红外+可见光图像对
   - 模型：编码器（含CBAM）+ 融合层 + 解码器
   - 目标：融合损失
   - 可训练参数：全部参数或部分参数（根据end_to_end_finetune参数）

**新增命令行参数**:
```bash
# 训练阶段选择
--train_all_stages          # 执行完整的三阶段训练
--stage {1,2,3}           # 选择训练阶段

# 阶段一参数
--stage1_epochs            # 阶段一训练轮数 (默认: 80)
--stage1_lr               # 阶段一学习率 (默认: 2e-4)

# 阶段二参数
--stage2_epochs            # 阶段二训练轮数 (默认: 40)
--stage2_lr               # 阶段二学习率 (默认: 1e-4)

# 阶段三参数
--stage3_epochs            # 阶段三训练轮数 (默认: 60)
--stage3_lr               # 阶段三学习率 (默认: 1e-4)
--fusion_strategy         # 融合策略 (默认: l1_norm)
--end_to_end_finetune    # 阶段三是否进行端到端微调

# 恢复训练参数
--resume_stage1          # 阶段一模型路径（用于阶段二恢复）
--resume_stage2          # 阶段二模型路径（用于阶段三恢复）
--resume_path            # 恢复训练的模型路径（用于当前阶段）

# 输出参数
--base_dir               # 基础输出目录 (默认: ./runs)
```

**使用方法**:
```bash
# 完整三阶段训练
python run_train.py --train_all_stages

# 单独训练某个阶段
python run_train.py --stage 1
python run_train.py --stage 2 --resume_stage1 ./runs/stage1_autoencoder/checkpoints/stage1_final.pth
python run_train.py --stage 3 --resume_stage2 ./runs/stage2_cbam/checkpoints/stage2_final.pth

# 自定义融合策略
python run_train.py --train_all_stages --fusion_strategy addition
python run_train.py --train_all_stages --fusion_strategy l1_norm
python run_train.py --train_all_stages --fusion_strategy weighted_average

# 端到端微调
python run_train.py --stage 3 --resume_stage2 ./runs/stage2_cbam/checkpoints/stage2_final.pth --end_to_end_finetune
```

## 📊 训练流程图

```
阶段一: 自编码器预训练
├─ 输入: 单张可见光图像
├─ 模型: 编码器(不含CBAM) + 解码器
├─ 目标: 重建损失
├─ 可训练参数: 全部参数
└─ 输出: stage1_final.pth

阶段二: CBAM微调
├─ 输入: 单张可见光图像
├─ 模型: 编码器(含CBAM) + 解码器
├─ 目标: 重建损失
├─ 可训练参数: 仅CBAM模块
├─ 加载: stage1_final.pth
└─ 输出: stage2_final.pth

阶段三: 端到端融合
├─ 输入: 红外+可见光图像对
├─ 模型: 编码器(含CBAM) + 融合层 + 解码器
├─ 目标: 融合损失
├─ 可训练参数: 全部参数或仅融合层
├─ 加载: stage2_final.pth
└─ 输出: stage3_final.pth
```

## 🔧 技术细节

### 参数冻结机制

**阶段一**: 所有参数可训练
- 编码器参数: 可训练
- 解码器参数: 可训练

**阶段二**: 仅CBAM参数可训练
- 编码器主干参数: 冻结
- CBAM参数: 可训练
- 解码器参数: 冻结

**阶段三**: 根据参数决定
- 不启用端到端微调:
  - 编码器主干参数: 冻结
  - CBAM参数: 冻结
  - 融合层参数: 可训练
  - 解码器参数: 冻结
- 启用端到端微调:
  - 所有参数: 可训练

### 损失函数

**阶段一和阶段二**: 重建损失
```
L = λ_ssim × L_ssim + λ_pixel × L_pixel
```

**阶段三**: 融合损失
```
L = λ_1 × L_1 + λ_2 × (1-SSIM) + λ_3 × L_grad + λ_4 × L_TV
```

### 输出目录结构

```
runs/
├── stage1_autoencoder/
│   ├── checkpoints/
│   │   ├── stage1_final.pth
│   │   └── epochXXX-lossX.XXX.pth
│   └── logs/
├── stage2_cbam/
│   ├── checkpoints/
│   │   ├── stage2_final.pth
│   │   └── epochXXX-lossX.XXX.pth
│   └── logs/
└── stage3_fusion/
    ├── checkpoints/
    │   ├── stage3_final.pth
    │   └── epochXXX-lossX.XXX.pth
    └── logs/
```

## ✅ 测试验证

### 测试项目
1. ✅ 单图像数据集加载
2. ✅ 模型初始化和参数冻结
3. ✅ 融合层功能
4. ✅ 带融合层的模型
5. ✅ 阶段间权重传递

### 测试结果
- 通过率: 5/5 (100.0%)
- 所有功能正常工作
- 梯度计算正确
- 参数冻结/解冻功能正常

## 📝 使用示例

### 完整三阶段训练
```bash
python run_train.py --train_all_stages
```

### 单独训练阶段一
```bash
python run_train.py --stage 1 --stage1_epochs 80 --stage1_lr 2e-4
```

### 单独训练阶段二
```bash
python run_train.py --stage 2 \
    --resume_stage1 ./runs/stage1_autoencoder/checkpoints/stage1_final.pth \
    --stage2_epochs 40 --stage2_lr 1e-4
```

### 单独训练阶段三（仅融合层）
```bash
python run_train.py --stage 3 \
    --resume_stage2 ./runs/stage2_cbam/checkpoints/stage2_final.pth \
    --stage3_epochs 60 --stage3_lr 1e-4 \
    --fusion_strategy l1_norm
```

### 单独训练阶段三（端到端微调）
```bash
python run_train.py --stage 3 \
    --resume_stage2 ./runs/stage2_cbam/checkpoints/stage2_final.pth \
    --stage3_epochs 60 --stage3_lr 1e-4 \
    --fusion_strategy l1_norm \
    --end_to_end_finetune
```

## 🎯 总结

已成功实现三阶段训练流程，包括：

1. ✅ 创建单图像数据集类用于阶段一和阶段二
2. ✅ 修改DenseFuse模型支持参数冻结
3. ✅ 创建融合层模块支持多种融合策略
4. ✅ 创建带融合层的DenseFuse模型
5. ✅ 重写run_train.py实现完整三阶段训练
6. ✅ 添加丰富的命令行参数支持
7. ✅ 实现阶段间权重传递机制
8. ✅ 通过所有功能测试

所有修改均与 `train/` 文件夹中的组件正确集成，符合 `docs/TRAINING_FLOWCHART.md` 中定义的训练阶段流程和规范。

## 📚 相关文档

- [训练流程图](docs/TRAINING_FLOWCHART.md)
- [项目README](README.md)
- [训练模块](train/)
- [模型模块](models/)

---

**文档版本**: v1.0.0  
**更新时间**: 2026-03-30  
**作者**: wokaka209
