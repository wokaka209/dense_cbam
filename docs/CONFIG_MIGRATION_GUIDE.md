# 配置迁移使用说明

## 概述

项目已完成配置迁移，从命令行参数解析迁移到外部JSON配置文件。

### 主要变更

1. **创建统一配置加载器**: `configs_loader.py`
2. **训练配置**: `train_configs.json`
3. **融合配置**: `fusion_configs.json`
4. **简化命令行接口**: 保留必要的命令行参数，核心配置通过JSON文件管理

---

## 使用方法

### 1. 训练脚本使用

#### 完整两阶段训练
```bash
python run_train.py --train_all_stages
```

#### 单阶段训练
```bash
# 阶段一：自编码器预训练
python run_train.py --stage 1

# 阶段二：CBAM微调
python run_train.py --stage 2 --resume_stage1 ./runs/stage1_autoencoder/checkpoints/best.pth
```

#### 使用自定义配置文件
```bash
python run_train.py --config custom_train_config.json --stage 1
```

#### 命令行参数覆盖
```bash
python run_train.py --stage 1 --resume_path ./runs/stage1_autoencoder/checkpoints/best.pth
```

### 2. 融合脚本使用

#### 批量融合模式
```bash
python run_fusion.py --batch
```

#### 单对融合模式
```bash
python run_fusion.py --single --ir test_ir.png --vi test_vi.png --output result.png
```

#### 使用自定义配置文件
```bash
python run_fusion.py --config custom_fusion_config.json --batch
```

#### 命令行参数覆盖
```bash
python run_fusion.py --batch --batch_size 4 --strategy l1_norm
```

---

## 配置文件说明

### train_configs.json

训练配置文件包含以下主要部分：

```json
{
    "dataset": {
        "ir_path": "红外图像路径",
        "vi_path": "可见光图像路径",
        "gray": false
    },
    "training": {
        "device": "cuda",
        "batch_size": 16,
        "num_workers": 8,
        "base_dir": "./runs"
    },
    "stage1": {
        "epochs": 30,
        "learning_rate": 1e-4
    },
    "stage2": {
        "epochs": 40,
        "learning_rate": 1e-4,
        "resume_stage1_path": "阶段一模型路径"
    },
    "optimizer": {
        "type": "AdamW",
        "warmup_epochs": 5,
        "use_lr_decay": true,
        "use_gradient_clipping": true
    },
    "loss_function": {
        "use_adaptive_weights": true,
        "weights": {
            "l1_weight": 1.0,
            "ssim_weight": 100.0
        }
    }
}
```

### fusion_configs.json

融合配置文件包含以下主要部分：

```json
{
    "io_paths": {
        "single_mode": {
            "ir_image": "",
            "vi_image": "",
            "output": "data_result/fusion_optimized"
        },
        "batch_mode": {
            "ir_dir": "红外图像目录",
            "vi_dir": "可见光图像目录",
            "output_dir": "data_result/fusion_optimized"
        }
    },
    "model": {
        "model_path": "./runs/stage2_cbam/checkpoints/best.pth",
        "fusion_strategy": "l1_norm"
    },
    "device": {
        "type": "cuda"
    },
    "performance": {
        "batch_size": 1,
        "show_stats": true
    }
}
```

---

## 配置优先级

命令行参数会覆盖JSON配置文件中的对应值：

1. 命令行参数（最高优先级）
2. JSON配置文件
3. JSON默认值（最低优先级）

---

## 配置验证

配置文件在加载时会自动验证完整性：

```python
from configs_loader import TrainingConfig, FusionConfig

# 加载并验证训练配置
train_config = TrainingConfig()
if train_config.validate():
    print("训练配置验证通过")

# 加载并验证融合配置
fusion_config = FusionConfig()
if fusion_config.validate():
    print("融合配置验证通过")
```

---

## 优点

1. **集中管理**: 所有配置集中在JSON文件中，便于维护
2. **版本控制**: 配置文件可以提交到Git，便于追踪变更
3. **环境切换**: 不同环境（开发/测试/生产）使用不同配置文件
4. **代码简洁**: 删除大量命令行参数解析代码
5. **易于扩展**: 新增配置项只需修改JSON文件

---

## 故障排查

### 配置文件不存在
```
[ERROR] 配置文件不存在: train_configs.json
```
**解决方案**: 确保配置文件在项目根目录下

### JSON格式错误
```
[ERROR] JSON格式错误: ...
```
**解决方案**: 检查JSON文件语法，确保格式正确

### 配置验证失败
```
[ERROR] 配置验证失败
```
**解决方案**: 检查配置文件是否包含所有必需字段

---

## 迁移注意事项

1. **备份**: 如果有自定义的训练脚本，备份后再升级
2. **测试**: 首次使用建议先用小数据集测试
3. **文档**: 记录自定义配置项的使用方式
4. **版本**: 建议在Git中标记配置文件版本

---

## 联系与支持

如有问题，请查看：
- `configs_loader.py`: 配置加载器源码
- `train_configs.json`: 训练配置示例
- `fusion_configs.json`: 融合配置示例
