# Tools 工具目录

本目录包含DenseFuse项目中的各种实用工具和辅助脚本。

## 📁 目录结构

```
tools/
├── analyze_loss/           # 损失分析工具
│   ├── image/             # 生成的损失曲线图像
│   ├── loss_analyzer.py   # TensorBoard事件文件解析工具
│   └── README.md          # 损失分析工具说明
├── image_batch_renamer/    # 图像批量重命名工具
│   ├── renamer_core.py    # 核心重命名逻辑
│   ├── renamer_gui.py     # 图形用户界面
│   └── README.md          # 重命名工具说明
└── metric calculation/     # 图像质量评估工具
    ├── iqa_results/       # 评估结果输出
    ├── checkpoints/       # 断点续传检查点
    ├── main.py           # 主评估程序
    ├── metrics.py        # 评估指标实现
    └── README.md         # 评估工具说明
```

## 🔧 工具说明

### 1. 损失分析工具 (analyze_loss)

**功能**：解析TensorBoard事件文件，生成训练损失曲线图

**主要文件**：
- `loss_analyzer.py` - 主程序文件
- `image/` - 生成的损失曲线图像保存目录

**使用方法**：
```bash
cd tools/analyze_loss
python loss_analyzer.py
```

**特性**：
- 自动过滤loss相关数据，移除学习率和权重信息
- 增强可视化效果，线条宽度优化至2.5
- 跨平台路径兼容性
- 基于事件文件时间戳自动生成文件名

### 2. 图像质量评估工具 (metric calculation)

**功能**：计算融合图像的各项质量指标

**支持指标**：
- AG (平均梯度)
- EN (信息熵)
- MI (互信息)
- SD (标准差)
- SSIM (结构相似性)
- Qabf (基于梯度的质量指标)

**使用方法**：
```bash
cd tools/metric\ calculation
python main.py
```

### 3. 图像批量重命名工具 (image_batch_renamer)

**功能**：批量重命名图像文件，支持多种命名规则

**特性**：
- 图形用户界面 (GUI)
- 支持前缀、后缀、序号等命名方式
- 实时预览重命名效果

**使用方法**：
```bash
cd tools/image_batch_renamer
python renamer_gui.py
```

## 🛠️ 安装依赖

所有工具都需要以下依赖：

```bash
pip install torch torchvision tensorboard matplotlib numpy opencv-python tqdm
```

## 📊 输出示例

### 损失分析工具输出

损失曲线图像将保存在 `tools/analyze_loss/image/` 目录下，文件名格式为：
- `loss_curve_{timestamp}.png`

### 图像质量评估输出

评估结果保存在 `tools/metric calculation/iqa_results/` 目录下，包含：
- 各项指标的平均值统计
- 总图像数量信息
- 生成时间戳

## 🔄 更新日志

### 2026-04-02
- **损失分析工具优化**：
  - 增强可视化效果，线条宽度增加至2.5
  - 过滤图表内容，仅保留loss曲线
  - 改进跨平台路径兼容性
  - 优化图片保存路径和文件名生成

## 📝 注意事项

1. **路径兼容性**：所有工具都使用 `os.path.join()` 确保跨平台兼容性
2. **依赖安装**：首次使用前请确保安装所有必要的Python包
3. **文件权限**：确保对输出目录有写入权限
4. **TensorBoard文件**：损失分析工具需要有效的TensorBoard事件文件

## 🤝 贡献指南

欢迎提交改进建议和bug报告！请确保：
- 遵循现有的代码风格
- 添加适当的文档说明
- 测试新功能在不同平台上的兼容性

## 📄 许可证

本项目采用MIT许可证，详见各工具目录中的LICENSE文件。