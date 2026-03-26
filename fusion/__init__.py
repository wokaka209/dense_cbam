# -*- coding: utf-8 -*-
"""
@file name:fusion/__init__.py
@desc: 融合模块包初始化
@Writer: wokaka209
@Date: 2026-03-26

模块说明：
---------
本包提供图像融合的核心功能，包括：
- 图像预处理和后处理
- 融合器基类和接口定义
- 融合策略实现（性能优化版）
- 融合引擎（整合所有功能）

包结构：
-------
fusion/
├── __init__.py                # 包初始化（当前文件）
├── base.py                    # 融合器基类和注册表
├── preprocessor.py            # 图像预处理器
├── postprocessor.py           # 图像后处理器
├── strategies_optimized.py    # 融合策略（性能优化版）
└── fusion_engine.py           # 融合引擎

性能优化说明：
-----------
strategies_optimized.py 包含性能优化版本的融合策略：
- EnhancedL1Strategy: 去除对比度计算，使用ReLU代替sqrt
- MultiScaleStrategy: 减少尺度数量到2个
- HybridFusionStrategy: 合并计算，简化权重混合

性能提升：3-6倍（从5.95秒/张优化到<1秒/张）

使用示例：
---------
```python
# 基础用法
from fusion import ImageFusionEngine, create_fusion_engine

# 使用工厂函数创建引擎
engine = create_fusion_engine(
    model_path='path/to/model.pth',
    device='cuda',
    strategy='hybrid'
)

# 单对图像融合
engine.fuse('ir.png', 'vi.png', 'output.png')

# 批量融合
engine.batch_fuse(ir_dir='data/ir', vi_dir='data/vi', output_dir='output')

# 直接使用融合策略
from fusion import EnhancedL1Strategy, MultiScaleStrategy, HybridFusionStrategy

strategy = HybridFusionStrategy()
```

详细文档：
---------
每个模块的详细说明请参阅：
- base.py: BaseFusionStrategy, FusionStrategyRegistry
- preprocessor.py: ImagePreprocessor
- postprocessor.py: ImagePostprocessor
- strategies_optimized.py: 性能优化版融合策略
- fusion_engine.py: ImageFusionEngine, create_fusion_engine
"""

# 导入基类
from .base import BaseFusionStrategy, FusionStrategyRegistry

# 导入融合策略（优化版本）
from .strategies_optimized import (
    EnhancedL1Strategy,
    MultiScaleStrategy,
    GradientGuidedStrategy,
    HybridFusionStrategy
)

# 导入预处理和后处理
from .preprocessor import ImagePreprocessor
from .postprocessor import ImagePostprocessor

# 导入融合引擎
from .fusion_engine import ImageFusionEngine, create_fusion_engine

# 定义公共API列表
__all__ = [
    # 基类和注册表
    'BaseFusionStrategy',
    'FusionStrategyRegistry',
    
    # 融合策略（优化版本）
    'EnhancedL1Strategy',
    'MultiScaleStrategy',
    'GradientGuidedStrategy',
    'HybridFusionStrategy',
    
    # 预处理器和后处理器
    'ImagePreprocessor',
    'ImagePostprocessor',
    
    # 融合引擎
    'ImageFusionEngine',
    'create_fusion_engine',
]

# 包元信息
__version__ = '2.0.0'
__author__ = 'wokaka209'
__email__ = '1325536985@qq.com'
__description__ = '红外可见光图像融合模块（性能优化版）'
__url__ = 'https://github.com/wokaka209/my_densefuse_advantive'
