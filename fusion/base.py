# -*- coding: utf-8 -*-
"""
@file name:fusion/base.py
@desc: 融合器基类 - 定义融合策略的接口和注册表
@Writer: wokaka209
@Date: 2026-03-26

功能说明：
---------
本模块定义了融合策略的基类和注册表机制：
- BaseFusionStrategy: 融合策略抽象基类
- FusionStrategyRegistry: 融合策略注册表

设计模式：
---------
采用策略模式和注册表模式：
- 策略模式：允许在运行时切换融合策略
- 注册表模式：方便策略的扩展和管理

使用示例：
---------
```python
from fusion.base import BaseFusionStrategy, FusionStrategyRegistry

# 定义新策略
class MyStrategy(BaseFusionStrategy):
    def fuse(self, feature1, feature2):
        return (feature1 + feature2) / 2

# 注册策略
FusionStrategyRegistry.register('my_strategy', MyStrategy)

# 使用策略
strategy = FusionStrategyRegistry.get('my_strategy')
```

依赖：
-----
- torch
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
import torch


class BaseFusionStrategy(ABC):
    """
    融合策略抽象基类
    
    功能：
    -----
    定义所有融合策略必须实现的接口。
    任何融合策略都应继承此类并实现 fuse 方法。
    
    设计原则：
    ---------
    - 策略模式：每种融合方法封装为独立策略
    - 开闭原则：新增策略无需修改现有代码
    - 依赖倒置：通过抽象接口依赖，不依赖具体实现
    
    属性：
    -----
    name : str
        策略名称
    description : str
        策略描述
    
    方法：
    -----
    fuse(feature1, feature2):
        执行特征融合（必须实现）
    get_config():
        获取策略配置（可选重写）
    
    使用示例：
    ---------
    ```python
    class WeightedAverageStrategy(BaseFusionStrategy):
        def __init__(self, weight1=0.5, weight2=0.5):
            self.name = "weighted_average"
            self.description = "加权平均融合策略"
            self.weight1 = weight1
            self.weight2 = weight2
        
        def fuse(self, feature1, feature2):
            return self.weight1 * feature1 + self.weight2 * feature2
    ```
    """
    
    def __init__(self):
        """初始化融合策略基类"""
        self.name = self.__class__.__name__
        self.description = "基类融合策略"
    
    @abstractmethod
    def fuse(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """
        执行特征融合
        
        这是所有融合策略必须实现的核心方法。
        
        Args:
            feature1: 第一个特征张量 [B, C, H, W]
            feature2: 第二个特征张量 [B, C, H, W]
        
        Returns:
            torch.Tensor: 融合后的特征张量 [B, C, H, W]
        
        Raises:
            NotImplementedError: 如果子类没有实现此方法
        
        示例：
        ------
        ```python
        class SimpleAverageStrategy(BaseFusionStrategy):
            def fuse(self, feature1, feature2):
                return (feature1 + feature2) / 2
        ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 fuse 方法"
        )
    
    def get_config(self) -> Dict:
        """
        获取策略配置
        
        可被子类重写以返回自定义配置。
        
        Returns:
            dict: 策略配置字典
        
        示例：
        ------
        ```python
        def get_config(self):
            return {
                'name': self.name,
                'description': self.description,
                # 子类可以添加更多配置项
            }
        ```
        """
        return {
            'name': self.name,
            'description': self.description
        }
    
    def __repr__(self) -> str:
        """返回策略的字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}')"


class FusionStrategyRegistry:
    """
    融合策略注册表
    
    功能：
    -----
    提供融合策略的注册和获取机制：
    - 注册新策略
    - 获取已注册策略
    - 列出所有策略
    - 检查策略是否存在
    
    使用方式：
    ---------
    ```python
    # 注册策略
    FusionStrategyRegistry.register('my_strategy', MyStrategyClass)
    
    # 获取策略
    strategy = FusionStrategyRegistry.get('my_strategy')
    
    # 检查存在
    exists = FusionStrategyRegistry.exists('my_strategy')
    
    # 列出所有
    all_strategies = FusionStrategyRegistry.list_strategies()
    
    # 创建实例
    instance = FusionStrategyRegistry.create('my_strategy', *args, **kwargs)
    ```
    
    预注册策略：
    -----------
    以下策略已预注册：
    - 'enhanced_l1': EnhancedL1Strategy
    - 'multi_scale': MultiScaleStrategy
    - 'gradient': GradientGuidedStrategy
    - 'hybrid': HybridFusionStrategy
    
    Note:
    -----
    使用 @FusionStrategyRegistry.register('name') 装饰器注册
    """
    
    _strategies: Dict[str, Type[BaseFusionStrategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        注册融合策略的装饰器
        
        用法：
        -----
        ```python
        @FusionStrategyRegistry.register('my_strategy')
        class MyStrategy(BaseFusionStrategy):
            def fuse(self, feature1, feature2):
                return (feature1 + feature2) / 2
        ```
        
        Args:
            name: 策略名称（用于后续获取）
        
        Returns:
            装饰器函数
        """
        def decorator(strategy_class: Type[BaseFusionStrategy]):
            if not issubclass(strategy_class, BaseFusionStrategy):
                raise TypeError(
                    f"{strategy_class.__name__} 必须继承 BaseFusionStrategy"
                )
            
            cls._strategies[name.lower()] = strategy_class
            return strategy_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseFusionStrategy]]:
        """
        获取指定名称的策略类
        
        Args:
            name: 策略名称
        
        Returns:
            Type[BaseFusionStrategy] 或 None: 策略类
        
        示例：
        ------
        ```python
        strategy_class = FusionStrategyRegistry.get('enhanced_l1')
        if strategy_class:
            strategy = strategy_class()
        ```
        """
        return cls._strategies.get(name.lower())
    
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Optional[BaseFusionStrategy]:
        """
        创建策略实例
        
        Args:
            name: 策略名称
            *args: 传递给策略构造器的位置参数
            **kwargs: 传递给策略构造器的关键字参数
        
        Returns:
            BaseFusionStrategy 或 None: 策略实例
        
        示例：
        ------
        ```python
        strategy = FusionStrategyRegistry.create(
            'enhanced_l1',
            weight1=0.6,
            weight2=0.4
        )
        ```
        """
        strategy_class = cls.get(name)
        
        if strategy_class is None:
            raise ValueError(
                f"策略 '{name}' 未注册。可用策略: {cls.list_strategies()}"
            )
        
        return strategy_class(*args, **kwargs)
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """
        检查策略是否存在
        
        Args:
            name: 策略名称
        
        Returns:
            bool: 策略是否存在
        
        示例：
        ------
        ```python
        if FusionStrategyRegistry.exists('enhanced_l1'):
            strategy = FusionStrategyRegistry.create('enhanced_l1')
        ```
        """
        return name.lower() in cls._strategies
    
    @classmethod
    def list_strategies(cls) -> list:
        """
        列出所有已注册的策略
        
        Returns:
            list: 策略名称列表
        
        示例：
        ------
        ```python
        strategies = FusionStrategyRegistry.list_strategies()
        print("可用策略:", strategies)
        ```
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        取消注册策略
        
        Args:
            name: 策略名称
        
        Returns:
            bool: 是否成功取消注册
        
        示例：
        ------
        ```python
        success = FusionStrategyRegistry.unregister('my_strategy')
        ```
        """
        if name.lower() in cls._strategies:
            del cls._strategies[name.lower()]
            return True
        return False
    
    @classmethod
    def clear(cls):
        """清空所有注册的策略"""
        cls._strategies.clear()
    
    @classmethod
    def get_info(cls, name: str) -> Optional[Dict]:
        """
        获取策略的详细信息
        
        Args:
            name: 策略名称
        
        Returns:
            dict 或 None: 策略信息，包含名称、描述和配置
        """
        strategy_class = cls.get(name)
        
        if strategy_class is None:
            return None
        
        # 创建临时实例以获取配置
        try:
            instance = strategy_class()
            return {
                'name': name,
                'class': strategy_class.__name__,
                'description': getattr(instance, 'description', ''),
                'config': instance.get_config()
            }
        except:
            return {
                'name': name,
                'class': strategy_class.__name__
            }
