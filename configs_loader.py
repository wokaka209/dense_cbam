# -*- coding: utf-8 -*-
"""
@file name:configs_loader.py
@desc: 配置文件加载器 - 提供统一的配置管理接口
@Writer: wokaka209
@Date: 2026-03-31

功能说明：
---------
本模块提供配置文件加载和管理功能，支持从JSON文件读取训练和融合配置。

使用方法：
---------
    from configs_loader import ConfigLoader
    
    # 加载训练配置
    train_config = ConfigLoader.load_train_config()
    
    # 加载融合配置
    fusion_config = ConfigLoader.load_fusion_config()
    
    # 使用配置
    batch_size = train_config.get('training', 'batch_size')
"""

import os
import json
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    配置加载器类
    
    功能：
    -----
    统一加载和管理项目的配置文件：
    - train_configs.json: 训练配置
    - fusion_configs.json: 融合配置
    
    使用方式：
    ---------
    ```python
    config = ConfigLoader.load_train_config()
    batch_size = config.training.batch_size
    ```
    """
    
    _train_config: Optional[Dict[str, Any]] = None
    _fusion_config: Optional[Dict[str, Any]] = None
    _config_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    @classmethod
    def load_json_file(cls, file_path: str) -> Dict[str, Any]:
        """
        加载JSON配置文件
        
        Args:
            file_path: JSON文件路径
        
        Returns:
            Dict[str, Any]: 解析后的配置字典
        
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON格式错误: {e}", e.doc, e.pos)
    
    @classmethod
    def load_train_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载训练配置文件
        
        Args:
            config_path: 配置文件路径（可选，默认使用项目根目录下的train_configs.json）
        
        Returns:
            Dict[str, Any]: 训练配置字典
        
        使用示例：
        ---------
        ```python
        config = ConfigLoader.load_train_config()
        
        # 访问配置
        batch_size = config['training']['batch_size']
        stage1_epochs = config['stage1']['epochs']
        ```
        """
        if cls._train_config is not None and config_path is None:
            return cls._train_config
        
        if config_path is None:
            config_path = os.path.join(cls._config_dir, 'train_configs.json')
        
        cls._train_config = cls.load_json_file(config_path)
        return cls._train_config
    
    @classmethod
    def load_fusion_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载融合配置文件
        
        Args:
            config_path: 配置文件路径（可选，默认使用项目根目录下的fusion_configs.json）
        
        Returns:
            Dict[str, Any]: 融合配置字典
        
        使用示例：
        ---------
        ```python
        config = ConfigLoader.load_fusion_config()
        
        # 访问配置
        batch_size = config['performance']['batch_size']
        fusion_strategy = config['model']['fusion_strategy']
        ```
        """
        if cls._fusion_config is not None and config_path is None:
            return cls._fusion_config
        
        if config_path is None:
            config_path = os.path.join(cls._config_dir, 'fusion_configs.json')
        
        cls._fusion_config = cls.load_json_file(config_path)
        return cls._fusion_config
    
    @classmethod
    def get_train_config(cls) -> Dict[str, Any]:
        """
        获取已加载的训练配置（不重新加载）
        
        Returns:
            Dict[str, Any]: 训练配置字典
        """
        if cls._train_config is None:
            return cls.load_train_config()
        return cls._train_config
    
    @classmethod
    def get_fusion_config(cls) -> Dict[str, Any]:
        """
        获取已加载的融合配置（不重新加载）
        
        Returns:
            Dict[str, Any]: 融合配置字典
        """
        if cls._fusion_config is None:
            return cls.load_fusion_config()
        return cls._fusion_config
    
    @classmethod
    def reload_train_config(cls) -> Dict[str, Any]:
        """
        重新加载训练配置
        
        Returns:
            Dict[str, Any]: 重新加载的训练配置字典
        """
        cls._train_config = None
        return cls.load_train_config()
    
    @classmethod
    def reload_fusion_config(cls) -> Dict[str, Any]:
        """
        重新加载融合配置
        
        Returns:
            Dict[str, Any]: 重新加载的融合配置字典
        """
        cls._fusion_config = None
        return cls.load_fusion_config()
    
    @classmethod
    def get_value(cls, config_dict: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        """
        安全获取嵌套配置值
        
        Args:
            config_dict: 配置字典
            *keys: 嵌套键路径（如：'training', 'batch_size'）
            default: 默认值（当键不存在时返回）
        
        Returns:
            Any: 配置值或默认值
        
        使用示例：
        ---------
        ```python
        config = ConfigLoader.load_train_config()
        
        # 安全获取嵌套值
        batch_size = ConfigLoader.get_value(config, 'training', 'batch_size', default=16)
        stage1_lr = ConfigLoader.get_value(config, 'stage1', 'learning_rate', default=1e-4)
        ```
        """
        current = config_dict
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    @classmethod
    def validate_train_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证训练配置完整性
        
        Args:
            config: 训练配置字典
        
        Returns:
            bool: 配置是否完整
        
        检查项：
        -----
        - 必须包含：dataset, training, stage1, stage2, optimizer, loss_function, model
        - stage1和stage2必须包含：epochs, learning_rate
        - training必须包含：device, batch_size, num_workers
        """
        required_keys = ['dataset', 'training', 'stage1', 'stage2', 'optimizer', 'loss_function', 'model']
        
        for key in required_keys:
            if key not in config:
                print(f"[ERROR] 配置缺少必需项: {key}")
                return False
        
        for stage in ['stage1', 'stage2']:
            if 'epochs' not in config[stage]:
                print(f"[ERROR] {stage}配置缺少epochs参数")
                return False
            if 'learning_rate' not in config[stage]:
                print(f"[ERROR] {stage}配置缺少learning_rate参数")
                return False
        
        for key in ['device', 'batch_size', 'num_workers']:
            if key not in config['training']:
                print(f"[ERROR] training配置缺少{key}参数")
                return False
        
        return True
    
    @classmethod
    def validate_fusion_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证融合配置完整性
        
        Args:
            config: 融合配置字典
        
        Returns:
            bool: 配置是否完整
        
        检查项：
        -----
        - 必须包含：io_paths, model, device, performance
        - model必须包含：model_path, model_name, fusion_strategy
        - performance必须包含：batch_size
        """
        required_keys = ['io_paths', 'model', 'device', 'performance']
        
        for key in required_keys:
            if key not in config:
                print(f"[ERROR] 配置缺少必需项: {key}")
                return False
        
        model_required = ['model_path', 'model_name', 'fusion_strategy']
        for key in model_required:
            if key not in config['model']:
                print(f"[ERROR] model配置缺少{key}参数")
                return False
        
        if 'batch_size' not in config['performance']:
            print("[ERROR] performance配置缺少batch_size参数")
            return False
        
        return True
    
    @classmethod
    def print_config_info(cls, config: Dict[str, Any], config_type: str = 'unknown'):
        """
        打印配置信息
        
        Args:
            config: 配置字典
            config_type: 配置类型（'train' 或 'fusion'）
        """
        print("=" * 60)
        print(f"配置文件信息: {config_type}")
        print("=" * 60)
        
        if config_type == 'train':
            print(f"\n数据集:")
            print(f"  - 红外图像路径: {cls.get_value(config, 'dataset', 'ir_path')}")
            print(f"  - 可见光图像路径: {cls.get_value(config, 'dataset', 'vi_path')}")
            print(f"  - 灰度模式: {cls.get_value(config, 'dataset', 'gray')}")
            
            print(f"\n训练参数:")
            print(f"  - 设备: {cls.get_value(config, 'training', 'device')}")
            print(f"  - 批量大小: {cls.get_value(config, 'training', 'batch_size')}")
            print(f"  - 线程数: {cls.get_value(config, 'training', 'num_workers')}")
            
            print(f"\n阶段一:")
            print(f"  - 训练轮数: {cls.get_value(config, 'stage1', 'epochs')}")
            print(f"  - 学习率: {cls.get_value(config, 'stage1', 'learning_rate')}")
            
            print(f"\n阶段二:")
            print(f"  - 训练轮数: {cls.get_value(config, 'stage2', 'epochs')}")
            print(f"  - 学习率: {cls.get_value(config, 'stage2', 'learning_rate')}")
            
            print(f"\n优化器:")
            print(f"  - 类型: {cls.get_value(config, 'optimizer', 'type')}")
            print(f"  - 学习率衰减: {cls.get_value(config, 'optimizer', 'use_lr_decay')}")
            print(f"  - 梯度裁剪: {cls.get_value(config, 'optimizer', 'use_gradient_clipping')}")
            
            print(f"\n损失函数:")
            print(f"  - 自适应权重: {cls.get_value(config, 'loss_function', 'use_adaptive_weights')}")
            print(f"  - L1权重: {cls.get_value(config, 'loss_function', 'weights', 'l1_weight')}")
            print(f"  - SSIM权重: {cls.get_value(config, 'loss_function', 'weights', 'ssim_weight')}")
            
        elif config_type == 'fusion':
            print(f"\nI/O路径:")
            single_mode = cls.get_value(config, 'io_paths', 'single_mode', default={})
            batch_mode = cls.get_value(config, 'io_paths', 'batch_mode', default={})
            print(f"  - 单对模式输出: {cls.get_value(single_mode, 'output')}")
            print(f"  - 批量模式红外目录: {cls.get_value(batch_mode, 'ir_dir')}")
            print(f"  - 批量模式可见光目录: {cls.get_value(batch_mode, 'vi_dir')}")
            
            print(f"\n模型:")
            print(f"  - 模型路径: {cls.get_value(config, 'model', 'model_path')}")
            print(f"  - 模型名称: {cls.get_value(config, 'model', 'model_name')}")
            print(f"  - 融合策略: {cls.get_value(config, 'model', 'fusion_strategy')}")
            
            print(f"\n性能:")
            print(f"  - 批量大小: {cls.get_value(config, 'performance', 'batch_size')}")
            print(f"  - 显示统计: {cls.get_value(config, 'performance', 'show_stats')}")
            print(f"  - CUDA优化: {cls.get_value(config, 'performance', 'enable_cuda_optimize')}")
        
        print("=" * 60)


class TrainingConfig:
    """
    训练配置类
    
    提供更方便的配置访问接口，支持属性访问方式。
    
    使用示例：
    ---------
    ```python
    config = TrainingConfig()
    
    # 属性访问
    batch_size = config.training.batch_size
    stage1_epochs = config.stage1.epochs
    stage1_lr = config.stage1.learning_rate
    
    # 获取配置字典
    config_dict = config.to_dict()
    ```
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化训练配置
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self._config = ConfigLoader.load_train_config(config_path)
        self._setup_attributes()
    
    def _setup_attributes(self):
        """设置配置属性"""
        self.dataset = self._config.get('dataset', {})
        self.training = self._config.get('training', {})
        self.stage1 = self._config.get('stage1', {})
        self.stage2 = self._config.get('stage2', {})
        self.optimizer = self._config.get('optimizer', {})
        self.loss_function = self._config.get('loss_function', {})
        self.model = self._config.get('model', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            *keys: 嵌套键路径
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        return ConfigLoader.get_value(self._config, *keys, default=default)
    
    def validate(self) -> bool:
        """验证配置完整性"""
        return ConfigLoader.validate_train_config(self._config)
    
    def print_info(self):
        """打印配置信息"""
        ConfigLoader.print_config_info(self._config, 'train')


class FusionConfig:
    """
    融合配置类
    
    提供更方便的配置访问接口，支持属性访问方式。
    
    使用示例：
    ---------
    ```python
    config = FusionConfig()
    
    # 属性访问
    batch_size = config.performance.batch_size
    fusion_strategy = config.model.fusion_strategy
    
    # 获取配置字典
    config_dict = config.to_dict()
    ```
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化融合配置
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self._config = ConfigLoader.load_fusion_config(config_path)
        self._setup_attributes()
    
    def _setup_attributes(self):
        """设置配置属性"""
        self.io_paths = self._config.get('io_paths', {})
        self.model = self._config.get('model', {})
        self.device = self._config.get('device', {})
        self.performance = self._config.get('performance', {})
        self.fusion_strategies = self._config.get('fusion_strategies', {})
        self.image_preprocessing = self._config.get('image_preprocessing', {})
        self.image_postprocessing = self._config.get('image_postprocessing', {})
        self.optimization = self._config.get('optimization', {})
        self.performance_monitoring = self._config.get('performance_monitoring', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            *keys: 嵌套键路径
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        return ConfigLoader.get_value(self._config, *keys, default=default)
    
    def validate(self) -> bool:
        """验证配置完整性"""
        return ConfigLoader.validate_fusion_config(self._config)
    
    def print_info(self):
        """打印配置信息"""
        ConfigLoader.print_config_info(self._config, 'fusion')
