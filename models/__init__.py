'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-13 12:04:33
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-13 13:22:40
FilePath: \my_densefuse_advantive\models\__init__.py
Description: 模型导入函数 - 支持三种融合方案
'''
from .DenseFuse import DenseFuse_train


def fuse_model(model_name, input_nc, output_nc, fusion_strategy=1):
    """
    创建融合模型
    
    Args:
        model_name: 模型名称（"DenseFuse"）
        input_nc: 输入通道数（1=灰度，3=RGB）
        output_nc: 输出通道数
        fusion_strategy: 融合方案选择（1/2/3）
            - 1: DenseBlock内部实时引导融合（推荐IVIF任务）
            - 2: Decoder中解码特征选择（高质量融合需求）
            - 3: 多层次组合全方位增强（最佳融合质量）
    
    Returns:
        model_ft: 融合模型实例
    
    Example:
        >>> # 方案1：推荐用于IVIF任务
        >>> model = fuse_model("DenseFuse", input_nc=1, output_nc=1, fusion_strategy=1)
        
        >>> # 方案2：高质量融合需求
        >>> model = fuse_model("DenseFuse", input_nc=1, output_nc=1, fusion_strategy=2)
        
        >>> # 方案3：最佳融合质量
        >>> model = fuse_model("DenseFuse", input_nc=1, output_nc=1, fusion_strategy=3)
    """
    # 选择合适的模型
    model_ft = None

    if model_name == "DenseFuse":
        """ DenseFuse - 支持三种融合方案
        """
        model_ft = DenseFuse_train(
            input_nc=input_nc, 
            output_nc=output_nc, 
            fusion_strategy=fusion_strategy
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
