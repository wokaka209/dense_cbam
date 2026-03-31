'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-13 12:04:33
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-13 13:22:40
FilePath: \my_densefuse_advantive\models\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1/wiki/%E9%85%8D%E7%BD%AE
'''
from .DenseFuse import DenseFuse_train
from .fusion_layer import FusionLayer, get_fusion_layer


def fuse_model(model_name, input_nc, output_nc, use_attention=True):
    # 选择合适的模型
    model_ft = None

    if model_name == "DenseFuse":
        """ DenseFuse
        """
        model_ft = DenseFuse_train(input_nc=input_nc, output_nc=output_nc, use_attention=use_attention)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def fuse_model_with_fusion_layer(model_name, input_nc, output_nc, use_attention=True, fusion_strategy='l1_norm'):
    """
    创建带融合层的融合模型（用于阶段三：端到端融合）
    
    Args:
        model_name (str): 模型名称
        input_nc (int): 输入通道数
        output_nc (int): 输出通道数
        use_attention (bool): 是否使用注意力机制
        fusion_strategy (str): 融合策略
    
    Returns:
        DenseFuseWithFusion: 融合模型实例
    """
    from .DenseFuse_with_fusion import DenseFuseWithFusion
    
    if model_name == "DenseFuse":
        model_ft = DenseFuseWithFusion(
            input_nc=input_nc,
            output_nc=output_nc,
            use_attention=use_attention,
            fusion_strategy=fusion_strategy
        )
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
