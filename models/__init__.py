'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-13 12:04:33
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-13 13:22:40
FilePath: \my_densefuse_advantive\models\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from .DenseFuse import DenseFuse_train


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
