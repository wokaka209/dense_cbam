# -*- coding: utf-8 -*-
"""
@file name:util_fusion.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2024/02/22
"""

import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from models import fuse_model


class FusionConfig:
    gray: bool = True
    model_name: str = 'DenseFuse'
    model_weights: str = "../runs/train_COCO/checkpoints/epoch003-loss0.000.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_strategy: str = "l1norm"  # 可选: "mean", "max", "l1norm"


'''
/********************************/
    模型推理
/********************************/
'''


class ImageFusion:
    # -----------------------------------#
    #   初始化
    # -----------------------------------#
    def __init__(self, config):
        self.config = config
        self.load_model()

    def load_model(self):
        # -----------------#
        #   创建模型
        # -----------------#
        in_channel = 1 if self.config.gray else 3
        out_channel = 1 if self.config.gray else 3
        self.model = fuse_model(self.config.model_name,
                                input_nc=in_channel,
                                output_nc=out_channel)
        # -----------------#
        #   device
        # -----------------#
        self.model = self.model.to(self.config.device)
        # -----------------#
        #   载入模型权重
        # -----------------#
        checkpoint = torch.load(self.config.model_weights,
                                map_location=self.config.device, weights_only=False)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print('{} model loaded.'.format(self.config.model_weights))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path,
                           mode=ImageReadMode.GRAY if self.config.gray else ImageReadMode.RGB)

        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ToTensor(),
                                               ])
        image = image_transforms(image).unsqueeze(0)
        return image

    def fusion_strategy(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        if self.config.fusion_strategy == "mean":
            return (feature1 + feature2) / 2
        elif self.config.fusion_strategy == "max":
            return torch.maximum(feature1, feature2)
        elif self.config.fusion_strategy == "l1norm":
            # L1范数融合策略：计算每个像素位置的L1范数，并选择具有较高L1范数的特征
            # 计算每个像素位置的L1范数
            l1_norm1 = torch.abs(feature1)
            l1_norm2 = torch.abs(feature2)
            
            # 创建掩码，选择具有更高L1范数的位置
            mask = (l1_norm1 > l1_norm2).float()
            
            # 根据掩码融合特征
            fused_features = mask * feature1 + (1 - mask) * feature2
            return fused_features
        elif self.config.fusion_strategy == "adaptive_l1":
            # 自适应L1范数融合策略：使用动态权重分配
            # 计算L1范数
            l1_norm1 = torch.abs(feature1)
            l1_norm2 = torch.abs(feature2)
            
            # 计算总能量
            total_energy = l1_norm1 + l1_norm2 + 1e-8  # 添加小常数避免除零
            
            # 基于相对强度计算权重
            weight1 = l1_norm1 / total_energy
            weight2 = l1_norm2 / total_energy
            
            # 应用加权融合
            fused_features = weight1 * feature1 + weight2 * feature2
            return fused_features
        elif self.config.fusion_strategy == "gradient_based":
            # 基于梯度的融合策略：优先保留边缘和细节信息
            # 计算梯度信息
            grad1_x = torch.abs(feature1[:, :, :, 1:] - feature1[:, :, :, :-1])
            grad1_y = torch.abs(feature1[:, :, 1:, :] - feature1[:, :, :-1, :])
            grad2_x = torch.abs(feature2[:, :, :, 1:] - feature2[:, :, :, :-1])
            grad2_y = torch.abs(feature2[:, :, 1:, :] - feature2[:, :, :-1, :])
            
            # 计算梯度幅值
            grad_mag1 = torch.sqrt(grad1_x[:, :, :, :-1]**2 + grad1_y[:, :, :-1, :]**2)
            grad_mag2 = torch.sqrt(grad2_x[:, :, :, :-1]**2 + grad2_y[:, :, :-1, :]**2)
            
            # 扩展梯度信息以匹配原始尺寸
            # 对x方向梯度，在最后一列填充
            pad_x = torch.zeros_like(grad_mag1[:, :, :, -1:]).expand(-1, -1, -1, 1)
            grad_mag1 = torch.cat([grad_mag1, pad_x], dim=3)
            grad_mag2 = torch.cat([grad_mag2, pad_x.clone()], dim=3)
            
            # 对y方向，同样处理
            pad_y = torch.zeros_like(grad_mag1[:, :, -1:, :]).expand(-1, -1, 1, -1)
            grad_mag1 = torch.cat([grad_mag1, pad_y], dim=2)
            grad_mag2 = torch.cat([grad_mag2, pad_y.clone()], dim=2)
            
            # 创建掩码，选择梯度更大的区域
            mask = (grad_mag1 > grad_mag2).float()
            
            # 融合特征
            fused_features = mask * feature1 + (1 - mask) * feature2
            return fused_features
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.config.fusion_strategy}")

    def run(self, image1_path, image2_path):
        self.model.eval()
        with torch.no_grad():
            # 图像预处理
            image1 = self.preprocess_image(image1_path).to(self.config.device)
            image2 = self.preprocess_image(image2_path).to(self.config.device)

            # Encoder
            image1_features = self.model.encoder(image1)
            image2_features = self.model.encoder(image2)

            # 特征融合
            fused_features = self.fusion_strategy(image1_features, image2_features)

            # Decoder
            fused_image = self.model.decoder(fused_features)

            # 张量后处理
            fused_image = fused_image.cpu().squeeze(0)
        return fused_image


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    import os
    from torchvision.utils import save_image

    # 配置参数
    config = FusionConfig()
    fusion_model = ImageFusion(config)

    # 设置输入输出路径
    test_pairs = [
        {
            "inf": "../data_test/Road/INF_images/1.jpg",
            "vis": "../data_test/Road/VIS_images/1.jpg",
            "output": "../data_result/pair/fused_1.png"
        },
        # {
        #     "inf": "../data_test/Road/INF_images/2.jpg",
        #     "vis": "../data_test/Road/VIS_images/2.jpg",
        #     "output": "../data_result/pair/fused_2.png"
        # },
    ]

    # 创建输出目录
    os.makedirs("../data_result/pair", exist_ok=True)

    # 执行融合
    for pair in test_pairs:
        image1_path = pair["inf"]
        image2_path = pair["vis"]
        if not os.path.exists(image1_path):
            raise FileNotFoundError(f"图像文件不存在: {image1_path}")
        if not os.path.exists(image2_path):
            raise FileNotFoundError(f"图像文件不存在: {image2_path}")
        # 融合图像
        fused_image = fusion_model.run(pair["inf"], pair["vis"])
        # 保存结果
        save_image(fused_image, pair["output"])
        print(f"Fusion completed: {pair['output']}")