# -*- coding: utf-8 -*-
"""
@file name:DenseFuse_with_fusion.py
@desc: 带融合层的DenseFuse模型，用于三阶段训练中的阶段三（端到端融合）
@Writer: wokaka209
@Date: 2026-03-30

功能说明：
---------
本模块提供带融合层的DenseFuse模型，用于红外和可见光图像的端到端融合训练。
模型包含两个共享权重的编码器、一个融合层和一个解码器。

使用方法：
---------
    from models import fuse_model_with_fusion_layer
    
    # 创建融合模型
    model = fuse_model_with_fusion_layer(
        model_name="DenseFuse",
        input_nc=1,
        output_nc=1,
        use_attention=True,
        fusion_strategy='l1_norm'
    )
"""

import torch
from torch import nn
from .DenseFuse import Dense_Encoder, CNN_Decoder


class DenseFuseWithFusion(nn.Module):
    """
    带融合层的DenseFuse模型
    
    用于红外和可见光图像的端到端融合训练。
    
    Attributes:
        encoder (Dense_Encoder): 编码器（共享权重）
        fusion_layer (nn.Module): 融合层
        decoder (CNN_Decoder): 解码器
    """
    
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1, 
                 use_attention=True, fusion_strategy='l1_norm'):
        """
        初始化带融合层的DenseFuse模型
        
        Args:
            input_nc (int): 输入通道数
            output_nc (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            use_attention (bool): 是否使用注意力机制
            fusion_strategy (str): 融合策略，可选 'addition', 'l1_norm', 'weighted_average'
        """
        super().__init__()
        
        # 编码器（共享权重）
        self.encoder = Dense_Encoder(
            input_nc=input_nc,
            kernel_size=kernel_size,
            stride=stride,
            use_attention=use_attention
        )
        
        # 融合层
        from .fusion_layer import get_fusion_layer
        self.fusion_layer = get_fusion_layer(strategy=fusion_strategy)
        
        # 解码器
        self.decoder = CNN_Decoder(
            output_nc=output_nc,
            kernel_size=kernel_size,
            stride=stride
        )
        
        self.use_attention = use_attention
        self.fusion_strategy = fusion_strategy
    
    def forward(self, ir_image, vi_image):
        """
        前向传播
        
        Args:
            ir_image (torch.Tensor): 红外图像
            vi_image (torch.Tensor): 可见光图像
        
        Returns:
            torch.Tensor: 融合后的图像
        """
        # 提取红外图像特征
        ir_features = self.encoder(ir_image)
        
        # 提取可见光图像特征
        vi_features = self.encoder(vi_image)
        
        # 融合特征
        fused_features = self.fusion_layer(ir_features, vi_features)
        
        # 解码生成融合图像
        fused_image = self.decoder(fused_features)
        
        return fused_image
    
    def freeze_backbone(self):
        """冻结编码器和解码器的主干网络参数（不包括CBAM）"""
        for param in self.encoder.conv.parameters():
            param.requires_grad = False
        for param in self.encoder.DenseBlock.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻编码器和解码器的主干网络参数"""
        for param in self.encoder.conv.parameters():
            param.requires_grad = True
        for param in self.encoder.DenseBlock.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def freeze_cbam(self):
        """冻结CBAM模块参数"""
        if self.encoder.use_attention:
            for param in self.encoder.attention.parameters():
                param.requires_grad = False
    
    def unfreeze_cbam(self):
        """解冻CBAM模块参数"""
        if self.encoder.use_attention:
            for param in self.encoder.attention.parameters():
                param.requires_grad = True
    
    def freeze_decoder(self):
        """冻结解码器参数"""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        """解冻解码器参数"""
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def freeze_fusion_layer(self):
        """冻结融合层参数"""
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
    
    def unfreeze_fusion_layer(self):
        """解冻融合层参数"""
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_backbone_params(self):
        """获取主干网络参数（不包括CBAM）"""
        params = []
        params.extend(self.encoder.conv.parameters())
        params.extend(self.encoder.DenseBlock.parameters())
        params.extend(self.decoder.parameters())
        return params
    
    def get_cbam_params(self):
        """获取CBAM模块参数"""
        if self.encoder.use_attention:
            return list(self.encoder.attention.parameters())
        return []
    
    def get_fusion_params(self):
        """获取融合层参数"""
        return list(self.fusion_layer.parameters())
    
    def get_encoder_params(self):
        """获取编码器参数"""
        return list(self.encoder.parameters())
    
    def get_decoder_params(self):
        """获取解码器参数"""
        return list(self.decoder.parameters())


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    channels = 1
    height = 256
    width = 256
    
    # 创建测试图像
    ir_image = torch.randn(batch_size, channels, height, width)
    vi_image = torch.randn(batch_size, channels, height, width)
    
    print(f"红外图像形状: {ir_image.shape}")
    print(f"可见光图像形状: {vi_image.shape}")
    
    # 测试L1-norm融合
    print("\n测试L1-norm融合...")
    model_l1 = DenseFuseWithFusion(
        input_nc=channels,
        output_nc=channels,
        use_attention=True,
        fusion_strategy='l1_norm'
    )
    fused_l1 = model_l1(ir_image, vi_image)
    print(f"融合图像形状: {fused_l1.shape}")
    print(f"融合策略: {model_l1.fusion_strategy}")
    
    # 测试加法融合
    print("\n测试加法融合...")
    model_add = DenseFuseWithFusion(
        input_nc=channels,
        output_nc=channels,
        use_attention=True,
        fusion_strategy='addition'
    )
    fused_add = model_add(ir_image, vi_image)
    print(f"融合图像形状: {fused_add.shape}")
    print(f"融合策略: {model_add.fusion_strategy}")
    
    # 测试参数冻结
    print("\n测试参数冻结...")
    model_l1.freeze_backbone()
    trainable_params = model_l1.get_trainable_params()
    print(f"冻结主干后可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    model_l1.unfreeze_backbone()
    trainable_params = model_l1.get_trainable_params()
    print(f"解冻主干后可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    # 测试反向传播
    print("\n测试反向传播...")
    loss = fused_l1.sum()
    loss.backward()
    print(f"反向传播成功，梯度计算正常")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model_l1.parameters())
    trainable_params = sum(p.numel() for p in model_l1.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
