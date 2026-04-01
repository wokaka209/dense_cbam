# -*- coding: utf-8 -*-
"""
@file name:DenseFuse.py
@desc: DenseFuse网络模型优化 - 支持三种特征融合方案
@Writer: Cat2eacher, wokaka209
@Date: 2024/02/21
@Update: 2026/04/01 - 添加三种融合方案支持
"""
import torch
from torch import nn
try:
    from torchsummary import summary
except ImportError:
    summary = None


# -------------------------#
#   基本卷积模块
# -------------------------#
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out


# -------------------------#
#   密集卷积
# -------------------------#
class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([x, out], 1)
        return out


# -------------------------#
#   Dense Block unit (支持方案1：内部注意力)
# -------------------------#
class DenseBlock(torch.nn.Module):
    """
    DenseBlock模块
    
    Args:
        in_channels: 输入通道数
        kernel_size: 卷积核大小
        stride: 步长
        use_attention: 是否使用注意力机制（方案1和方案3启用）
    
    融合方案说明：
        - 方案1：use_attention=True，在DenseBlock内部添加CBAM，实现实时引导融合
        - 方案2：use_attention=False，不在DenseBlock内部添加注意力
        - 方案3：use_attention=True，作为多层次注意力的一部分
    """
    def __init__(self, in_channels, kernel_size, stride, use_attention=False):
        super().__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)
        
        # 方案1和方案3：在DenseBlock内部添加注意力机制
        self.use_attention = use_attention
        if use_attention:
            try:
                from .attention_modules import ColorAwareCBAM
            except ImportError:
                from attention_modules import ColorAwareCBAM
            # DenseBlock输出64通道，使用ColorAwareCBAM保护颜色特征
            self.attention = ColorAwareCBAM(in_channels=64, reduction=None, kernel_size=7, color_preservation_weight=0.3)

    def forward(self, x):
        out = self.denseblock(x)
        if self.use_attention:
            out = self.attention(out)
        return out


'''
/****************************************************/
    DenseFuse Network - 支持三种融合方案
/****************************************************/
'''


# ===================== Dense_Encoder =====================
class Dense_Encoder(nn.Module):
    """
    DenseFuse编码器
    
    Args:
        input_nc: 输入通道数
        kernel_size: 卷积核大小
        stride: 步长
        fusion_strategy: 融合方案选择（1/2/3）
            - 1: DenseBlock内部实时引导融合（推荐IVIF任务）
            - 2: Decoder中解码特征选择（高质量融合需求）
            - 3: 多层次组合全方位增强（最佳融合质量）
    """
    def __init__(self, input_nc=1, kernel_size=3, stride=1, fusion_strategy=1):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        
        # 根据融合方案决定DenseBlock是否使用注意力
        # 方案1和方案3：DenseBlock内部使用注意力
        use_attention_in_denseblock = (fusion_strategy in [1, 3])
        
        self.conv = ConvLayer(input_nc, 16, kernel_size, stride)
        self.DenseBlock = DenseBlock(16, kernel_size, stride, use_attention=use_attention_in_denseblock)
        
        # 方案3：在Encoder末尾也添加注意力（多层次组合）
        if fusion_strategy == 3:
            try:
                from .attention_modules import ColorAwareCBAM
            except ImportError:
                from attention_modules import ColorAwareCBAM
            self.attention = ColorAwareCBAM(in_channels=64, reduction=None, kernel_size=7, color_preservation_weight=0.3)
        else:
            self.attention = None

    def forward(self, x):
        output = self.conv(x)
        output = self.DenseBlock(output)
        
        # 方案3：Encoder末尾额外注意力增强
        if self.fusion_strategy == 3 and self.attention is not None:
            output = self.attention(output)
        
        return output


# ====================== CNN_Decoder ======================
class CNN_Decoder(nn.Module):
    """
    DenseFuse解码器
    
    Args:
        output_nc: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        fusion_strategy: 融合方案选择（1/2/3）
            - 1: DenseBlock内部实时引导融合（不在Decoder添加注意力）
            - 2: Decoder中解码特征选择（在Decoder添加注意力）
            - 3: 多层次组合全方位增强（在Decoder添加注意力）
    """
    def __init__(self, output_nc=1, kernel_size=3, stride=1, fusion_strategy=1):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        
        # 解码器卷积层
        self.conv1 = ConvLayer(64, 64, kernel_size, stride)
        self.conv2 = ConvLayer(64, 32, kernel_size, stride)
        self.conv3 = ConvLayer(32, 16, kernel_size, stride)
        self.conv4 = ConvLayer(16, output_nc, kernel_size, stride, is_last=True)
        
        # 方案2和方案3：在解码过程中添加注意力机制
        if fusion_strategy in [2, 3]:
            try:
                from .attention_modules import ColorAwareCBAM
            except ImportError:
                from attention_modules import ColorAwareCBAM
            self.attention1 = ColorAwareCBAM(in_channels=64, reduction=None, kernel_size=7, color_preservation_weight=0.3)
            self.attention2 = ColorAwareCBAM(in_channels=32, reduction=None, kernel_size=7, color_preservation_weight=0.3)
        else:
            self.attention1 = None
            self.attention2 = None

    def forward(self, encoder_output):
        # 第一层解码 + 注意力（方案2和方案3）
        x = self.conv1(encoder_output)
        if self.fusion_strategy in [2, 3] and self.attention1 is not None:
            x = self.attention1(x)
        
        # 第二层解码 + 注意力（方案2和方案3）
        x = self.conv2(x)
        if self.fusion_strategy in [2, 3] and self.attention2 is not None:
            x = self.attention2(x)
        
        # 后续解码层
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# ====================== AutoEncoder ======================
class DenseFuse_train(nn.Module):
    """
    DenseFuse训练模型 - 支持三种融合方案
    
    Args:
        input_nc: 输入通道数（1=灰度，3=RGB）
        output_nc: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        fusion_strategy: 融合方案选择（1/2/3）
            - 方案1：DenseBlock内部实时引导融合
              特点：在DenseBlock内部的特征融合过程中实现实时引导机制
              优势：计算量略有增加，推荐用于IVIF（红外与可见光图像融合）任务
              适用场景：红外与可见光图像融合，需要实时引导特征对齐
            
            - 方案2：Decoder中解码特征选择
              特点：在Decoder模块的解码过程中实现特征选择功能
              优势：会增加模型参数量，适用于高质量融合需求场景
              适用场景：追求高质量融合结果，对细节保留要求高
            
            - 方案3：多层次组合全方位增强
              特点：实现多层次组合的全方位特征增强机制
              优势：计算开销最大，但能获得最佳融合质量
              适用场景：追求最高融合质量，计算资源充足
    
    Example:
        >>> # 方案1：推荐用于IVIF任务
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, fusion_strategy=1)
        
        >>> # 方案2：高质量融合需求
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, fusion_strategy=2)
        
        >>> # 方案3：最佳融合质量
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, fusion_strategy=3)
    """
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1, fusion_strategy=1):
        super().__init__()
        
        # 验证融合方案参数
        if fusion_strategy not in [1, 2, 3]:
            raise ValueError(f"fusion_strategy must be 1, 2, or 3, got {fusion_strategy}")
        
        self.fusion_strategy = fusion_strategy
        
        # Encoder和Decoder使用相同的融合方案
        self.encoder = Dense_Encoder(
            input_nc=input_nc, 
            kernel_size=kernel_size, 
            stride=stride, 
            fusion_strategy=fusion_strategy
        )
        self.decoder = CNN_Decoder(
            output_nc=output_nc, 
            kernel_size=kernel_size, 
            stride=stride, 
            fusion_strategy=fusion_strategy
        )
        
        # 打印融合方案信息
        self._print_strategy_info()

    def _print_strategy_info(self):
        """打印当前融合方案信息"""
        strategy_info = {
            1: "方案1：DenseBlock内部实时引导融合（推荐IVIF任务）",
            2: "方案2：Decoder中解码特征选择（高质量融合需求）",
            3: "方案3：多层次组合全方位增强（最佳融合质量）"
        }
        print(f"使用融合方案：{strategy_info[self.fusion_strategy]}")

    def forward(self, x):
        encoder_feature = self.encoder(x)
        out = self.decoder(encoder_feature)
        return out


def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    print("="*60)
    print("DenseFuse模型测试 - 三种融合方案对比")
    print("="*60)
    
    # 测试三种融合方案
    for strategy in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"测试融合方案 {strategy}")
        print(f"{'='*60}")
        
        model = DenseFuse_train(input_nc=3, output_nc=3, fusion_strategy=strategy)
        param_count = sum(x.numel() for x in model.parameters())
        print(f"模型参数量: {param_count:,}")
        
        # 测试前向传播
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        print(f"输入shape: {input_tensor.shape}")
        print(f"输出shape: {output.shape}")
        
        # 获取中间特征
        features = model.encoder(input_tensor)
        print(f"编码器特征shape: {features.shape}")
    
    print("\n" + "="*60)
    print("所有融合方案测试完成！")
    print("="*60)
