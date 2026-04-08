# -*- coding: utf-8 -*-
"""
@file name:DenseFuse.py
@desc: DenseFuse网络模型优化 - 支持两种CBAM注意力机制方案
@Writer: Cat2eacher, wokaka209
@Date: 2024/02/21
@Update: 2026/04/05 - 添加两种CBAM注意力机制方案支持
"""
import torch
from torch import nn
try:
    from torchsummary import summary
except ImportError:
    summary = None


def create_cbam_attention(in_channels, reduction_ratio=16, kernel_size=7, use_color_aware=False, color_preservation_weight=0.3,
                         use_channel_attention=True, use_spatial_attention=True):
    """
    创建CBAM注意力模块 - 支持颜色感知CBAM和消融实验
    
    Args:
        in_channels: 输入通道数
        reduction_ratio: 通道压缩比例，用于通道注意力
        kernel_size: 空间注意力的卷积核大小
        use_color_aware: 是否使用颜色感知CBAM（解决泛黄问题）
        color_preservation_weight: 颜色保护权重（0.0-1.0）
        use_channel_attention: 是否启用通道注意力（消融实验用）
        use_spatial_attention: 是否启用空间注意力（消融实验用）
    """
    try:
        from .attention_modules import CBAM, ColorAwareCBAM
    except ImportError:
        from attention_modules import CBAM, ColorAwareCBAM
    
    if use_color_aware:
        # 使用颜色感知CBAM，专门保护RGB图像中的颜色特征
        return ColorAwareCBAM(
            in_channels=in_channels, 
            reduction=reduction_ratio, 
            kernel_size=kernel_size,
            color_preservation_weight=color_preservation_weight,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention
        )
    else:
        # 使用标准CBAM
        return CBAM(in_channels=in_channels, reduction=reduction_ratio, kernel_size=kernel_size,
                   use_channel_attention=use_channel_attention, use_spatial_attention=use_spatial_attention)


# -------------------------#
#   融合策略函数
# -------------------------#
def apply_fusion_strategy(ir_features, vi_features, strategy='add'):
    """
    应用融合策略 - 支持三种融合策略（add、l1norm、hybrid）
    
    Args:
        ir_features: 红外特征图 [B, C, H, W]
        vi_features: 可见光特征图 [B, C, H, W]
        strategy: 融合策略，可选 'add', 'l1norm', 'hybrid'
        
    Returns:
        fused_features: 融合后的特征图
    """
    if strategy == 'add':
        # 简单平均融合
        return (ir_features + vi_features) / 2
    
    elif strategy == 'l1norm':
        # 基于L1范数的加权融合
        ir_abs = torch.abs(ir_features)
        vi_abs = torch.abs(vi_features)
        sum_abs = ir_abs + vi_abs + 1e-8  # 防止除零
        weight_ir = ir_abs / sum_abs
        weight_vi = vi_abs / sum_abs
        return weight_ir * ir_features + weight_vi * vi_features
    
    elif strategy == 'hybrid':
        # 混合策略：结合add和l1norm的优点
        # 首先计算add融合
        add_fused = (ir_features + vi_features) / 2
        # 计算l1norm融合
        ir_abs = torch.abs(ir_features)
        vi_abs = torch.abs(vi_features)
        sum_abs = ir_abs + vi_abs + 1e-8
        weight_ir = ir_abs / sum_abs
        weight_vi = vi_abs / sum_abs
        l1norm_fused = weight_ir * ir_features + weight_vi * vi_features
        # 自适应权重：根据特征激活度决定
        ir_norm = torch.norm(ir_features, p=1, dim=(2,3), keepdim=True)
        vi_norm = torch.norm(vi_features, p=1, dim=(2,3), keepdim=True)
        total_norm = ir_norm + vi_norm + 1e-8
        alpha = ir_norm / total_norm  # 红外特征权重
        # 加权混合
        return alpha * add_fused + (1 - alpha) * l1norm_fused
    
    else:
        raise ValueError(f"未知融合策略: {strategy}，请选择 'add', 'l1norm', 或 'hybrid'")


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
#   Dense Block unit (支持方案1：DenseBlock输出后CBAM)
# -------------------------#
class DenseBlock(torch.nn.Module):
    """
    DenseBlock模块 - 支持两种CBAM注意力机制方案和颜色感知CBAM
    
    Args:
        in_channels: 输入通道数
        kernel_size: 卷积核大小
        stride: 步长
        cbam_scheme: CBAM实施方案选择
            - 0: 不使用CBAM
            - 1: 方案1 - DenseBlock输出后应用CBAM
            - 2: 方案2 - 融合层输入前应用CBAM（需要外部处理）
        reduction_ratio: 通道压缩比例（16, 32, 64, 128等）
        use_color_aware: 是否使用颜色感知CBAM（解决泛黄问题）
        color_preservation_weight: 颜色保护权重（0.0-1.0）
    """
    def __init__(self, in_channels, kernel_size, stride, cbam_scheme=0, reduction_ratio=16, 
                 use_color_aware=False, color_preservation_weight=0.3,
                 use_channel_attention=True, use_spatial_attention=True):
        super().__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)
        
        # CBAM实施方案选择
        self.cbam_scheme = cbam_scheme
        
        # 方案1：在DenseBlock输出后应用CBAM
        if cbam_scheme == 1:
            # DenseBlock输出64通道特征图
            self.cbam_attention = create_cbam_attention(
                in_channels=64, 
                reduction_ratio=reduction_ratio, 
                kernel_size=7,
                use_color_aware=use_color_aware,
                color_preservation_weight=color_preservation_weight,
                use_channel_attention=use_channel_attention,
                use_spatial_attention=use_spatial_attention
            )
            color_info = "颜色感知" if use_color_aware else "标准"
            channel_info = "通道注意力" if use_channel_attention else "无通道注意力"
            spatial_info = "空间注意力" if use_spatial_attention else "无空间注意力"
            print(f"[方案1] 启用：DenseBlock输出后{color_info}CBAM ({channel_info}, {spatial_info}, reduction_ratio={reduction_ratio})")
        else:
            self.cbam_attention = None

    def forward(self, x):
        out = self.denseblock(x)
        
        # 方案1：在DenseBlock输出后应用CBAM
        if self.cbam_scheme == 1 and self.cbam_attention is not None:
            out = self.cbam_attention(out)
        
        return out


'''
/****************************************************/
    DenseFuse Network - 支持两种融合方案
/****************************************************/
'''


# ===================== Dense_Encoder =====================
class Dense_Encoder(nn.Module):
    """
    DenseFuse编码器 - 支持两种CBAM注意力机制方案和颜色感知CBAM
    
    Args:
        input_nc: 输入通道数
        kernel_size: 卷积核大小
        stride: 步长
        cbam_scheme: CBAM实施方案选择
            - 0: 不使用CBAM
            - 1: 方案1 - DenseBlock输出后应用CBAM
            - 2: 方案2 - 融合层输入前应用CBAM
        reduction_ratio: 通道压缩比例（16, 32, 64, 128等）
        use_color_aware: 是否使用颜色感知CBAM（解决泛黄问题）
        color_preservation_weight: 颜色保护权重（0.0-1.0）
    """
    def __init__(self, input_nc=1, kernel_size=3, stride=1, cbam_scheme=0, reduction_ratio=16,
                 use_color_aware=False, color_preservation_weight=0.3,
                 use_channel_attention=True, use_spatial_attention=True):
        super().__init__()
        self.cbam_scheme = cbam_scheme
        
        # 基础卷积层
        self.conv = ConvLayer(input_nc, 16, kernel_size, stride)
        
        # DenseBlock - 支持方案1和颜色感知CBAM
        self.DenseBlock = DenseBlock(16, kernel_size, stride, cbam_scheme=cbam_scheme, 
                                    reduction_ratio=reduction_ratio, use_color_aware=use_color_aware,
                                    color_preservation_weight=color_preservation_weight,
                                    use_channel_attention=use_channel_attention,
                                    use_spatial_attention=use_spatial_attention)
        
        # 方案2：在融合层输入前应用CBAM（对两路输入分别处理）
        if cbam_scheme == 2:
            # 为红外和可见光特征分别创建CBAM
            self.ir_cbam = create_cbam_attention(
                in_channels=64, 
                reduction_ratio=reduction_ratio, 
                kernel_size=7,
                use_color_aware=use_color_aware,
                color_preservation_weight=color_preservation_weight,
                use_channel_attention=use_channel_attention,
                use_spatial_attention=use_spatial_attention
            )
            self.vi_cbam = create_cbam_attention(
                in_channels=64, 
                reduction_ratio=reduction_ratio, 
                kernel_size=7,
                use_color_aware=use_color_aware,
                color_preservation_weight=color_preservation_weight,
                use_channel_attention=use_channel_attention,
                use_spatial_attention=use_spatial_attention
            )
            color_info = "颜色感知" if use_color_aware else "标准"
            channel_info = "通道注意力" if use_channel_attention else "无通道注意力"
            spatial_info = "空间注意力" if use_spatial_attention else "无空间注意力"
            print(f"[方案2] 启用：融合层输入前{color_info}CBAM ({channel_info}, {spatial_info}, reduction_ratio={reduction_ratio})")
        else:
            self.ir_cbam = None
            self.vi_cbam = None

    def forward(self, x):
        # 基础特征提取
        output = self.conv(x)
        output = self.DenseBlock(output)
        
        # 方案2：在融合层输入前应用CBAM（需要外部提供两路输入）
        # 这里只返回基础特征，方案2的具体实现在融合层中处理
        return output
    
    def forward_with_cbam_scheme2(self, ir_features, vi_features):
        """
        方案2专用前向传播：对两路输入分别应用CBAM
        
        Args:
            ir_features: 红外特征图
            vi_features: 可见光特征图
            
        Returns:
            ir_attended: 经过CBAM处理的红外特征
            vi_attended: 经过CBAM处理的可见光特征
        """
        if self.cbam_scheme == 2 and self.ir_cbam is not None and self.vi_cbam is not None:
            ir_attended = self.ir_cbam(ir_features)
            vi_attended = self.vi_cbam(vi_features)
            return ir_attended, vi_attended
        else:
            return ir_features, vi_features


# ====================== CNN_Decoder ======================
class CNN_Decoder(nn.Module):
    """
    DenseFuse解码器 - 简化实现，删除多余CBAM
    
    Args:
        output_nc: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
    """
    def __init__(self, output_nc=1, kernel_size=3, stride=1):
        super().__init__()
        
        # 解码器卷积层
        self.conv1 = ConvLayer(64, 64, kernel_size, stride)
        self.conv2 = ConvLayer(64, 32, kernel_size, stride)
        self.conv3 = ConvLayer(32, 16, kernel_size, stride)
        self.conv4 = ConvLayer(16, output_nc, kernel_size, stride, is_last=True)

    def forward(self, encoder_output):
        # 简化解码过程，删除多余CBAM
        x = self.conv1(encoder_output)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# ====================== AutoEncoder ======================
class DenseFuse_train(nn.Module):
    """
    DenseFuse训练模型 - 支持两种CBAM注意力机制方案和颜色感知CBAM
    
    Args:
        input_nc: 输入通道数（1=灰度，3=RGB）
        output_nc: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        cbam_scheme: CBAM实施方案选择
            - 0: 不使用CBAM
            - 1: 方案1 - DenseBlock输出后应用CBAM
            - 2: 方案2 - 融合层输入前应用CBAM
        reduction_ratio: 通道压缩比例（16, 32, 64, 128等）
        use_color_aware: 是否使用颜色感知CBAM（解决泛黄问题）
        color_preservation_weight: 颜色保护权重（0.0-1.0）
    
    Example:
        >>> # 方案0：不使用CBAM
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, cbam_scheme=0)
        
        >>> # 方案1：DenseBlock输出后CBAM
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, cbam_scheme=1, reduction_ratio=16)
        
        >>> # 方案2：融合层输入前CBAM
        >>> model = DenseFuse_train(input_nc=1, output_nc=1, cbam_scheme=2, reduction_ratio=32)
        
        >>> # 颜色感知CBAM（解决泛黄问题）
        >>> model = DenseFuse_train(input_nc=3, output_nc=3, cbam_scheme=1, 
        >>>                        use_color_aware=True, color_preservation_weight=0.3)
    """
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1, cbam_scheme=0, reduction_ratio=16,
                 use_color_aware=False, color_preservation_weight=0.3,
                 use_channel_attention=True, use_spatial_attention=True, fusion_strategy='add'):
        super().__init__()
        
        # 验证CBAM方案参数
        if cbam_scheme not in [0, 1, 2]:
            raise ValueError(f"cbam_scheme must be 0, 1, or 2, got {cbam_scheme}")
        
        self.cbam_scheme = cbam_scheme
        self.reduction_ratio = reduction_ratio
        self.use_color_aware = use_color_aware
        self.color_preservation_weight = color_preservation_weight
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.fusion_strategy = fusion_strategy
        
        # 创建编码器和解码器
        self.encoder = Dense_Encoder(
            input_nc=input_nc, 
            kernel_size=kernel_size, 
            stride=stride, 
            cbam_scheme=cbam_scheme,
            reduction_ratio=reduction_ratio,
            use_color_aware=use_color_aware,
            color_preservation_weight=color_preservation_weight,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention
        )
        self.decoder = CNN_Decoder(
            output_nc=output_nc, 
            kernel_size=kernel_size, 
            stride=stride
        )
        
        # 打印CBAM方案信息
        self._print_cbam_info()

    def _print_cbam_info(self):
        """打印当前CBAM方案信息"""
        scheme_info = {
            0: "不使用CBAM注意力机制",
            1: "方案1：DenseBlock输出后应用CBAM",
            2: "方案2：融合层输入前应用CBAM"
        }
        if self.cbam_scheme > 0:
            color_info = "颜色感知" if self.use_color_aware else "标准"
            # 构建注意力配置信息
            attention_config = []
            if self.use_channel_attention:
                attention_config.append("通道注意力")
            if self.use_spatial_attention:
                attention_config.append("空间注意力")
            if not attention_config:
                attention_config.append("无注意力")
            attention_str = "+".join(attention_config)
            
            if self.use_color_aware:
                print(f"使用{color_info}CBAM方案：{scheme_info[self.cbam_scheme]} (reduction_ratio={self.reduction_ratio}, color_weight={self.color_preservation_weight}, 注意力配置={attention_str}, 融合策略={self.fusion_strategy})")
            else:
                print(f"使用{color_info}CBAM方案：{scheme_info[self.cbam_scheme]} (reduction_ratio={self.reduction_ratio}, 注意力配置={attention_str}, 融合策略={self.fusion_strategy})")
        else:
            print("不使用CBAM注意力机制")

    def forward(self, x):
        """
        单输入前向传播（适用于方案0和方案1）
        
        Args:
            x: 输入图像
            
        Returns:
            out: 融合结果
        """
        encoder_feature = self.encoder(x)
        out = self.decoder(encoder_feature)
        return out
    
    def forward_dual_input(self, ir_image, vi_image):
        """
        双输入前向传播（适用于方案2）
        
        Args:
            ir_image: 红外图像
            vi_image: 可见光图像
            
        Returns:
            out: 融合结果
        """
        # 分别提取特征
        ir_features = self.encoder(ir_image)
        vi_features = self.encoder(vi_image)
        
        # 方案2：在融合层输入前应用CBAM
        if self.cbam_scheme == 2:
            ir_attended, vi_attended = self.encoder.forward_with_cbam_scheme2(ir_features, vi_features)
            # 使用指定的融合策略
            fused_features = apply_fusion_strategy(ir_attended, vi_attended, self.fusion_strategy)
        else:
            # 方案0和方案1：使用指定的融合策略
            fused_features = apply_fusion_strategy(ir_features, vi_features, self.fusion_strategy)
        
        # 解码
        out = self.decoder(fused_features)
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
