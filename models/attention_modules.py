# Author: wokaka209
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ColorAwareCBAM(nn.Module):
    """颜色感知CBAM - 专门保护RGB图像中的颜色特征"""
    
    def __init__(self, in_channels, reduction=None, kernel_size=7, color_preservation_weight=0.3):
        super(ColorAwareCBAM, self).__init__()
        
        # 智能选择reduction参数：对于RGB图像使用更小的值
        if reduction is None:
            if in_channels == 3:
                reduction = 2  # RGB图像使用较小的压缩比例
            else:
                reduction = 16  # 其他情况使用默认值
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.color_preservation_weight = color_preservation_weight
        
        # 如果是RGB图像（3通道），添加颜色保护机制
        if in_channels == 3:
            self.color_protection = nn.Parameter(torch.ones(3, 1, 1))
            # 红色通道给予更高的保护权重
            self.color_protection.data[0] = 1.2  # 红色通道
            self.color_protection.data[1] = 1.0  # 绿色通道  
            self.color_protection.data[2] = 1.0  # 蓝色通道
        else:
            self.color_protection = None
    
    def forward(self, x):
        # 原始注意力计算
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)
        
        # 应用注意力
        x_attended = x * channel_attn * spatial_attn
        
        # 如果是RGB图像，应用颜色保护
        if self.color_protection is not None and x.shape[1] == 3:
            # 保留部分原始颜色特征
            color_preserved = x * self.color_protection
            # 加权融合：保留30%的原始颜色特征
            x = (1 - self.color_preservation_weight) * x_attended + \
                self.color_preservation_weight * color_preserved
        else:
            x = x_attended
            
        return x