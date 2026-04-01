# Author: wokaka209
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedFusionStrategy:
    """高级融合策略 - 专门优化EN、AG、MI、Qabf指标"""
    
    def __init__(self):
        pass
    
    def enhanced_adaptive_l1(self, feature1, feature2):
        """
        增强版自适应L1融合 - 结合多种信息源
        
        改进点：
        1. L1范数（反映特征强度）
        2. 局部方差（反映纹理丰富度）
        3. 局部梯度（反映边缘信息）
        4. 局部对比度（反映细节清晰度）
        """
        # 1. 计算L1范数（特征强度）
        l1_norm1 = torch.sum(torch.abs(feature1), dim=1, keepdim=True)
        l1_norm2 = torch.sum(torch.abs(feature2), dim=1, keepdim=True)
        
        # 2. 计算局部方差（纹理丰富度）
        def compute_local_variance(x, kernel_size=3):
            pad = kernel_size // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            local_mean = F.avg_pool2d(x_padded, kernel_size, stride=1)
            local_mean_sq = local_mean ** 2
            local_sq_mean = F.avg_pool2d(x_padded ** 2, kernel_size, stride=1)
            local_var = local_sq_mean - local_mean_sq
            return torch.clamp(local_var, min=0)
        
        local_var1 = compute_local_variance(feature1)
        local_var2 = compute_local_variance(feature2)
        
        # 3. 计算局部梯度（边缘信息）
        def compute_gradient(x):
            grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))
            grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            return grad
        
        grad1 = compute_gradient(feature1)
        grad2 = compute_gradient(feature2)
        
        # 4. 计算局部对比度（细节清晰度）
        def compute_local_contrast(x, kernel_size=5):
            pad = kernel_size // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            local_mean = F.avg_pool2d(x_padded, kernel_size, stride=1)
            local_std = torch.sqrt(compute_local_variance(x_padded[:, :, pad:-pad, pad:-pad], kernel_size))
            contrast = local_std / (local_mean + 1e-8)
            return contrast
        
        contrast1 = compute_local_contrast(feature1)
        contrast2 = compute_local_contrast(feature2)
        
        # 5. 综合能量计算（加权融合多种信息）
        # 权重配置（可调整）
        w_l1 = 1.0      # L1范数权重
        w_var = 0.4      # 局部方差权重
        w_grad = 0.3      # 梯度权重
        w_contrast = 0.2   # 对比度权重
        
        energy1 = w_l1 * l1_norm1 + w_var * local_var1 + w_grad * grad1 + w_contrast * contrast1
        energy2 = w_l1 * l1_norm2 + w_var * local_var2 + w_grad * grad2 + w_contrast * contrast2
        
        # 6. 计算融合权重
        total_energy = energy1 + energy2 + 1e-8
        weight1 = energy1 / total_energy
        weight2 = energy2 / total_energy
        
        # 7. 应用加权融合
        fused_features = weight1 * feature1 + weight2 * feature2
        return fused_features
    
    def multi_scale_fusion(self, feature1, feature2):
        """
        多尺度融合策略 - 在不同尺度上融合特征
        
        优势：
        - 保留大尺度结构信息
        - 保留小尺度细节信息
        """
        scales = [1, 2, 4]
        fused_results = []
        
        for scale in scales:
            # 下采样
            if scale > 1:
                feat1_down = F.avg_pool2d(feature1, scale, stride=scale)
                feat2_down = F.avg_pool2d(feature2, scale, stride=scale)
            else:
                feat1_down = feature1
                feat2_down = feature2
            
            # 在当前尺度上融合
            fused_scale = self.enhanced_adaptive_l1(feat1_down, feat2_down)
            
            # 上采样回原始尺寸
            if scale > 1:
                fused_scale = F.interpolate(fused_scale, size=feature1.shape[2:], mode='bilinear', align_corners=False)
            
            fused_results.append(fused_scale)
        
        # 加权融合多尺度结果
        weights = [0.5, 0.3, 0.2]  # 大尺度权重更高
        fused_features = sum(w * f for w, f in zip(weights, fused_results))
        
        return fused_features
    
    def gradient_guided_fusion(self, feature1, feature2):
        """
        梯度引导融合 - 优先保留梯度大的区域（边缘和细节）
        """
        def compute_gradient_magnitude(x):
            grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            
            grad_x_padded = F.pad(grad_x, (0, 1, 0, 0))
            grad_y_padded = F.pad(grad_y, (0, 0, 0, 1))
            
            grad_mag = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2)
            return grad_mag
        
        grad_mag1 = compute_gradient_magnitude(feature1)
        grad_mag2 = compute_gradient_magnitude(feature2)
        
        total_grad = grad_mag1 + grad_mag2 + 1e-8
        weight1 = grad_mag1 / total_grad
        weight2 = grad_mag2 / total_grad
        
        fused_features = weight1 * feature1 + weight2 * feature2
        
        return fused_features
    
    def hybrid_fusion(self, feature1, feature2):
        """
        混合融合策略 - 结合多种融合策略的优势
        
        组合：
        1. 增强版自适应L1（80%权重）
        2. 梯度引导融合（10%权重）
        3. 多尺度融合（10%权重）
        """
        # 策略1：增强版自适应L1
        fusion1 = self.enhanced_adaptive_l1(feature1, feature2)
        
        # 策略2：梯度引导融合
        fusion2 = self.gradient_guided_fusion(feature1, feature2)
        
        # 策略3：多尺度融合
        fusion3 = self.multi_scale_fusion(feature1, feature2)
        
        # 加权融合
        fused_features = 0.9 * fusion1 + 0.1 * fusion2 + 0.1 * fusion3
        
        return fused_features


def apply_fusion_strategy(feature1, feature2, strategy='hybrid'):
    """
    应用融合策略的统一接口
    
    Args:
        feature1: 第一个特征图 (B, C, H, W)
        feature2: 第二个特征图 (B, C, H, W)
        strategy: 融合策略名称
            - 'enhanced_l1': 增强版自适应L1
            - 'multi_scale': 多尺度融合
            - 'gradient': 梯度引导融合
            - 'hybrid': 混合融合（推荐）
    
    Returns:
        fused_features: 融合后的特征图
    """
    fusion_obj = AdvancedFusionStrategy()
    
    if strategy == 'enhanced_l1':
        return fusion_obj.enhanced_adaptive_l1(feature1, feature2)
    elif strategy == 'multi_scale':
        return fusion_obj.multi_scale_fusion(feature1, feature2)
    elif strategy == 'gradient':
        return fusion_obj.gradient_guided_fusion(feature1, feature2)
    elif strategy == 'hybrid':
        return fusion_obj.hybrid_fusion(feature1, feature2)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


if __name__ == "__main__":
    # 测试融合策略
    print("测试高级融合策略...")
    
    # 创建随机特征图
    feat1 = torch.randn(1, 64, 256, 256)
    feat2 = torch.randn(1, 64, 256, 256)
    
    # 测试不同策略
    strategies = ['enhanced_l1', 'multi_scale', 'gradient', 'hybrid']
    
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        fused = apply_fusion_strategy(feat1, feat2, strategy=strategy)
        print(f"输入形状: {feat1.shape}")
        print(f"输出形状: {fused.shape}")
        print(f"✓ {strategy} 策略测试成功")
    
    print("\n所有融合策略测试完成！")