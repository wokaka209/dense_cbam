# Author: wokaka209
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedFusionStrategyOptimized:
    """高级融合策略优化版 - 消除计算冗余，提升速度"""
    
    def __init__(self, hybrid_weights=None):
        """
        初始化高级融合策略
        
        Args:
            hybrid_weights: 混合融合权重配置 [w_l1, w_gradient, w_multi_scale]
                - w_l1: 增强版自适应L1权重（推荐0.6-0.9）
                - w_gradient: 梯度引导融合权重（推荐0.05-0.25）
                - w_multi_scale: 多尺度融合权重（推荐0.05-0.15）
                默认值为 [0.8, 0.11, 0.09]
        """
        if hybrid_weights is None:
            # 优化的默认权重配置
            # 基于实验验证的最佳平衡点
            self.hybrid_weights = {
                'l1_weight': 0.75,      # 主导策略，保证基础质量
                'gradient_weight': 0.15, # 边缘增强，提升清晰度
                'multi_scale_weight': 0.10 # 结构保持，保留细节
            }
        else:
            if len(hybrid_weights) != 3:
                raise ValueError(f"hybrid_weights必须包含3个元素: [l1, gradient, multi_scale]")
            
            self.hybrid_weights = {
                'l1_weight': hybrid_weights[0],
                'gradient_weight': hybrid_weights[1],
                'multi_scale_weight': hybrid_weights[2]
            }
        
        # 验证权重和为1.0（允许一定误差）
        total_weight = sum(self.hybrid_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"[警告] 权重总和不为1.0 ({total_weight:.3f})，将自动归一化")
            for key in self.hybrid_weights:
                self.hybrid_weights[key] /= total_weight
    
    def compute_common_features(self, feature1, feature2):
        """
        预计算所有特征，避免重复计算
        
        Returns:
            dict: 包含所有预计算特征的字典
        """
        features = {}
        
        # 1. L1范数（特征强度）
        features['l1_norm1'] = torch.sum(torch.abs(feature1), dim=1, keepdim=True)
        features['l1_norm2'] = torch.sum(torch.abs(feature2), dim=1, keepdim=True)
        
        # 2. 梯度计算（边缘信息）- 预计算一次，多处复用
        grad_x1 = torch.abs(feature1[:, :, :, 1:] - feature1[:, :, :, :-1])
        grad_y1 = torch.abs(feature1[:, :, 1:, :] - feature1[:, :, :-1, :])
        grad_x1 = F.pad(grad_x1, (0, 1, 0, 0))
        grad_y1 = F.pad(grad_y1, (0, 0, 0, 1))
        features['grad1'] = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2 + 1e-8)
        
        grad_x2 = torch.abs(feature2[:, :, :, 1:] - feature2[:, :, :, :-1])
        grad_y2 = torch.abs(feature2[:, :, 1:, :] - feature2[:, :, :-1, :])
        grad_x2 = F.pad(grad_x2, (0, 1, 0, 0))
        grad_y2 = F.pad(grad_y2, (0, 0, 0, 1))
        features['grad2'] = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2 + 1e-8)
        
        # 3. 局部方差（纹理丰富度）- 使用融合后的局部均值
        def compute_local_var_fast(x, kernel_size=3):
            pad = kernel_size // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            local_mean = F.avg_pool2d(x_padded, kernel_size, stride=1)
            local_sq_mean = F.avg_pool2d(x_padded ** 2, kernel_size, stride=1)
            local_var = local_sq_mean - local_mean ** 2
            return torch.clamp(local_var, min=0)
        
        features['local_var1'] = compute_local_var_fast(feature1)
        features['local_var2'] = compute_local_var_fast(feature2)
        
        return features
    
    def adaptive_l1_with_precomputed(self, feature1, feature2, features, weights=None):
        """
        使用预计算特征的增强版自适应L1融合
        
        Args:
            feature1, feature2: 输入特征
            features: 预计算特征字典
            weights: 自定义权重配置
        """
        if weights is None:
            weights = {'l1': 1.0, 'var': 0.4, 'grad': 0.3, 'contrast': 0.2}
        
        l1_norm1 = features['l1_norm1']
        l1_norm2 = features['l1_norm2']
        local_var1 = features['local_var1']
        local_var2 = features['local_var2']
        grad1 = features['grad1']
        grad2 = features['grad2']
        
        # 对比度计算（复用局部统计量）
        contrast1 = torch.sqrt(local_var1 + 1e-8) / (l1_norm1 / feature1.shape[1] + 1e-8)
        contrast2 = torch.sqrt(local_var2 + 1e-8) / (l1_norm2 / feature2.shape[1] + 1e-8)
        
        # 综合能量计算
        energy1 = (weights['l1'] * l1_norm1 + 
                   weights['var'] * local_var1 + 
                   weights['grad'] * grad1 + 
                   weights['contrast'] * contrast1)
        energy2 = (weights['l1'] * l1_norm2 + 
                   weights['var'] * local_var2 + 
                   weights['grad'] * grad2 + 
                   weights['contrast'] * contrast2)
        
        # 融合权重
        total_energy = energy1 + energy2 + 1e-8
        weight1 = energy1 / total_energy
        weight2 = energy2 / total_energy
        
        return weight1 * feature1 + weight2 * feature2
    
    def gradient_guided_with_precomputed(self, feature1, feature2, features):
        """
        使用预计算特征的梯度引导融合
        """
        grad1 = features['grad1']
        grad2 = features['grad2']
        
        total_grad = grad1 + grad2 + 1e-8
        weight1 = grad1 / total_grad
        weight2 = grad2 / total_grad
        
        return weight1 * feature1 + weight2 * feature2
    
    def multi_scale_with_precomputed(self, feature1, feature2, features, weights=None):
        """
        优化的多尺度融合 - 避免重复调用 enhanced_adaptive_l1
        
        优化点：
        1. 复用预计算的梯度信息
        2. 减少重复的池化和上采样操作
        """
        if weights is None:
            weights = {'l1': 1.0, 'var': 0.4, 'grad': 0.3, 'contrast': 0.2}
        
        scales = [1, 2, 4]
        scale_weights = [0.5, 0.3, 0.2]
        fused_results = []
        
        for scale in scales:
            if scale > 1:
                # 下采样
                feat1_down = F.avg_pool2d(feature1, scale, stride=scale)
                feat2_down = F.avg_pool2d(feature2, scale, stride=scale)
                
                # 下采样的预计算特征
                l1_norm1_down = torch.sum(torch.abs(feat1_down), dim=1, keepdim=True)
                l1_norm2_down = torch.sum(torch.abs(feat2_down), dim=1, keepdim=True)
                
                # 局部方差（下采样后纹理计算更快）
                var1_down = F.avg_pool2d(
                    F.pad(feat1_down, (1, 1, 1, 1), mode='reflect'), 3, stride=1
                )
                var2_down = F.avg_pool2d(
                    F.pad(feat2_down, (1, 1, 1, 1), mode='reflect'), 3, stride=1
                )
                var1_down = torch.clamp(var1_down - var1_down ** 2, min=0)
                var2_down = torch.clamp(var2_down - var2_down ** 2, min=0)
                
                # 梯度（简化为L1范数在下采样尺度上更快）
                grad1_down = torch.sum(torch.abs(feat1_down), dim=1, keepdim=True)
                grad2_down = torch.sum(torch.abs(feat2_down), dim=1, keepdim=True)
                
                # 对比度
                contrast1_down = torch.sqrt(var1_down + 1e-8) / (l1_norm1_down + 1e-8)
                contrast2_down = torch.sqrt(var2_down + 1e-8) / (l1_norm2_down + 1e-8)
                
                energy1 = (weights['l1'] * l1_norm1_down + 
                           weights['var'] * var1_down + 
                           weights['grad'] * grad1_down + 
                           weights['contrast'] * contrast1_down)
                energy2 = (weights['l1'] * l1_norm2_down + 
                           weights['var'] * var2_down + 
                           weights['grad'] * grad2_down + 
                           weights['contrast'] * contrast2_down)
                
            else:
                # scale=1 时使用原始预计算特征
                energy1 = (weights['l1'] * features['l1_norm1'] + 
                           weights['var'] * features['local_var1'] + 
                           weights['grad'] * features['grad1'] + 
                           weights['contrast'] * (
                               torch.sqrt(features['local_var1'] + 1e-8) / 
                               (features['l1_norm1'] / feature1.shape[1] + 1e-8)
                           ))
                energy2 = (weights['l1'] * features['l1_norm2'] + 
                           weights['var'] * features['local_var2'] + 
                           weights['grad'] * features['grad2'] + 
                           weights['contrast'] * (
                               torch.sqrt(features['local_var2'] + 1e-8) / 
                               (features['l1_norm2'] / feature2.shape[1] + 1e-8)
                           ))
            
            total_energy = energy1 + energy2 + 1e-8
            weight1 = energy1 / total_energy
            weight2 = energy2 / total_energy
            fused_scale = weight1 * (feat1_down if scale > 1 else feature1) + \
                          weight2 * (feat2_down if scale > 1 else feature2)
            
            if scale > 1:
                fused_scale = F.interpolate(fused_scale, size=feature1.shape[2:], 
                                            mode='bilinear', align_corners=False)
            
            fused_results.append(fused_scale)
        
        return sum(w * f for w, f in zip(scale_weights, fused_results))
    
    def hybrid_fusion_optimized(self, feature1, feature2):
        """
        优化的混合融合策略
        
        优化点：
        1. 一次性预计算所有公共特征
        2. 避免 enhanced_adaptive_l1 重复调用4次
        3. 复用梯度、方差等中间结果
        4. 减少内存分配和带宽占用
        
        预计性能提升：3-4倍
        """
        # 步骤1：预计算所有公共特征（仅计算一次）
        features = self.compute_common_features(feature1, feature2)
        
        # 步骤2：使用预计算特征进行三种融合
        fusion1 = self.adaptive_l1_with_precomputed(feature1, feature2, features)
        fusion2 = self.gradient_guided_with_precomputed(feature1, feature2, features)
        fusion3 = self.multi_scale_with_precomputed(feature1, feature2, features)
        
        # 步骤3：使用可配置权重进行加权融合
        w_l1 = self.hybrid_weights['l1_weight']
        w_gradient = self.hybrid_weights['gradient_weight']
        w_multi_scale = self.hybrid_weights['multi_scale_weight']
        
        fused_features = w_l1 * fusion1 + w_gradient * fusion2 + w_multi_scale * fusion3
        
        return fused_features

    def enhanced_adaptive_l1(self, feature1, feature2):
        """
        增强版自适应L1融合 - 兼容原始接口
        
        调用优化版本实现
        """
        features = self.compute_common_features(feature1, feature2)
        return self.adaptive_l1_with_precomputed(feature1, feature2, features)
    
    def multi_scale_fusion(self, feature1, feature2):
        """
        多尺度融合 - 兼容原始接口
        
        调用优化版本实现
        """
        features = self.compute_common_features(feature1, feature2)
        return self.multi_scale_with_precomputed(feature1, feature2, features)
    
    def gradient_guided_fusion(self, feature1, feature2):
        """
        梯度引导融合 - 兼容原始接口
        
        调用优化版本实现
        """
        features = self.compute_common_features(feature1, feature2)
        return self.gradient_guided_with_precomputed(feature1, feature2, features)
    
    def hybrid_fusion(self, feature1, feature2):
        """
        混合融合策略 - 兼容原始接口
        
        调用优化版本实现
        """
        return self.hybrid_fusion_optimized(feature1, feature2)
    
    def get_weights(self):
        """获取当前混合融合权重"""
        return self.hybrid_weights.copy()
    
    def set_weights(self, l1_weight=None, gradient_weight=None, multi_scale_weight=None):
        """
        设置混合融合权重
        
        Args:
            l1_weight: 增强版自适应L1权重（0.6-0.9）
            gradient_weight: 梯度引导融合权重（0.05-0.25）
            multi_scale_weight: 多尺度融合权重（0.05-0.15）
        """
        if l1_weight is not None:
            self.hybrid_weights['l1_weight'] = l1_weight
        if gradient_weight is not None:
            self.hybrid_weights['gradient_weight'] = gradient_weight
        if multi_scale_weight is not None:
            self.hybrid_weights['multi_scale_weight'] = multi_scale_weight
        
        # 自动归一化
        total_weight = sum(self.hybrid_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"[警告] 权重总和不为1.0 ({total_weight:.3f})，自动归一化")
            for key in self.hybrid_weights:
                self.hybrid_weights[key] /= total_weight
    
    @staticmethod
    def get_preset_config(preset_name='balanced'):
        """
        获取预设的权重配置
        
        Args:
            preset_name: 预设名称
                - 'balanced': 平衡配置（默认）[0.75, 0.15, 0.10]
                - 'quality': 高质量优先 [0.80, 0.12, 0.08]
                - 'detail': 细节增强 [0.70, 0.20, 0.10]
                - 'speed': 快速处理 [0.85, 0.10, 0.05]
        
        Returns:
            list: 权重配置 [l1, gradient, multi_scale]
        """
        presets = {
            'balanced': [0.75, 0.15, 0.10],      # 平衡质量与速度
            'quality': [0.80, 0.12, 0.08],      # 高质量优先，适合精细图像
            'detail': [0.70, 0.20, 0.10],      # 细节增强，适合边缘丰富的场景
            'speed': [0.85, 0.10, 0.05],       # 快速处理，减少计算量
            'edge_enhanced': [0.65, 0.25, 0.10], # 边缘增强，适合医学影像
            'structure_preserve': [0.72, 0.13, 0.15] # 结构保持，适合建筑/场景
        }
        
        if preset_name not in presets:
            raise ValueError(f"未知预设配置: {preset_name}。可选预设: {list(presets.keys())}")
        
        return presets[preset_name]


def benchmark_fusion_methods(feature1, feature2, num_iterations=100):
    """
    性能基准测试
    
    Args:
        feature1, feature2: 测试特征
        num_iterations: 迭代次数
    """
    import time
    
    # 导入原始版本
    from advanced_fusion import AdvancedFusionStrategy
    original = AdvancedFusionStrategy()
    
    # 优化版本
    optimized = AdvancedFusionStrategyOptimized()
    
    # GPU预热
    for _ in range(10):
        _ = original.hybrid_fusion(feature1, feature2)
        _ = optimized.hybrid_fusion_optimized(feature1, feature2)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # 测试原始版本
    start = time.time()
    for _ in range(num_iterations):
        _ = original.hybrid_fusion(feature1, feature2)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start
    
    # 测试优化版本
    start = time.time()
    for _ in range(num_iterations):
        _ = optimized.hybrid_fusion_optimized(feature1, feature2)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    optimized_time = time.time() - start
    
    speedup = original_time / optimized_time
    
    print(f"原始版本平均耗时: {original_time/num_iterations*1000:.2f} ms")
    print(f"优化版本平均耗时: {optimized_time/num_iterations*1000:.2f} ms")
    print(f"性能提升: {speedup:.2f}x")


if __name__ == "__main__":
    # 测试性能
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    channels = 64
    height, width = 128, 128
    
    feature1 = torch.randn(batch_size, channels, height, width).to(device)
    feature2 = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"特征尺寸: {feature1.shape}")
    print(f"测试融合性能...")
    
    benchmark_fusion_methods(feature1, feature2, num_iterations=50)