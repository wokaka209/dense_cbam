# -*- coding: utf-8 -*-
"""
@file name:batch_fusion_optimized.py
@desc: 优化版批量融合脚本 - 支持CBAM注意力机制和高级融合策略
@Writer: wokaka209
@Date: 2026-04-05
"""
import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
import argparse


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='支持CBAM的批量图像融合')
    
    # 基础参数
    parser.add_argument('--ir_dir', type=str, 
                        default='E:/whx_Graduation project/baseline_project/dataset/ir', 
                        help='红外图像目录')
    parser.add_argument('--vi_dir', type=str, 
                        default='E:/whx_Graduation project/baseline_project/dataset/vi', 
                        help='可见光图像目录')
    parser.add_argument('--output_dir', type=str, 
                        default='data_result/RGB_CBAM1_spatial_bidirectional_04-09_15-57', 
                        help='输出目录')
    parser.add_argument('--model_weights', type=str, 
                        default='runs/RGB_CBAM1_spatial_bidirectional_04-09_15-57/checkpoints/best.pth', 
                        help='模型权重路径')
    
    # CBAM参数
    parser.add_argument('--cbam_scheme', type=int, 
                        default=1, 
                        choices=[0, 1, 2],
                        help='CBAM实施方案选择: 0=不使用, 1=方案1, 2=方案2')
    parser.add_argument('--reduction_ratio', type=int, 
                        default=2, 
                        choices=[2,8, 16, 32, 64, 128, 256],
                        help='CBAM通道压缩比例')
    parser.add_argument('--use_color_aware', action='store_true', 
                        default=True,
                        help='是否使用颜色感知CBAM（解决泛黄问题）')
    parser.add_argument('--color_preservation_weight', type=float, 
                        default=0.5, 
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='颜色保护权重（0.0-1.0），推荐0.4')
    
    # CBAM消融实验参数（仅在CBAM方案不为0时有效）
    parser.add_argument('--use_channel_attention', action='store_true', default=False,
                        help='是否启用通道注意力（仅在CBAM方案不为0时有效）')
    parser.add_argument('--use_spatial_attention', action='store_true', default=True,
                        help='是否启用空间注意力（仅在CBAM方案不为0时有效）')
    
    # 融合策略参数
    parser.add_argument('--fusion_strategy', type=str, 
                        default='mean',
                        choices=['mean', 'max', 'l1norm', 'adaptive_l1', 'gradient_based', 
                                'enhanced_l1', 'multi_scale', 'gradient', 'hybrid'],
                        help='融合策略选择: mean=平均, max=最大值, l1norm=L1范数, adaptive_l1=自适应L1, gradient_based=基于梯度, enhanced_l1=增强L1, multi_scale=多尺度, gradient=梯度引导, hybrid=混合融合')
    
    # 混合融合权重配置（仅对hybrid策略有效）
    parser.add_argument('--hybrid_weights_preset', type=str,
                        default='balanced',
                        choices=['balanced', 'quality', 'detail', 'speed', 'edge_enhanced', 'structure_preserve'],
                        help='混合融合权重预设: balanced=平衡(默认), quality=高质量, detail=细节增强, speed=快速处理, edge_enhanced=边缘增强, structure_preserve=结构保持（仅在hybrid策略下有效）')
    
    # 其他参数
    parser.add_argument('--gray', action='store_true', 
                        default=False,
                        help='是否使用灰度模式')
    
    return parser.parse_args()


# 导入新的CBAM模型
from models.DenseFuse import DenseFuse_train

# 导入融合策略模块
try:
    from fusion_strategy.advanced_fusion import apply_fusion_strategy
    from utils.util_fusion import FusionConfig as BaseFusionConfig
    FUSION_STRATEGY_AVAILABLE = True
except ImportError:
    print("[警告] 高级融合策略模块不可用，使用简单融合策略")
    FUSION_STRATEGY_AVAILABLE = False

# 参数配置类
class FusionConfig:
    """融合配置类 - 统一管理所有参数"""
    
    # 类属性：默认配置值（与命令行参数保持一致）
    DEFAULT_CONFIG = {
        # 基础参数
        'ir_dir': 'E:/whx_Graduation project/baseline_project/dataset/ir',
        'vi_dir': 'E:/whx_Graduation project/baseline_project/dataset/vi',
        'output_dir': 'data_result/batch_fusion_optimized_rgb_no_cbam_add_bidirectional',
        'model_weights': 'runs/RGB_noCBAM_add_bidirectional_04-08_11-44/checkpoints/best.pth',
        
        # CBAM参数
        'cbam_scheme': 0,  # 0=不使用, 1=方案1, 2=方案2（与命令行默认值一致）
        'reduction_ratio': None,  # CBAM通道压缩比例（与命令行默认值一致）
        'use_color_aware': True,  # 是否使用颜色感知CBAM（解决泛黄问题）
        'color_preservation_weight': 0.4,  # 颜色保护权重（0.0-1.0）
        
        # CBAM消融实验参数（仅在CBAM方案不为0时有效）
        'use_channel_attention': False,  # 是否启用通道注意力
        'use_spatial_attention': True,  # 是否启用空间注意力
        
        # 融合算法参数
        'fusion_strategy': 'mean',  # 融合策略选择（与命令行默认值一致）
        'hybrid_weights_preset': 'balanced',  # 混合融合权重预设（与命令行默认值一致）
        'gray': False,  # 是否使用灰度模式
        'model_name': 'DenseFuse',
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        
        # 图像处理参数
        'target_size': (768, 1024)
    }
    
    def __init__(self, args=None):
        """
        初始化配置
        
        Args:
            args: 命令行参数对象，如果提供则使用参数值，否则使用默认值
        """
        # 使用类默认配置初始化所有属性
        for key, value in self.DEFAULT_CONFIG.items():
            setattr(self, key, value)
        
        # 如果提供了命令行参数，更新配置
        if args is not None:
            self.update_from_args(args)
        
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        if hasattr(args, 'ir_dir') and args.ir_dir:
            self.ir_dir = args.ir_dir
        if hasattr(args, 'vi_dir') and args.vi_dir:
            self.vi_dir = args.vi_dir
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = args.output_dir
        if hasattr(args, 'model_weights') and args.model_weights:
            self.model_weights = args.model_weights
        if hasattr(args, 'fusion_strategy') and args.fusion_strategy:
            self.fusion_strategy = args.fusion_strategy
        if hasattr(args, 'hybrid_weights_preset') and args.hybrid_weights_preset:
            self.hybrid_weights_preset = args.hybrid_weights_preset
        if hasattr(args, 'cbam_scheme') and args.cbam_scheme is not None:
            self.cbam_scheme = args.cbam_scheme
        if hasattr(args, 'reduction_ratio') and args.reduction_ratio is not None:
            self.reduction_ratio = args.reduction_ratio
        if hasattr(args, 'use_color_aware') and args.use_color_aware is not None:
            self.use_color_aware = args.use_color_aware
        if hasattr(args, 'color_preservation_weight') and args.color_preservation_weight is not None:
            self.color_preservation_weight = args.color_preservation_weight
        # CBAM消融实验参数
        if hasattr(args, 'use_channel_attention') and args.use_channel_attention is not None:
            self.use_channel_attention = args.use_channel_attention
        if hasattr(args, 'use_spatial_attention') and args.use_spatial_attention is not None:
            self.use_spatial_attention = args.use_spatial_attention
        if hasattr(args, 'gray') and args.gray is not None:
            self.gray = args.gray
    
    def to_dict(self):
        """将配置转换为字典格式"""
        return {key: getattr(self, key) for key in self.DEFAULT_CONFIG.keys()}
    
    def validate(self):
        """验证配置参数的有效性"""
        # 检查路径是否存在
        if not os.path.exists(self.ir_dir):
            raise ValueError(f"红外图像目录不存在: {self.ir_dir}")
        if not os.path.exists(self.vi_dir):
            raise ValueError(f"可见光图像目录不存在: {self.vi_dir}")
        
        # 检查模型权重文件是否存在
        if not os.path.exists(self.model_weights):
            raise ValueError(f"模型权重文件不存在: {self.model_weights}")
        
        # 检查参数范围
        if self.cbam_scheme not in [0, 1, 2]:
            raise ValueError(f"CBAM方案参数无效: {self.cbam_scheme}")
        if self.reduction_ratio not in [2, 8, 16, 32, 64, 128, 256]:
            raise ValueError(f"reduction_ratio参数无效: {self.reduction_ratio}")
        if not 0.1 <= self.color_preservation_weight <= 0.5:
            raise ValueError(f"颜色保护权重超出范围: {self.color_preservation_weight}")
        
        return True
    
    def __str__(self):
        """返回配置的字符串表示"""
        config_info = [
            f"FusionConfig:",
            f"  红外图像目录: {self.ir_dir}",
            f"  可见光图像目录: {self.vi_dir}",
            f"  输出目录: {self.output_dir}",
            f"  模型权重: {self.model_weights}",
            f"  CBAM方案: {self.cbam_scheme}",
            f"  reduction_ratio: {self.reduction_ratio}",
            f"  颜色感知CBAM: {self.use_color_aware}",
            f"  颜色保护权重: {self.color_preservation_weight}",
            f"  通道注意力: {self.use_channel_attention}",
            f"  空间注意力: {self.use_spatial_attention}",
            f"  灰度模式: {self.gray}",
            f"  设备: {self.device}"
        ]
        return "\n".join(config_info)


class BatchImageFusionOptimized:
    def __init__(self, config):
        self.config = config
        self.target_size = config.target_size
        self.load_model()

    def load_model(self):
        in_channel = 1 if self.config.gray else 3
        out_channel = 1 if self.config.gray else 3
        
        # 从权重文件中读取训练时使用的CBAM方案
        checkpoint = torch.load(self.config.model_weights,
                                map_location=self.config.device, weights_only=False)
        
        # 检查权重文件中是否保存了CBAM方案信息
        cbam_scheme = self.config.cbam_scheme  # 默认使用配置中的CBAM方案
        reduction_ratio = self.config.reduction_ratio
        use_color_aware = self.config.use_color_aware
        color_preservation_weight = self.config.color_preservation_weight
        # CBAM消融实验参数
        use_channel_attention = self.config.use_channel_attention
        use_spatial_attention = self.config.use_spatial_attention
        
        if 'cbam_scheme' in checkpoint:
            cbam_scheme = checkpoint['cbam_scheme']
            print(f'[读取] 从权重文件读取CBAM方案: {cbam_scheme}')
        else:
            print(f'[默认] 权重文件未保存CBAM方案信息，使用配置方案: {cbam_scheme}')
            
        if 'reduction_ratio' in checkpoint:
            reduction_ratio = checkpoint['reduction_ratio']
            print(f'[读取] 从权重文件读取reduction_ratio: {reduction_ratio}')
        else:
            print(f'[默认] 权重文件未保存reduction_ratio信息，使用配置值: {reduction_ratio}')
            
        if 'use_color_aware' in checkpoint:
            use_color_aware = checkpoint['use_color_aware']
            print(f'[读取] 从权重文件读取use_color_aware: {use_color_aware}')
        else:
            print(f'[默认] 权重文件未保存use_color_aware信息，使用配置值: {use_color_aware}')
            
        if 'color_preservation_weight' in checkpoint:
            color_preservation_weight = checkpoint['color_preservation_weight']
            print(f'[读取] 从权重文件读取color_preservation_weight: {color_preservation_weight}')
        else:
            print(f'[默认] 权重文件未保存color_preservation_weight信息，使用配置值: {color_preservation_weight}')
        
        # 读取CBAM消融实验参数
        if 'use_channel_attention' in checkpoint:
            use_channel_attention = checkpoint['use_channel_attention']
            print(f'[读取] 从权重文件读取通道注意力: {use_channel_attention}')
        else:
            print(f'[默认] 权重文件未保存通道注意力信息，使用配置值: {use_channel_attention}')
            
        if 'use_spatial_attention' in checkpoint:
            use_spatial_attention = checkpoint['use_spatial_attention']
            print(f'[读取] 从权重文件读取空间注意力: {use_spatial_attention}')
        else:
            print(f'[默认] 权重文件未保存空间注意力信息，使用配置值: {use_spatial_attention}')
        
        # 使用CBAM方案创建模型
        self.model = DenseFuse_train(
            input_nc=in_channel,
            output_nc=out_channel,
            cbam_scheme=cbam_scheme,
            reduction_ratio=reduction_ratio,
            use_color_aware=use_color_aware,
            color_preservation_weight=color_preservation_weight,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention
        )
        self.model = self.model.to(self.config.device)
        
        # 使用strict=False处理权重不匹配（如缺少注意力权重）
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
        
        print(f'[成功] 模型加载成功: {self.config.model_weights}')
        print(f'[方案] 使用CBAM方案: {cbam_scheme} (reduction_ratio={reduction_ratio})')
        if cbam_scheme == 0:
            print('  [方案0] 不使用CBAM注意力机制')
        elif cbam_scheme == 1:
            print('  [方案1] DenseBlock输出后应用CBAM')
        elif cbam_scheme == 2:
            print('  [方案2] 融合层输入前应用CBAM')
        
        # CBAM消融实验配置信息
        if cbam_scheme > 0:
            print(f'[消融实验] 通道注意力: {use_channel_attention}, 空间注意力: {use_spatial_attention}')

    def preprocess_image(self, image_path):
        image = read_image(image_path,
                           mode=ImageReadMode.GRAY if self.config.gray else ImageReadMode.RGB)
        
        original_size = image.shape[1:]
        
        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize(self.target_size),
                                               transforms.ToTensor(),
                                               ])
        image = image_transforms(image).unsqueeze(0)
        return image, original_size

    def run_single(self, ir_path, vi_path, output_path):
        self.model.eval()
        with torch.no_grad():
            ir_image, original_size = self.preprocess_image(ir_path)
            vi_image, _ = self.preprocess_image(vi_path)
            
            ir_image = ir_image.to(self.config.device)
            vi_image = vi_image.to(self.config.device)

            # 根据CBAM方案选择不同的前向传播方式
            if self.model.cbam_scheme == 2:
                # 方案2：使用双输入前向传播
                fused_image = self.model.forward_dual_input(ir_image, vi_image)
            else:
                # 方案0和方案1：分别处理然后融合
                ir_features = self.model.encoder(ir_image)
                vi_features = self.model.encoder(vi_image)
                
                # 简单平均融合策略
                fused_features = (ir_features + vi_features) / 2
                fused_image = self.model.decoder(fused_features)

            fused_image = fused_image.cpu().squeeze(0)
            
            if original_size != self.target_size:
                fused_image = torch.nn.functional.interpolate(
                    fused_image.unsqueeze(0),
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_image(fused_image, output_path)
            return True
    
    def fusion_strategy(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """融合策略 - 支持多种融合算法和混合融合权重配置"""
        
        # 如果高级融合策略可用，使用高级策略
        if FUSION_STRATEGY_AVAILABLE and self.config.fusion_strategy in ['enhanced_l1', 'multi_scale', 'gradient', 'hybrid']:
            try:
                # 对于hybrid策略，使用优化版并支持权重配置
                if self.config.fusion_strategy == 'hybrid':
                    from fusion_strategy.advanced_fusion_optimized import AdvancedFusionStrategyOptimized
                    
                    # 获取预设的权重配置
                    hybrid_weights = AdvancedFusionStrategyOptimized.get_preset_config(
                        self.config.hybrid_weights_preset
                    )
                    
                    # 创建带权重的融合策略实例
                    fusion_obj = AdvancedFusionStrategyOptimized(hybrid_weights=hybrid_weights)
                    
                    print(f"[配置] 使用混合融合策略，预设: {self.config.hybrid_weights_preset}，权重: {hybrid_weights}")
                    
                    return fusion_obj.hybrid_fusion(feature1, feature2)
                else:
                    return apply_fusion_strategy(feature1, feature2, strategy=self.config.fusion_strategy)
            except Exception as e:
                print(f"[警告] 高级融合策略失败，使用简单策略: {e}")
        
        # 简单融合策略
        if self.config.fusion_strategy == 'mean':
            return (feature1 + feature2) / 2
        elif self.config.fusion_strategy == 'max':
            return torch.maximum(feature1, feature2)
        elif self.config.fusion_strategy == 'l1norm':
            # L1范数融合策略
            l1_norm1 = torch.abs(feature1)
            l1_norm2 = torch.abs(feature2)
            mask = (l1_norm1 > l1_norm2).float()
            return mask * feature1 + (1 - mask) * feature2
        elif self.config.fusion_strategy == 'adaptive_l1':
            # 自适应L1范数融合策略
            l1_norm1 = torch.abs(feature1)
            l1_norm2 = torch.abs(feature2)
            total_energy = l1_norm1 + l1_norm2 + 1e-8
            weight1 = l1_norm1 / total_energy
            weight2 = l1_norm2 / total_energy
            return weight1 * feature1 + weight2 * feature2
        elif self.config.fusion_strategy == 'gradient_based':
            # 基于梯度的融合策略
            grad1_x = torch.abs(feature1[:, :, :, 1:] - feature1[:, :, :, :-1])
            grad1_y = torch.abs(feature1[:, :, 1:, :] - feature1[:, :, :-1, :])
            grad2_x = torch.abs(feature2[:, :, :, 1:] - feature2[:, :, :, :-1])
            grad2_y = torch.abs(feature2[:, :, 1:, :] - feature2[:, :, :-1, :])
            
            grad_mag1 = torch.sqrt(grad1_x[:, :, :, :-1]**2 + grad1_y[:, :, :-1, :]**2)
            grad_mag2 = torch.sqrt(grad2_x[:, :, :, :-1]**2 + grad2_y[:, :, :-1, :]**2)
            
            # 扩展梯度信息以匹配原始尺寸
            pad_x = torch.zeros_like(grad_mag1[:, :, :, -1:]).expand(-1, -1, -1, 1)
            grad_mag1 = torch.cat([grad_mag1, pad_x], dim=3)
            grad_mag2 = torch.cat([grad_mag2, pad_x.clone()], dim=3)
            
            pad_y = torch.zeros_like(grad_mag1[:, :, -1:, :]).expand(-1, -1, 1, -1)
            grad_mag1 = torch.cat([grad_mag1, pad_y], dim=2)
            grad_mag2 = torch.cat([grad_mag2, pad_y.clone()], dim=2)
            
            mask = (grad_mag1 > grad_mag2).float()
            return mask * feature1 + (1 - mask) * feature2
        else:
            # 默认使用平均融合
            return (feature1 + feature2) / 2

    def batch_fusion(self, ir_dir, vi_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        vi_files = sorted([f for f in os.listdir(vi_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f'找到 {len(ir_files)} 张红外图像, {len(vi_files)} 张可见光图像')
        
        if len(ir_files) != len(vi_files):
            print(f'[警告] 图像数量不匹配 ({len(ir_files)} vs {len(vi_files)})')
            
        processed_count = 0
        failed_count = 0
        
        print(f'开始批量融合（CBAM方案: {self.model.cbam_scheme}, 融合策略: {self.config.fusion_strategy}）...')
        
        for i, (ir_file, vi_file) in enumerate(tqdm(zip(ir_files, vi_files), total=len(ir_files), desc="融合进度")):
            ir_path = os.path.join(ir_dir, ir_file)
            vi_path = os.path.join(vi_dir, vi_file)
            
            base_name = os.path.splitext(ir_file)[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                success = self.run_single(ir_path, vi_path, output_path)
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f'\n[失败] 处理失败 {ir_file} 和 {vi_file}: {str(e)}')
        
        print(f'\n{"="*60}')
        print(f'批量融合完成！')
        print(f'成功处理: {processed_count}/{len(ir_files)} 对图像')
        print(f'失败数量: {failed_count}')
        print(f'成功率: {processed_count/len(ir_files)*100:.2f}%')
        print(f'输出目录: {output_dir}')
        print(f'CBAM方案: {self.model.cbam_scheme}')
        print(f'融合策略: {self.config.fusion_strategy}')
        print(f'{"="*60}')
        return processed_count, failed_count


def batch_fusion_main(config):
    """批量融合主函数"""
    fusion_model = BatchImageFusionOptimized(config)
    
    processed, failed = fusion_model.batch_fusion(config.ir_dir, config.vi_dir, config.output_dir)
    
    return processed, failed


def print_config_summary(config):
    """打印配置摘要 - 清晰显示当前使用的方案"""
    print("="*60)
    print("支持CBAM的批量图像融合 - 配置摘要")
    print("="*60)
    
    # 基础配置
    print("[基础配置]")
    print(f"  红外图像目录: {config.ir_dir}")
    print(f"  可见光图像目录: {config.vi_dir}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  模型权重: {config.model_weights}")
    
    # CBAM方案配置
    print("\n[CBAM注意力方案]")
    if config.cbam_scheme == 0:
        print("  方案0: 不使用CBAM注意力机制")
    elif config.cbam_scheme == 1:
        print("  方案1: DenseBlock输出后应用CBAM")
    elif config.cbam_scheme == 2:
        print("  方案2: 融合层输入前应用CBAM")
    print(f"  reduction_ratio: {config.reduction_ratio}")
    
    # 颜色感知配置
    print("\n[颜色感知配置]")
    if config.use_color_aware:
        print(f"  [启用] 颜色感知CBAM (解决泛黄问题)")
        print(f"  颜色保护权重: {config.color_preservation_weight}")
    else:
        print("  [禁用] 颜色感知CBAM")
    
    # CBAM消融实验配置
    print("\n[CBAM消融实验配置]")
    if config.cbam_scheme > 0:
        print(f"  通道注意力: {'启用' if config.use_channel_attention else '禁用'}")
        print(f"  空间注意力: {'启用' if config.use_spatial_attention else '禁用'}")
        
        # 构建注意力配置描述
        attention_components = []
        if config.use_channel_attention:
            attention_components.append("通道注意力")
        if config.use_spatial_attention:
            attention_components.append("空间注意力")
        
        if attention_components:
            attention_str = " + ".join(attention_components)
            print(f"  注意力配置: {attention_str}")
        else:
            print("  注意力配置: 无注意力（完全禁用）")
    else:
        print("  消融实验: 未启用（CBAM方案为0）")
    
    # 融合策略配置
    print("\n[融合策略配置]")
    print(f"  融合策略: {config.fusion_strategy}")
    
    # 策略描述
    strategy_descriptions = {
        'mean': '简单平均融合',
        'max': '最大值融合',
        'l1norm': 'L1范数融合',
        'adaptive_l1': '自适应L1融合',
        'gradient_based': '基于梯度融合',
        'enhanced_l1': '增强L1融合',
        'multi_scale': '多尺度融合',
        'gradient': '梯度引导融合',
        'hybrid': '混合融合（推荐）'
    }
    description = strategy_descriptions.get(config.fusion_strategy, '未知策略')
    print(f"  策略描述: {description}")
    
    # 混合融合权重配置（仅对hybrid策略显示）
    if config.fusion_strategy == 'hybrid':
        print(f"\n  [混合融合权重预设: {config.hybrid_weights_preset}]")
        
        preset_descriptions = {
            'balanced': '平衡质量与速度（默认推荐）',
            'quality': '高质量优先，适合精细图像',
            'detail': '细节增强，适合边缘丰富场景',
            'speed': '快速处理，减少计算量',
            'edge_enhanced': '边缘增强，适合医学影像',
            'structure_preserve': '结构保持，适合建筑/场景'
        }
        preset_desc = preset_descriptions.get(config.hybrid_weights_preset, '未知预设')
        print(f"  预设描述: {preset_desc}")
    
    # 其他配置
    print("\n[其他配置]")
    print(f"  灰度模式: {'启用' if config.gray else '禁用'}")
    print(f"  设备: {config.device}")
    print(f"  目标尺寸: {config.target_size}")
    
    print("="*60)


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置对象（直接传入命令行参数）
    config = FusionConfig(args)
    
    # 验证配置
    try:
        config.validate()
        print("[通过] 配置验证通过")
    except ValueError as e:
        print(f"[失败] 配置验证失败: {e}")
        exit(1)
    
    # 打印清晰的配置摘要
    print_config_summary(config)
    
    # 执行批量融合
    batch_fusion_main(config)