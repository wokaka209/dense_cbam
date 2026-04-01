# -*- coding: utf-8 -*-
"""
@file name:batch_fusion_optimized.py
@desc: 优化版批量融合脚本 - 使用高级融合策略提升EN、AG、MI、Qabf指标
@Writer: wokaka209
@Date: 2026-03-13
"""
import os
import torch
from torchvision.utils import save_image
from models import fuse_model
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
import argparse
from fusion_strategy.advanced_fusion_optimized import AdvancedFusionStrategyOptimized


class BatchImageFusionOptimized:
    def __init__(self, config):
        self.config = config
        self.target_size = (768, 1024)
        self.fusion_strategy_obj = AdvancedFusionStrategyOptimized()
        self.load_model()

    def load_model(self):
        in_channel = 1 if self.config.gray else 3
        out_channel = 1 if self.config.gray else 3
        self.model = fuse_model(self.config.model_name,
                                input_nc=in_channel,
                                output_nc=out_channel)
        self.model = self.model.to(self.config.device)
        checkpoint = torch.load(self.config.resume_path,
                                map_location=self.config.device, weights_only=False)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f'✓ 模型加载成功: {self.config.resume_path}')
        print(f'✓ 使用融合策略: {self.config.fusion_strategy}')

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

    def run_single(self, ir_path, vi_path, output_path, fusion_strategy="hybrid"):
        self.model.eval()
        with torch.no_grad():
            ir_image, original_size = self.preprocess_image(ir_path)
            vi_image, _ = self.preprocess_image(vi_path)
            
            ir_image = ir_image.to(self.config.device)
            vi_image = vi_image.to(self.config.device)

            ir_features = self.model.encoder(ir_image)
            vi_features = self.model.encoder(vi_image)

            fused_features = self.fusion_strategy(ir_features, vi_features, fusion_strategy)

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
    
    def fusion_strategy(self, feature1: torch.Tensor, feature2: torch.Tensor, strategy="hybrid") -> torch.Tensor:
        """高级融合策略 - 调用fusion_strategy.advanced_fusion模块"""
        if strategy == "enhanced_l1":
            return self.fusion_strategy_obj.enhanced_adaptive_l1(feature1, feature2)
        elif strategy == "multi_scale":
            return self.fusion_strategy_obj.multi_scale_fusion(feature1, feature2)
        elif strategy == "gradient":
            return self.fusion_strategy_obj.gradient_guided_fusion(feature1, feature2)
        elif strategy == "hybrid":
            return self.fusion_strategy_obj.hybrid_fusion(feature1, feature2)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def batch_fusion(self, ir_dir, vi_dir, output_dir, fusion_strategy='hybrid'):
        os.makedirs(output_dir, exist_ok=True)
        
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        vi_files = sorted([f for f in os.listdir(vi_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f'找到 {len(ir_files)} 张红外图像, {len(vi_files)} 张可见光图像')
        
        if len(ir_files) != len(vi_files):
            print(f'⚠️  警告: 图像数量不匹配 ({len(ir_files)} vs {len(vi_files)})')
            
        processed_count = 0
        failed_count = 0
        
        print(f'开始批量融合（策略: {fusion_strategy}）...')
        
        for i, (ir_file, vi_file) in enumerate(tqdm(zip(ir_files, vi_files), total=len(ir_files), desc="融合进度")):
            ir_path = os.path.join(ir_dir, ir_file)
            vi_path = os.path.join(vi_dir, vi_file)
            
            base_name = os.path.splitext(ir_file)[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                success = self.run_single(ir_path, vi_path, output_path, fusion_strategy)
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f'\n❌ 处理失败 {ir_file} 和 {vi_file}: {str(e)}')
        
        print(f'\n{"="*60}')
        print(f'批量融合完成！')
        print(f'成功处理: {processed_count}/{len(ir_files)} 对图像')
        print(f'失败数量: {failed_count}')
        print(f'成功率: {processed_count/len(ir_files)*100:.2f}%')
        print(f'输出目录: {output_dir}')
        print(f'融合策略: {fusion_strategy}')
        print(f'{"="*60}')
        return processed_count, failed_count


def batch_fusion_main(ir_dir, vi_dir, output_dir, model_weights, fusion_strategy='hybrid'):
    class Config:
        def __init__(self):
            self.ir_path = ir_dir
            self.vi_path = vi_dir
            self.resume_path = model_weights
            self.fusion_strategy = fusion_strategy
            self.gray = False
            self.model_name = 'DenseFuse'
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = Config()
    fusion_model = BatchImageFusionOptimized(config)
    
    processed, failed = fusion_model.batch_fusion(ir_dir, vi_dir, output_dir, fusion_strategy)
    
    return processed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='优化版批量图像融合')
    parser.add_argument('--ir_dir', type=str, 
                        default='E:/whx_Graduation project/baseline_project/dataset/ir', 
                        help='红外图像目录')
    parser.add_argument('--vi_dir', type=str, 
                        default='E:/whx_Graduation project/baseline_project/dataset/vi', 
                        help='可见光图像目录')
    parser.add_argument('--output_dir', type=str, 
                        default='data_result/batch_fusion_optimized_cbam1-epoch30', 
                        help='输出目录')
    parser.add_argument('--model_weights', type=str, 
                        default='runs/train_04-01_14-47/checkpoints/epoch023-loss4.941.pth', 
                        help='模型权重路径')
    parser.add_argument('--fusion_strategy', type=str, 
                        default='hybrid', 
                        choices=['enhanced_l1', 'multi_scale', 'gradient', 'hybrid'],
                        help='融合策略（推荐使用hybrid）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("优化版批量图像融合")
    print("="*60)
    print(f"红外图像目录: {args.ir_dir}")
    print(f"可见光图像目录: {args.vi_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型权重: {args.model_weights}")
    print(f"融合策略: {args.fusion_strategy}")
    print("="*60)
    
    batch_fusion_main(args.ir_dir, args.vi_dir, args.output_dir, 
                    args.model_weights, args.fusion_strategy)