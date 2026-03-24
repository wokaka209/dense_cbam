# Author: wokaka209
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_entropy(image_path):
    """
    计算图像的信息熵(EN)
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        float: 信息熵值
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # 计算灰度直方图
    hist, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    
    # 归一化直方图
    hist = hist / hist.sum()
    
    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return entropy


def calculate_psnr(img1_path, img2_path):
    """
    计算两幅图像之间的峰值信噪比(PSNR)
    
    Args:
        img1_path: 第一幅图像路径
        img2_path: 第二幅图像路径
        
    Returns:
        float: PSNR值
    """
    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_ssim(img1_path, img2_path):
    """
    计算两幅图像之间的结构相似性(SSIM)
    
    Args:
        img1_path: 第一幅图像路径
        img2_path: 第二幅图像路径
        
    Returns:
        float: SSIM值
    """
    from skimage.metrics import structural_similarity as ssim
    
    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))
    
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    
    return ssim_value


def evaluate_fusion_results(fused_dir, ir_dir=None, vi_dir=None, metrics=['EN', 'PSNR', 'SSIM']):
    """
    评估融合结果
    
    Args:
        fused_dir: 融合结果图像目录
        ir_dir: 红外图像目录（可选，用于计算PSNR和SSIM）
        vi_dir: 可见光图像目录（可选，用于计算PSNR和SSIM）
        metrics: 要计算的指标列表
        
    Returns:
        dict: 包含各项指标的字典
    """
    # 获取融合图像列表
    fused_images = sorted([f for f in os.listdir(fused_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results = {
        'EN': [],
        'PSNR_IR': [],
        'PSNR_VI': [],
        'SSIM_IR': [],
        'SSIM_VI': []
    }
    
    print(f"开始评估 {len(fused_images)} 幅融合图像...")
    
    for fused_img_name in tqdm(fused_images, desc="评估进度"):
        fused_img_path = os.path.join(fused_dir, fused_img_name)
        
        # 计算EN
        if 'EN' in metrics:
            en_value = calculate_entropy(fused_img_path)
            results['EN'].append(en_value)
        
        # 计算PSNR和SSIM（如果提供了参考图像）
        if ir_dir and vi_dir:
            base_name = os.path.splitext(fused_img_name)[0]
            
            # 尝试找到对应的红外和可见光图像
            ir_img_path = None
            vi_img_path = None
            
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_ir = os.path.join(ir_dir, base_name + ext)
                potential_vi = os.path.join(vi_dir, base_name + ext)
                
                if os.path.exists(potential_ir):
                    ir_img_path = potential_ir
                if os.path.exists(potential_vi):
                    vi_img_path = potential_vi
            
            if ir_img_path and 'PSNR' in metrics:
                psnr_ir = calculate_psnr(fused_img_path, ir_img_path)
                results['PSNR_IR'].append(psnr_ir)
            
            if vi_img_path and 'PSNR' in metrics:
                psnr_vi = calculate_psnr(fused_img_path, vi_img_path)
                results['PSNR_VI'].append(psnr_vi)
            
            if ir_img_path and 'SSIM' in metrics:
                ssim_ir = calculate_ssim(fused_img_path, ir_img_path)
                results['SSIM_IR'].append(ssim_ir)
            
            if vi_img_path and 'SSIM' in metrics:
                ssim_vi = calculate_ssim(fused_img_path, vi_img_path)
                results['SSIM_VI'].append(ssim_vi)
    
    # 计算平均值
    summary = {}
    for key, values in results.items():
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
    
    return summary


def print_evaluation_summary(summary):
    """
    打印评估结果摘要
    
    Args:
        summary: evaluate_fusion_results返回的结果字典
    """
    print("\n" + "="*60)
    print("融合质量评估结果")
    print("="*60)
    
    if 'EN_mean' in summary:
        print(f"\n信息熵(EN):")
        print(f"  平均值: {summary['EN_mean']:.4f}")
        print(f"  标准差: {summary['EN_std']:.4f}")
        print(f"  最小值: {summary['EN_min']:.4f}")
        print(f"  最大值: {summary['EN_max']:.4f}")
    
    if 'PSNR_IR_mean' in summary:
        print(f"\n峰值信噪比(PSNR) - 相对红外图像:")
        print(f"  平均值: {summary['PSNR_IR_mean']:.4f} dB")
        print(f"  标准差: {summary['PSNR_IR_std']:.4f} dB")
        print(f"  最小值: {summary['PSNR_IR_min']:.4f} dB")
        print(f"  最大值: {summary['PSNR_IR_max']:.4f} dB")
    
    if 'PSNR_VI_mean' in summary:
        print(f"\n峰值信噪比(PSNR) - 相对可见光图像:")
        print(f"  平均值: {summary['PSNR_VI_mean']:.4f} dB")
        print(f"  标准差: {summary['PSNR_VI_std']:.4f} dB")
        print(f"  最小值: {summary['PSNR_VI_min']:.4f} dB")
        print(f"  最大值: {summary['PSNR_VI_max']:.4f} dB")
    
    if 'SSIM_IR_mean' in summary:
        print(f"\n结构相似性(SSIM) - 相对红外图像:")
        print(f"  平均值: {summary['SSIM_IR_mean']:.4f}")
        print(f"  标准差: {summary['SSIM_IR_std']:.4f}")
        print(f"  最小值: {summary['SSIM_IR_min']:.4f}")
        print(f"  最大值: {summary['SSIM_IR_max']:.4f}")
    
    if 'SSIM_VI_mean' in summary:
        print(f"\n结构相似性(SSIM) - 相对可见光图像:")
        print(f"  平均值: {summary['SSIM_VI_mean']:.4f}")
        print(f"  标准差: {summary['SSIM_VI_std']:.4f}")
        print(f"  最小值: {summary['SSIM_VI_min']:.4f}")
        print(f"  最大值: {summary['SSIM_VI_max']:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估融合图像质量')
    parser.add_argument('--fused_dir', type=str, required=True, 
                        help='融合结果图像目录')
    parser.add_argument('--ir_dir', type=str, default=None, 
                        help='红外图像目录（可选，用于计算PSNR和SSIM）')
    parser.add_argument('--vi_dir', type=str, default=None, 
                        help='可见光图像目录（可选，用于计算PSNR和SSIM）')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['EN', 'PSNR', 'SSIM'],
                        choices=['EN', 'PSNR', 'SSIM'],
                        help='要计算的指标')
    
    args = parser.parse_args()
    
    # 评估融合结果
    summary = evaluate_fusion_results(
        fused_dir=args.fused_dir,
        ir_dir=args.ir_dir,
        vi_dir=args.vi_dir,
        metrics=args.metrics
    )
    
    # 打印结果
    print_evaluation_summary(summary)