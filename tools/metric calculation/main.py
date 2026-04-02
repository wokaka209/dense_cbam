# Author: wokaka209
"""
图像质量评估(IQA)主程序
功能：
    1. 计算融合图像的各项质量指标
    2. 支持用户手动选择处理图像数量
    3. 输出统计结果到txt文件
    4. 提供可选的统计数量显示功能
"""

import cv2
import os
from metrics import *
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import pickle
import json
import signal
import sys

eval_funcs = {
    "AG": ag,
    # "CE": cross_entropy,
    # "EI": edge_intensity,
    "EN": entropy,
    "MI": mutinf,
    # "MSE": mse,
    # "PSNR": psnr,
    "SD": sd,
    # "SF": sf,
    "SSIM": ssim,
    "Qabf": qabf,
    # "Qcb": qcb,
    # "Qcv": qcv,
    # "VIF": vif,
}

@dataclass
class IQAConfig:
    """
    IQA配置参数类
    
    属性:
        show_count: 是否在输出文件中显示统计数量
        output_format: 输出文件格式（固定为txt）
        decimal_places: 数值精度（小数位数）
        enable_image_count_selection: 是否启用用户手动选择图像数量功能
        enable_resume: 是否启用断点续传功能
        save_interval: 保存间隔（每处理多少张图像保存一次进度）
    """
    show_count: bool = True
    output_format: str = "txt"
    decimal_places: int = 4
    enable_image_count_selection: bool = True
    enable_resume: bool = True
    save_interval: int = 5

def get_image_count_from_user(total_images: int) -> int:
    """
    获取用户输入的图像处理数量，支持输入验证和默认值
    
    参数:
        total_images: 可用图像总数
        
    返回:
        用户选择的图像数量（1到total_images之间的整数）
        
    功能说明:
        - 显示可用图像总数
        - 支持直接回车处理全部图像
        - 验证输入是否为有效整数
        - 验证数值范围是否合法
        - 支持Ctrl+C中断程序
    """
    if total_images == 0:
        return 0
    
    print(f"\n当前可用图像总数：{total_images}")
    print("请输入需要处理的图像数量（直接回车处理全部图像）")
    
    while True:
        try:
            user_input = input(f"处理图像数量 [1-{total_images}] 或回车跳过: ").strip()
            
            if user_input == "":
                print(f"将处理全部 {total_images} 张图像")
                return total_images
            
            count = int(user_input)
            
            if count <= 0:
                print(f"错误：数量必须大于0，请重新输入")
                continue
            
            if count > total_images:
                print(f"错误：数量不能超过 {total_images}，请重新输入")
                continue
            
            print(f"将处理前 {count} 张图像")
            return count
            
        except ValueError:
            print("错误：请输入有效的整数，请重新输入")
        except KeyboardInterrupt:
            print("\n\n用户取消操作，程序退出")
            exit(0)
        except Exception as e:
            print(f"错误：输入无效 ({e})，请重新输入")

def save_results_to_txt(
    results: Dict[str, List[float]],
    output_path: str,
    config: IQAConfig,
    total_images: int,
    model_name: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("图像质量评估 (IQA) 结果\n")
        f.write("=" * 60 + "\n\n")
        
        if config.show_count:
            f.write(f"总图像数量：{total_images}\n")
            f.write("=" * 60 + "\n\n")
        
        f.write("指标平均值统计：\n")
        f.write("-" * 60 + "\n")
        
        for metric_name, values in results.items():
            if values:
                # 只计算非None值的平均值
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    mean_val = sum(valid_values) / len(valid_values)
                    mean_val = float(mean_val) if hasattr(mean_val, 'item') else float(mean_val)
                    f.write(f"{metric_name:8s} : {mean_val:.{config.decimal_places}f}\n")
                else:
                    f.write(f"{metric_name:8s} : 所有值均为None\n")
            else:
                f.write(f"{metric_name:8s} : 无有效数据\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型是{model_name}\n")


def save_progress(
    results: Dict[str, List[float]],
    current_index: int,
    total_images: int,
    path_fusimgs: List[str],
    path_viimgs: List[str],
    path_irimgs: List[str],
    metrics: List[str],
    model_name: str,
    checkpoint_dir: str
) -> None:
    """保存计算进度到检查点文件"""
    checkpoint_data = {
        'results': results,
        'current_index': current_index,
        'total_images': total_images,
        'path_fusimgs': path_fusimgs,
        'path_viimgs': path_viimgs,
        'path_irimgs': path_irimgs,
        'metrics': metrics,
        'model_name': model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    meta_file = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.json")
    meta_data = {
        'current_index': current_index,
        'total_images': total_images,
        'model_name': model_name,
        'timestamp': checkpoint_data['timestamp'],
        'progress_percent': f"{(current_index / total_images * 100):.2f}%"
    }
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)


def load_progress(
    checkpoint_dir: str,
    model_name: str
) -> Optional[Dict]:
    """从检查点文件加载计算进度"""
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.pkl")
    
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        return checkpoint_data
    except Exception as e:
        print(f"警告：加载检查点文件失败：{e}")
        return None


def clear_checkpoint(
    checkpoint_dir: str,
    model_name: str
) -> None:
    """清除检查点文件"""
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.pkl")
    meta_file = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.json")
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists(meta_file):
        os.remove(meta_file)


def calculate_metrics(
    path_fusimgs: List[str],
    path_viimgs: List[str],
    path_irimgs: List[str],
    metrics: List[str],
    config: IQAConfig,
    model_name: str,
    checkpoint_dir: str,
    start_index: int = 0
) -> Dict[str, List[float]]:
    results = {key: [None] * len(path_fusimgs) for key in metrics}
    
    global interrupted
    interrupted = False
    
    def signal_handler(signum, frame):
        global interrupted
        interrupted = True
        print("\n\n检测到中断信号，正在保存进度...")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    pbar = tqdm(range(start_index, len(path_fusimgs)), desc="计算指标中", initial=start_index, total=len(path_fusimgs))
    for i in pbar:
        if interrupted:
            print("\n程序被中断，正在保存当前进度...")
            save_progress(
                results=results,
                current_index=i,
                total_images=len(path_fusimgs),
                path_fusimgs=path_fusimgs,
                path_viimgs=path_viimgs,
                path_irimgs=path_irimgs,
                metrics=metrics,
                model_name=model_name,
                checkpoint_dir=checkpoint_dir
            )
            print(f"进度已保存到检查点文件，下次可从第 {i+1} 张图像继续")
            raise KeyboardInterrupt("用户中断程序")
        
        img_fus = cv2.imread(path_fusimgs[i], 0)
        img_vi = cv2.imread(path_viimgs[i], 0)
        img_ir = cv2.imread(path_irimgs[i], 0)
        
        if img_fus is None or img_vi is None or img_ir is None:
            print(f"警告：无法读取第 {i+1} 张图像，跳过")
            continue
        
        max_h, max_w = img_fus.shape[:2]
        img_vi = cv2.resize(img_vi, (max_w, max_h))
        img_ir = cv2.resize(img_ir, (max_w, max_h))
        
        for metric in metrics:
            try:
                # 根据函数签名决定传递的参数数量
                import inspect
                func = eval_funcs[metric]
                sig = inspect.signature(func)
                param_count = len(sig.parameters)
                
                if param_count == 1:
                    results[metric][i] = func(img_fus)
                    print(f"DEBUG: {metric} 计算成功 (1参数)")
                elif param_count == 3:
                    results[metric][i] = func(img_fus, img_vi, img_ir)
                    print(f"DEBUG: {metric} 计算成功 (3参数)")
                else:
                    print(f"警告：指标 {metric} 的参数数量 ({param_count}) 不支持")
                    results[metric][i] = None
                    
            except Exception as e:
                print(f"计算指标 {metric} 时出错：{e}")
                results[metric][i] = None
        
        pbar.write(f"\n图像 {i+1}/{len(path_fusimgs)}: {os.path.basename(path_fusimgs[i])}")
        pbar.write("-" * 80)
        for metric in metrics:
            val = results[metric][i]
            if val is not None:
                val = float(val) if hasattr(val, 'item') else float(val)
                pbar.write(f"  {metric:8s}: {val:.4f}")
            else:
                pbar.write(f"  {metric:8s}: N/A")
        
        if config.enable_resume and (i + 1) % config.save_interval == 0:
            save_progress(
                results=results,
                current_index=i + 1,
                total_images=len(path_fusimgs),
                path_fusimgs=path_fusimgs,
                path_viimgs=path_viimgs,
                path_irimgs=path_irimgs,
                metrics=metrics,
                model_name=model_name,
                checkpoint_dir=checkpoint_dir
            )
            pbar.write(f"进度已保存 ({i+1}/{len(path_fusimgs)})")
    
    return results


def load_image_paths(
    fus_root: str,
    vi_root: str,
    ir_root: str
) -> tuple:
    path_fusimgs, path_irimgs, path_viimgs = [], [], []
    
    if not os.path.exists(fus_root):
        raise FileNotFoundError(f"融合图像目录不存在：{fus_root}")
    
    img_list = os.listdir(fus_root)
    for img in img_list:
        img = img.strip()
        if not (img.endswith('.jpg') or img.endswith('.png')):
            continue
        
        path_fusimgs.append(os.path.join(fus_root, img))
        path_viimgs.append(os.path.join(vi_root, img))
        path_irimgs.append(os.path.join(ir_root, img))
    
    if len(path_viimgs) != len(path_irimgs):
        print("警告：可见光图像与红外图像数量不一致！")
    
    return path_fusimgs, path_viimgs, path_irimgs


def main():
    """
    主函数：执行完整的图像质量评估流程
    
    流程步骤:
        1. 初始化配置参数
        2. 检查是否有保存的进度（断点续传）
        3. 加载图像路径列表
        4. 用户选择处理图像数量（可选功能）
        5. 计算各项质量指标
        6. 保存统计结果到txt文件
        7. 清除检查点文件
    """
    config = IQAConfig(
        show_count=True,
        output_format="txt",
        decimal_places=4,
        enable_image_count_selection=True,
        enable_resume=True,
        save_interval=1000
    )
    
    model_name = "dense_cbam2_epoch30"
    fus_root = "data_result/batch_fusion_optimized_cbam2-epoch30"
    vi_root = "E:/whx_Graduation project/baseline_project/dataset/vi"
    ir_root = "E:/whx_Graduation project/baseline_project/dataset/ir"
    output_root = os.path.abspath("./tools/metric calculation/iqa_results")
    checkpoint_dir = os.path.abspath("./tools/metric calculation/checkpoints")
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_index = 0
    
    if config.enable_resume:
        checkpoint_data = load_progress(checkpoint_dir, model_name)
        if checkpoint_data is not None:
            print("\n" + "=" * 60)
            print("检测到之前的计算进度")
            print("=" * 60)
            print(f"模型名称：{checkpoint_data['model_name']}")
            print(f"上次保存时间：{checkpoint_data['timestamp']}")
            print(f"已处理图像：{checkpoint_data['current_index']}/{checkpoint_data['total_images']}")
            print(f"进度：{checkpoint_data['current_index'] / checkpoint_data['total_images'] * 100:.2f}%")
            print("=" * 60)
            
            while True:
                choice = input("\n是否从上次中断的位置继续计算？[Y/n]: ").strip().lower()
                if choice == '' or choice == 'y' or choice == 'yes':
                    start_index = checkpoint_data['current_index']
                    path_fusimgs = checkpoint_data['path_fusimgs']
                    path_viimgs = checkpoint_data['path_viimgs']
                    path_irimgs = checkpoint_data['path_irimgs']
                    metrics = checkpoint_data['metrics']
                    print(f"\n将从第 {start_index + 1} 张图像继续计算")
                    break
                elif choice == 'n' or choice == 'no':
                    print("\n将重新开始计算")
                    break
                else:
                    print("无效输入，请输入 Y 或 N")
    
    if start_index == 0:
        print("正在加载图像路径...")
        path_fusimgs, path_viimgs, path_irimgs = load_image_paths(fus_root, vi_root, ir_root)
        
        if not path_fusimgs:
            print("错误：未找到任何图像文件！")
            return
        
        total_available = len(path_fusimgs)
        
        if config.enable_image_count_selection:
            selected_count = get_image_count_from_user(total_available)
            path_fusimgs = path_fusimgs[:selected_count]
            path_viimgs = path_viimgs[:selected_count]
            path_irimgs = path_irimgs[:selected_count]
        else:
            selected_count = total_available
        
        metrics = list(eval_funcs.keys())
    else:
        selected_count = len(path_fusimgs)
    
    print(f"\n将计算以下指标：{', '.join(metrics)}")
    
    try:
        results = calculate_metrics(
            path_fusimgs, 
            path_viimgs, 
            path_irimgs, 
            metrics,
            config,
            model_name,
            checkpoint_dir,
            start_index
        )
        
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        txt_file = os.path.join(output_root, f"result_{timestamp}_{model_name}.txt")
        
        save_results_to_txt(results, txt_file, config, selected_count, model_name)
        print(f"\n结果已保存至：{txt_file}")
        
        if config.enable_resume:
            clear_checkpoint(checkpoint_dir, model_name)
            print("检查点文件已清除")
        
    except KeyboardInterrupt:
        print("\n程序已中断，进度已保存。下次启动时可以选择继续计算。")
        sys.exit(0)


if __name__ == "__main__":
    main()
