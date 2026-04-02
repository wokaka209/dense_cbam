# -*- coding: utf-8 -*-
"""
@file name: loss_analyzer.py
@desc: TensorBoard事件文件解析与Loss曲线绘制工具
@Author: wokaka209
@Date: 2026-04-01
"""

import os
import glob
from typing import List, Tuple, Dict, Optional


def ensure_directory(path: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def read_tensorboard_events(event_file: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    读取TensorBoard事件文件并提取loss数据
    
    Args:
        event_file: TensorBoard事件文件路径
        
    Returns:
        包含各tag数据的字典，格式为 {tag: [(step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("需要安装tensorboard库: pip install tensorboard")
    
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
        }
    )
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    result = {}
    
    for tag in available_tags:
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]
    
    return result


def read_all_loss_data(event_file_path: str) -> Tuple[List[int], List[float], List[str]]:
    """
    读取事件文件中的所有loss数据
    
    Args:
        event_file_path: TensorBoard事件文件路径
        
    Returns:
        (steps, values, tags) 元组
    """
    all_data = read_tensorboard_events(event_file_path)
    
    all_steps = []
    all_values = []
    all_tags = []
    
    for tag, data in all_data.items():
        if 'loss' in tag.lower() or 'Loss' in tag:
            for step, value in data:
                all_steps.append(step)
                all_values.append(value)
                all_tags.append(tag)
    
    if not all_steps:
        for tag, data in all_data.items():
            for step, value in data:
                all_steps.append(step)
                all_values.append(value)
                all_tags.append(tag)
    
    return all_steps, all_values, all_tags


def configure_matplotlib() -> None:
    """
    配置matplotlib的字体设置，确保中文显示正常
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (12, 6)


def plot_loss_curve(
    steps: List[int],
    values: List[float],
    tags: List[str],
    title: str = "训练Loss曲线",
    xlabel: str = "训练步数 (Epoch)",
    ylabel: str = "Loss值",
    save_path: Optional[str] = None,
    figure_size: Tuple[float, float] = (12, 6)
) -> None:
    """
    绘制loss变化曲线
    
    Args:
        steps: 训练步数列表
        values: Loss值列表
        tags: 数据标签列表
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径，若为None则不保存
        figure_size: 图形尺寸 (宽, 高)
    """
    import matplotlib.pyplot as plt
    
    configure_matplotlib()
    
    plt.figure(figsize=figure_size)
    
    if len(set(tags)) == 1:
        plt.plot(steps, values, marker='o', markersize=2, linewidth=2.5, label=tags[0])
    else:
        unique_tags = list(set(tags))
        for tag in unique_tags:
            tag_steps = [steps[i] for i in range(len(steps)) if tags[i] == tag]
            tag_values = [values[i] for i in range(len(values)) if tags[i] == tag]
            plt.plot(tag_steps, tag_values, marker='o', markersize=2, linewidth=2.5, label=tag)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.close()


def plot_multiple_loss_curves(
    loss_data: Dict[str, List[Tuple[int, float]]],
    title: str = "训练Loss曲线对比",
    xlabel: str = "训练步数 (Epoch)",
    ylabel: str = "Loss值",
    save_path: Optional[str] = None,
    figure_size: Tuple[float, float] = (12, 6)
) -> None:
    """
    绘制多条loss曲线
    
    Args:
        loss_data: 字典，key为曲线名称，value为 [(epoch, value), ...] 格式的数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        figure_size: 图形尺寸
    """
    import matplotlib.pyplot as plt
    
    configure_matplotlib()
    
    plt.figure(figsize=figure_size)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (tag, data) in enumerate(loss_data.items()):
        if not data:
            continue
        steps = [d[0] for d in data]
        values = [d[1] for d in data]
        color = colors[idx % len(colors)]
        plt.plot(steps, values, marker='o', markersize=2, linewidth=2.5, 
                 label=tag, color=color)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12, loc='best', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.close()


def save_image(
    loss_data: Dict[str, List[Tuple[int, float]]],
    output_dir: str,
    base_name: str = "loss_curve.png"
) -> str:
    """
    将loss曲线保存为图像文件
    
    Args:
        loss_data: Loss数据字典
        output_dir: 输出目录
        base_name: 文件名
        
    Returns:
        保存的文件路径
    """
    ensure_directory(output_dir)
    save_path = os.path.join(output_dir, base_name)
    plot_multiple_loss_curves(loss_data, save_path=save_path)
    return save_path


def analyze_event_file(
    event_file: str,
    output_dir: str,
    output_name: str = "loss_curve.png"
) -> Dict[str, List[Tuple[int, float]]]:
    """
    分析TensorBoard事件文件并生成loss曲线图
    
    Args:
        event_file: TensorBoard事件文件路径
        output_dir: 输出目录
        output_name: 输出文件名
        
    Returns:
        提取的loss数据字典
    """
    print(f"正在读取事件文件: {event_file}")
    loss_data = read_tensorboard_events(event_file)
    
    # 过滤数据，仅保留loss相关曲线
    filtered_loss_data = {}
    for tag, data in loss_data.items():
        if 'loss' in tag.lower() and 'learning_rate' not in tag.lower() and 'weight' not in tag.lower():
            filtered_loss_data[tag] = data
    
    # 如果没有找到loss数据，使用所有数据
    if not filtered_loss_data:
        filtered_loss_data = loss_data
    
    print(f"发现 {len(loss_data)} 个标量标签，过滤后保留 {len(filtered_loss_data)} 个loss相关标签")
    for tag in filtered_loss_data.keys():
        print(f"  - {tag}: {len(filtered_loss_data[tag])} 个数据点")
    
    save_path = os.path.join(output_dir, output_name)
    plot_multiple_loss_curves(filtered_loss_data, save_path=save_path)
    
    return filtered_loss_data


def main():
    """
    主函数 - 读取TensorBoard事件文件并绘制loss曲线
    """
    # 使用跨平台路径处理
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    
    # 事件文件路径（可根据需要修改）
    event_file = os.path.join(project_root, "runs", "train_04-01_14-47", "logs_RGB_epoch=80", "events.out.tfevents.1775026052.coolkey.8864.0")
    
    # 确保输出目录存在
    output_dir = os.path.join(base_dir, "image")
    ensure_directory(output_dir)
    
    # 生成基于事件文件名的图片名称
    event_filename = os.path.basename(event_file)
    timestamp = event_filename.split('.')[2] if '.' in event_filename else "unknown"
    output_name = f"loss_curve_{timestamp}.png"
    
    print(f"项目根目录: {project_root}")
    print(f"事件文件: {event_file}")
    print(f"输出目录: {output_dir}")
    print(f"输出文件名: {output_name}")
    
    # 验证事件文件是否存在
    if not os.path.exists(event_file):
        print(f"错误：事件文件不存在: {event_file}")
        print("请检查文件路径是否正确")
        return
    
    analyze_event_file(event_file, output_dir, output_name)


if __name__ == "__main__":
    main()