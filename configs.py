'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-13 12:04:34
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-24 15:37:37
FilePath: \my_densefuse_advantive\configs.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-
"""
@file name:configs.py
@desc: 模型参数
@Writer: Cat2eacher
@Date: 2025/01/03
"""

import argparse
from utils.util_device import device_on
'''
/****************************************************/
    模型参数
/****************************************************/
'''


def set_args():
    # 创建ArgumentParser()对象
    parser = argparse.ArgumentParser(description="模型参数设置")

    # 调用add_argument()方法添加参数
    # parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    # parser.add_argument('--name', default="Cat2eacher", help="Coder Name")
    # 数据集相关参数 - M3FD数据集配置
    parser.add_argument('--image_path', default=r'../dataset/COCO_train2014', type=str, help='数据集路径 ')
    parser.add_argument('--dataset_type', default='ir_vi', type=str, choices=['coco', 'ir_vi'], help='数据集类型: coco用于通用图像, ir_vi用于M3FD红外可见光图像融合')
    parser.add_argument('--ir_path', default='E:/whx_Graduation project/baseline_project/dataset/ir', type=str, help='M3FD红外图像路径 (4200张RGB图像, 1024x768)')
    parser.add_argument('--vi_path', default='E:/whx_Graduation project/baseline_project/dataset/vi', type=str, help='M3FD可见光图像路径 (4200张RGB图像, 1024x768)')
    parser.add_argument('--gray', default=False, type=bool, help='是否使用灰度模式 (M3FD为RGB图像, 设为False)')
    parser.add_argument('--train_num', default=4200, type=int, help='M3FD训练图像对数量 (默认4200对红外-可见光图像)')
    # 训练相关参数 - M3FD数据集优化配置
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for M3FD training (default=16)')
    parser.add_argument('--num_epochs', type=int, default=80, help='M3FD训练轮数 (建议80 epochs以获得更好的融合效果)')
    parser.add_argument('--lr', type=float, default=2e-4, help='M3FD学习率 (优化版建议2e-4)')
    parser.add_argument('--resume_path', default='', type=str, help='导入已训练好的模型路径')
    parser.add_argument('--fusion_strategy', default='adaptive_l1', type=str, 
                        choices=['mean', 'max', 'l1norm', 'adaptive_l1', 'gradient_based'],
                        help='融合策略: mean, max, l1norm, adaptive_l1, gradient_based')
    parser.add_argument('--num_workers', type=int, default=8, help='载入M3FD数据集所调用的cpu线程数 (建议4-8)')
    # M3FD特定优化参数
    parser.add_argument('--use_attention', action='store_true', default=True, help='M3FD训练时使用注意力机制增强融合效果')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True, help='M3FD训练时使用混合精度加速')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='M3FD学习率预热epoch数')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()

    if args.output:
        print("=" * 50)
        print("M3FD数据集配置信息")
        print("=" * 50)
        print("----------数据集相关参数----------")
        print(f'dataset_type: {args.dataset_type}')
        if args.dataset_type == 'ir_vi':
            print(f'  ├─ ir_path: {args.ir_path}')
            print(f'  ├─ vi_path: {args.vi_path}')
            print(f'  ├─ image_count: {args.train_num} pairs')
            print(f'  └─ color_mode: {"RGB" if not args.gray else "Gray"}')
        else:
            print(f'  ├─ image_path: {args.image_path}')
            print(f'  └─ train_num: {args.train_num}')

        print("----------训练相关参数----------")
        print(f'  ├─ device: {args.device}')
        print(f'  ├─ batch_size: {args.batch_size}')
        print(f'  ├─ num_epochs: {args.num_epochs}')
        print(f'  ├─ num_workers: {args.num_workers}')
        print(f'  ├─ learning_rate: {args.lr}')
        print(f'  └─ resume_path: {args.resume_path if args.resume_path else "None"}')
        
        if args.dataset_type == 'ir_vi':
            print("----------M3FD优化参数----------")
            print(f'  ├─ use_attention: {args.use_attention}')
            print(f'  ├─ use_mixed_precision: {args.use_mixed_precision}')
            print(f'  ├─ warmup_epochs: {args.warmup_epochs}')
            print(f'  └─ fusion_strategy: {args.fusion_strategy}')
        print("=" * 50)
    return args