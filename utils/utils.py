import os
import numpy as np
import torch
import datetime

'''
/****************************************************/
获得学习率
/****************************************************/
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


'''
/****************************************************/
初始化模型权重
/****************************************************/
'''


def weights_init(model, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    model.apply(init_func)


'''
/****************************************************/
    运行程序时创建特定命名格式的文件夹
    以记录本次运行的相关日志和检查点信息
/****************************************************/
'''


def create_run_directory(args, base_dir='./runs'):
    """
    @desc：创建一个新的运行日志文件夹结构，包含logs和checkpoints子目录。
    @params：
    base_dir (str): 基础运行目录，默认为'./runs/train'
    @return：
    run_path (str): 新创建的此次运行的完整路径
    log_path (str): 子目录 logs 的完整路径
    checkpoints_path (str): 子目录 checkpoints 的完整路径
    """
    # 获取当前时间戳
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%m-%d_%H-%M')

    # 构建模型与策略标识符
    # 颜色模式
    tag = "Gray" if args.gray else "RGB"
    
    # CBAM方案
    cbam_scheme_map = {0: "noCBAM", 1: "CBAM1", 2: "CBAM2"}
    cbam_str = cbam_scheme_map.get(args.cbam_scheme, f"CBAM{args.cbam_scheme}")
    
    # 注意力配置
    attention_config = ""
    if args.cbam_scheme != 0:  # 仅当使用CBAM时显示注意力配置
        if args.use_channel_attention and args.use_spatial_attention:
            attention_config = "_full"
        elif args.use_channel_attention:
            attention_config = "_channel"
        elif args.use_spatial_attention:
            attention_config = "_spatial"
        else:
            attention_config = "_noAttn"
    
    # 梯度方向
    gradient_dir = f"_{args.gradient_direction}"
    
    # 构建此次运行的唯一标识符作为子目录名称
    run_identifier = f"{tag}_{cbam_str}{attention_config}{gradient_dir}_{time_str}"
    run_path = os.path.join(base_dir, run_identifier)

    # 定义并构建子目录路径
    # 子文件夹 logs 和 checkpoints
    checkpoints_path = os.path.join(run_path, "checkpoints")
    logs_name = f"logs_epoch={args.num_epochs}"
    logs_path = os.path.join(run_path, logs_name)

    # 创建所需的目录结构
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 打印配置信息
    print(f"[目录命名] 运行目录: {run_identifier}")
    print(f"[配置信息] 颜色模式: {tag}, CBAM方案: {cbam_str}, 注意力: {attention_config[1:] if attention_config else 'N/A'}")
    print(f"[配置信息] 梯度方向: {args.gradient_direction}")

    # return run_path, log_path, checkpoints_path
    return run_path, checkpoints_path, logs_path
