import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # 根据输入图像自动确定像素值范围（默认为0-255，也可以是其他范围，例如sigmoid激活函数输出的0-1或tanh激活函数输出的-1到1）
    if val_range is None:
        # 自动检测图像的最大最小值来确定范围
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val  # 计算范围差值
    else:
        L = val_range  # 若已知像素值范围则直接使用

    padd = 0  # 默认不进行额外填充
    (_, channel, height, width) = img1.size()  # 获取图像尺寸信息
    # 如果未提供预定义的窗口（高斯核），则根据输入图像的实际尺寸生成一个合适的窗口
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 使用高斯窗口计算图像的均值
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # 计算均值的平方、两图均值的乘积及其平方
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 计算图像方差及协方差（需减去各自的均值平方以消除均值的影响）
    # D(X)=E(X^INF_images)-[E(X)]^INF_images
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # COV(X,Y)=E(XY)-E(X)E(Y)
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    # 设置用于稳定比值的常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算对比敏感度 (contrast sensitivity)
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    # 计算 SSIM 映射（逐像素的 SSIM 值）
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # 根据参数决定返回的是所有像素的平均 SSIM 值还是整个映射
    if size_average:
        ret = ssim_map.mean()  # 计算整个 SSIM 映射的平均值
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 分别计算每幅图像各维度的平均值

    if full:  # 根据 full 参数决定是否返回完整的对比敏感度
        return ret, cs  # 返回 SSIM 值和对比敏感度
    return ret  # 只返回 SSIM 值


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume vis channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class GradientLoss(torch.nn.Module):
    """梯度损失函数 - 基于Sobel算子计算图像边缘信息，用于提升MI指标"""
    
    def __init__(self):
        super(GradientLoss, self).__init__()
        
        # Sobel算子 - X方向梯度
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Sobel算子 - Y方向梯度
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        """
        计算预测图像和目标图像之间的梯度差异损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            gradient_loss: 梯度损失值
        """
        # 获取设备和通道数
        device = pred.device
        num_channels = pred.size(1)
        
        # 将Sobel算子移动到相同设备，并扩展到与通道数匹配
        # 使用 depthwise convolution：每个通道使用相同的 Sobel 核
        sobel_x = self.sobel_x.to(device).expand(num_channels, 1, 3, 3)
        sobel_y = self.sobel_y.to(device).expand(num_channels, 1, 3, 3)
        
        # 计算预测图像的梯度 - depthwise convolution
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=num_channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=num_channels)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        
        # 计算目标图像的梯度
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=num_channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=num_channels)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
        
        # 计算L1损失
        loss = F.l1_loss(pred_grad, target_grad)
        
        return loss


class MultiScaleGradientLoss(torch.nn.Module):
    """多尺度梯度损失函数 - 在多个尺度上计算梯度差异"""
    
    def __init__(self, scales=[1, 2, 4]):
        super(MultiScaleGradientLoss, self).__init__()
        self.scales = scales
        self.gradient_loss = GradientLoss()
    
    def forward(self, pred, target):
        """
        计算多尺度梯度损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            multi_scale_loss: 多尺度梯度损失
        """
        total_loss = 0.0
        weight_sum = 0.0
        
        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                target_scaled = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            
            # 计算该尺度下的梯度损失
            scale_loss = self.gradient_loss(pred_scaled, target_scaled)
            total_loss += scale_loss
            weight_sum += 1.0
        
        return total_loss / weight_sum


def gradient_loss(pred, target):
    """
    函数式接口的梯度损失计算
    
    用于直接集成到现有的训练流程中
    
    Args:
        pred: 预测图像
        target: 目标图像
    
    Returns:
        gradient_loss_value: 梯度损失值
    """
    device = pred.device
    num_channels = pred.size(1)
    
    # Sobel算子 - 对每个通道使用相同的卷积核
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    # 使用深度可分离卷积：groups=num_channels
    # 每个通道使用相同的 Sobel 核，但分别计算
    sobel_x_expanded = sobel_x.expand(num_channels, 1, 3, 3)
    sobel_y_expanded = sobel_y.expand(num_channels, 1, 3, 3)
    
    # 计算梯度 - depthwise convolution
    pred_grad_x = F.conv2d(pred, sobel_x_expanded, padding=1, groups=num_channels)
    pred_grad_y = F.conv2d(pred, sobel_y_expanded, padding=1, groups=num_channels)
    pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
    
    target_grad_x = F.conv2d(target, sobel_x_expanded, padding=1, groups=num_channels)
    target_grad_y = F.conv2d(target, sobel_y_expanded, padding=1, groups=num_channels)
    target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
    
    # L1损失
    return F.l1_loss(pred_grad, target_grad)


def multi_scale_gradient_loss(pred, target, scales=[1, 2, 4]):
    """
    函数式接口的多尺度梯度损失计算
    
    Args:
        pred: 预测图像
        target: 目标图像
        scales: 尺度列表
    
    Returns:
        multi_scale_loss_value: 多尺度梯度损失值
    """
    total_loss = 0.0
    
    for scale in scales:
        if scale == 1:
            pred_scaled = pred
            target_scaled = target
        else:
            pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_scaled = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        
        total_loss += gradient_loss(pred_scaled, target_scaled)
    
    return total_loss / len(scales)


class TVLoss(torch.nn.Module):
    """Total Variation (TV) 损失函数 - 用于保持图像平滑性，减少噪声同时保留边缘"""
    
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, x):
        """
        计算Total Variation损失
        
        TVLoss = Σ|x(i+1,j) - x(i,j)| + |x(i,j+1) - x(i,j)|
        
        Args:
            x: 输入图像张量 [B, C, H, W]
        
        Returns:
            tv_loss: TV损失值
        """
        batch_size, channels, height, width = x.size()
        
        # 计算水平方向的差异 (与右侧像素)
        diff_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # 计算垂直方向的差异 (与下方像素)
        diff_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # 求和并平均
        tv_loss = (diff_x.sum() + diff_y.sum()) / (batch_size * channels)
        
        return tv_loss


def tv_loss(x):
    """
    函数式接口的TV损失计算
    
    Args:
        x: 输入图像张量 [B, C, H, W]
    
    Returns:
        tv_loss_value: TV损失值
    """
    batch_size, channels, height, width = x.size()
    
    # 计算水平方向的差异
    diff_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    
    # 计算垂直方向的差异
    diff_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    
    # 求和并平均
    tv_loss_value = (diff_x.sum() + diff_y.sum()) / (batch_size * channels)
    
    return tv_loss_value


class CombinedLoss(torch.nn.Module):
    """
    组合损失函数 - L1损失 + SSIM损失 + 梯度损失 + TV损失的加权组合
    
    总损失 = λ1 * L1_loss + λ2 * (1 - SSIM) + λ3 * Gradient_loss + λ4 * TV_loss
    
    Attributes:
        l1_weight (float): L1损失权重 (默认: 1.0)
        ssim_weight (float): SSIM损失权重 (默认: 1.0)
        grad_weight (float): 梯度损失权重 (默认: 1.0)
        tv_weight (float): TV损失权重 (默认: 0.3)
        ssim_loss_fn: SSIM损失函数
        grad_loss_fn: 梯度损失函数
        tv_loss_fn: TV损失函数
    """
    
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, grad_weight=1.0, tv_weight=0.3):
        super(CombinedLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.tv_weight = tv_weight
        
        # L1损失
        self.l1_loss = torch.nn.L1Loss()
        
        # SSIM损失
        self.ssim_loss_fn = msssim
        
        # 梯度损失
        self.grad_loss_fn = GradientLoss()
        
        # TV损失
        self.tv_loss_fn = TVLoss()
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            total_loss: 总损失值
            loss_dict: 包含各损失分量的字典
        """
        # L1损失
        l1_loss_value = self.l1_loss(pred, target)
        
        # SSIM损失 (1 - SSIM)
        ssim_loss_value = 1 - self.ssim_loss_fn(pred, target, normalize=True)
        
        # 梯度损失
        grad_loss_value = self.grad_loss_fn(pred, target)
        
        # TV损失
        tv_loss_value = self.tv_loss_fn(pred)
        
        # 加权组合
        total_loss = (
            self.l1_weight * l1_loss_value +
            self.ssim_weight * ssim_loss_value +
            self.grad_weight * grad_loss_value +
            self.tv_weight * tv_loss_value
        )
        
        loss_dict = {
            'l1_loss': l1_loss_value.item(),
            'ssim_loss': ssim_loss_value.item(),
            'grad_loss': grad_loss_value.item(),
            'tv_loss': tv_loss_value.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def update_weights(self, l1_weight=None, ssim_weight=None, grad_weight=None, tv_weight=None):
        """
        动态更新损失权重
        
        Args:
            l1_weight: L1损失权重
            ssim_weight: SSIM损失权重
            grad_weight: 梯度损失权重
            tv_weight: TV损失权重
        """
        if l1_weight is not None:
            self.l1_weight = l1_weight
        if ssim_weight is not None:
            self.ssim_weight = ssim_weight
        if grad_weight is not None:
            self.grad_weight = grad_weight
        if tv_weight is not None:
            self.tv_weight = tv_weight
