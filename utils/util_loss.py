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


class MultiScaleGradientLoss(torch.nn.Module):
    """
    融合方向感知多尺度梯度损失函数
    
    该损失函数通过多尺度梯度计算和方向感知机制，有效捕捉图像不同尺度下的梯度信息，
    特别适用于图像融合任务中保持边缘和纹理细节。
    
    特点：
    - 多尺度梯度计算：在不同尺度下计算梯度，捕捉不同层次的边缘信息
    - 方向感知：考虑水平和垂直方向的梯度特征
    - 数值稳定性：使用平滑处理和数值稳定机制
    - 可配置性：支持自定义尺度数量和权重
    """
    
    def __init__(self, scales=4, alpha=1.0, beta=1.0, epsilon=1e-8, size_average=True):
        """
        初始化多尺度梯度损失函数
        
        Args:
            scales (int): 梯度计算的尺度数量，默认为4
            alpha (float): 水平梯度权重，默认为1.0
            beta (float): 垂直梯度权重，默认为1.0
            epsilon (float): 数值稳定常数，防止除零错误，默认为1e-8
            size_average (bool): 是否对损失值进行平均，默认为True
        """
        super(MultiScaleGradientLoss, self).__init__()
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.size_average = size_average
        
        # 创建高斯核用于多尺度平滑
        self.gaussian_kernels = self._create_gaussian_kernels()
    
    def _create_gaussian_kernels(self):
        """创建多尺度高斯核用于图像平滑"""
        kernels = []
        for i in range(self.scales):
            # 不同尺度的高斯核大小
            kernel_size = 5 + 2 * i
            sigma = 1.0 + 0.5 * i
            
            # 创建1D高斯核
            x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
            gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            
            # 创建2D高斯核
            gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
            gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
            kernels.append(gauss_2d)
        
        return kernels
    
    def _compute_gradients(self, img):
        """计算图像在水平和垂直方向的梯度"""
        # Sobel算子用于梯度计算
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 将算子移动到与图像相同的设备
        sobel_x = sobel_x.to(img.device)
        sobel_y = sobel_y.to(img.device)
        
        # 对每个通道分别计算梯度
        channels = img.size(1)
        grad_x_list = []
        grad_y_list = []
        
        for i in range(channels):
            # 提取单个通道
            channel_img = img[:, i:i+1, :, :]
            
            # 计算该通道的梯度
            grad_x_channel = F.conv2d(channel_img, sobel_x, padding=1)
            grad_y_channel = F.conv2d(channel_img, sobel_y, padding=1)
            
            grad_x_list.append(grad_x_channel)
            grad_y_list.append(grad_y_channel)
        
        # 合并所有通道的梯度
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        
        return grad_x, grad_y
    
    def _smooth_image(self, img, kernel):
        """使用高斯核对图像进行平滑"""
        # 将高斯核移动到与图像相同的设备
        kernel = kernel.to(img.device)
        
        # 对每个通道进行平滑
        smoothed = F.conv2d(img, kernel.repeat(img.size(1), 1, 1, 1), 
                           padding=kernel.size(-1)//2, groups=img.size(1))
        return smoothed
    
    def _compute_scale_loss(self, pred, target, scale_idx):
        """在特定尺度下计算梯度损失"""
        # 对预测和目标图像进行平滑
        kernel = self.gaussian_kernels[scale_idx]
        pred_smooth = self._smooth_image(pred, kernel)
        target_smooth = self._smooth_image(target, kernel)
        
        # 计算梯度
        pred_grad_x, pred_grad_y = self._compute_gradients(pred_smooth)
        target_grad_x, target_grad_y = self._compute_gradients(target_smooth)
        
        # 计算梯度差异
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        # 计算方向感知损失
        direction_loss = self.alpha * grad_diff_x + self.beta * grad_diff_y
        
        # 应用尺度权重（越精细的尺度权重越大）
        scale_weight = 1.0 / (2 ** scale_idx)
        
        return direction_loss * scale_weight
    
    def forward(self, pred, target):
        """
        前向传播计算多尺度梯度损失
        
        Args:
            pred (torch.Tensor): 预测图像，形状为[B, C, H, W]
            target (torch.Tensor): 目标图像，形状为[B, C, H, W]
            
        Returns:
            torch.Tensor: 多尺度梯度损失值
        """
        # 输入验证
        assert pred.shape == target.shape, "预测和目标图像形状不匹配"
        assert pred.dim() == 4, "输入应为4D张量 [B, C, H, W]"
        
        # 初始化总损失
        total_loss = 0.0
        
        # 在每个尺度下计算损失
        for scale_idx in range(self.scales):
            scale_loss = self._compute_scale_loss(pred, target, scale_idx)
            
            # 数值稳定性处理
            scale_loss = torch.clamp(scale_loss, min=self.epsilon)
            
            if self.size_average:
                scale_loss = scale_loss.mean()
            else:
                scale_loss = scale_loss.sum()
            
            total_loss += scale_loss
        
        # 对总损失进行平均
        if self.size_average:
            total_loss = total_loss / self.scales
        
        return total_loss
