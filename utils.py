import torch
from torch import nn
from torch.nn import functional as F
import os
import yaml
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re


class jindu():
    """ 进度条用法:
    >>> import time
    >>> bar = jindu(100, name='Bar', limit_length=30)    # 定义进度条
    >>> for i in range(100):
    >>>     time.sleep(0.1) 
    >>>     if i % 20 == 0:
    >>>         bar.show(f'{i}+{i}={i + i}')             # 代码中的其他输出
    >>>     bar.update()                                 # 更新进度条
    0+0=0
    20+20=40
    40+40=80
    60+60=120
    80+80=160
    Bar: ==============================> 100.0%
    """
    def __init__(self, total, name='', limit_length=None):
        """ total: 项目的总数
        name: 进度条的名称
        limit_length: 显示的进度条长度
        """
        self.name = name
        self.real_now = 0
        self.now = 0
        self.total_lenght = total
        self.limit_lenght = limit_length if limit_length else total
        self.tail_length = 0
    
    def before_print(self):
        """ 输出前清除行
        """
        if self.name != '':
            print(' ' * (len(self.name) + 2), end='')
        print(' ' * (self.limit_lenght + 6 + self.tail_length + 1), end='\r')

    def update(self, i=1, tail=''):
        """ 更新进度条
        """
        self.real_now += i
        self.now = int(self.real_now / self.total_lenght * self.limit_lenght)
        if self.name != '':
            print(self.name + ':', end=' ')
        print('=' * self.now + '>' + '-' * (self.limit_lenght - 1 - self.now), 
              f'{(self.real_now / self.total_lenght * 100.):.1f}% {tail}',
              end='\r' if self.real_now < self.total_lenght else '\n')
        self.tail_length = len(tail)

    def show(self, *args, **kwargs):
        """ 和print使用方式一样
        """
        self.before_print()
        for item in args:
            print(item, end=' ')
        print(end=kwargs['end'] if kwargs else '\n')


class Smish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh(torch.log(1+torch.sigmoid(input)))


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def grayscale_linear_transform(images, eps=1e-12):
    """ Grayscale linear transformation
    """
    return (images - images.min()) / (images.max() - images.min() + eps)


def aggregate_metrics(root, output_file):
    # 存储所有 metrics 的数据
    all_metrics = []

    # 遍历每个文件夹
    for folder in os.listdir(root):
        # 构建 metrics.csv 文件的路径
        csv_path = os.path.join(root, folder, "metrics.csv")
        print(csv_path)

        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告：文件 {csv_path} 不存在，跳过此文件夹。")
            continue

        # 读取 CSV 文件
        try:
            # 读取 CSV 文件，跳过第一行作为列名
            df = pd.read_csv(csv_path, header=0)
            # 将数值转换为浮点数并存储
            all_metrics.append(df)
        except Exception as e:
            print(f"读取文件 {csv_path} 时出错：{e}")
            continue

    # 合并所有 metrics
    if not all_metrics:
        print("没有找到任何有效的 metrics 数据，无法创建输出文件。")
        return

    # 合并所有文件的数据
    combined_metrics = pd.concat(all_metrics, axis=0)
    
    # 计算每列的平均值和标准差
    mean_values = combined_metrics.mean().round(3)
    std_dev_values = combined_metrics.std().round(2)

    # 创建结果 DataFrame
    result = pd.DataFrame({
        'Mean': mean_values,
        'Std Dev': std_dev_values
    })

    result.to_csv(output_file, encoding='utf-8')



def load_config(path):
    """ Load configuration """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config):
    """ Save configuration """
    with open(os.path.join(config['save_path'], 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * torch.tanh(torch.log(1+torch.sigmoid(input)))


def nms_for_edge_confidence_batch(edge_confidence_batch, kernel_size=3):
    """
    Apply Non-Maximum Suppression (NMS) to batch of edge confidence images

    Args:
        edge_confidence_batch (torch.Tensor): Input batch of edge confidence images, shape (batch_size, height, width)
        kernel_size (int): Neighborhood size, typically 3 or 5

    Returns:
        torch.Tensor: Batch of edge confidence images after NMS processing
    """
    # Ensure input is torch.Tensor
    edge_confidence_batch = torch.as_tensor(edge_confidence_batch)
    batch_size, height, width = edge_confidence_batch.shape

    # Create an output image of the same size as input, initialized to 0
    output_batch = torch.zeros_like(edge_confidence_batch)

    # Calculate neighborhood radius
    pad_size = kernel_size // 2

    # Process each sample
    for b in range(batch_size):
        # Get the edge confidence image of current sample
        current_image = edge_confidence_batch[b]

        # Create an output image of the same size as current image, initialized to 0
        output_image = torch.zeros_like(current_image)

        # Use replicate_pad to pad the image to handle boundary cases
        padded_image = torch.nn.functional.pad(current_image.unsqueeze(0).unsqueeze(0), 
                                             [pad_size, pad_size, pad_size, pad_size], 
                                             mode='replicate').squeeze()

        # Iterate through each pixel in the current image
        for i in range(height):
            for j in range(width):
                # Extract current pixel and its neighborhood
                current_value = current_image[i, j].item()
                neighbor_values = padded_image[i:i+kernel_size, j:j+kernel_size]

                # Find the maximum value in the neighborhood
                max_value = neighbor_values.max().item()

                # If current pixel is the maximum in the neighborhood, keep it; otherwise, set to 0
                if current_value == max_value:
                    output_image[i, j] = current_value
                else:
                    output_image[i, j] = 0

        # Add the processed image to the output batch
        output_batch[b] = output_image

    return output_batch


def iterative_thresholding_torch(image_batch, max_iterations=100, tolerance=1e-3):
    """
    Iterative thresholding segmentation function (PyTorch-based), supports batch processing

    Args:
        image_batch (torch.Tensor): Input image tensor, shape (B, 1, H, W)
        max_iterations (int): Maximum number of iterations
        tolerance (float): Threshold change tolerance

    Returns:
        tuple: Segmented image tensor (0 and 255 values), final threshold for each image
    """
    batch_size, channels, height, width = image_batch.shape
    device = image_batch.device

    # Apply Gaussian smoothing
    kernel_size = 5
    sigma = 2.0
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]  # Create 2D Gaussian kernel
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Adapt kernel shape for convolution operation

    # Apply Gaussian smoothing to the image
    padding = kernel_size // 2
    smoothed_image_batch = F.conv2d(image_batch, kernel, padding=padding, groups=1)

    # Initialize threshold (can use image mean as initial threshold)
    thresholds = smoothed_image_batch.mean(dim=(1, 2, 3))  # Calculate initial threshold for each image

    for _ in range(max_iterations):
        new_thresholds = torch.zeros_like(thresholds)
        for i in range(batch_size):
            smoothed_image = smoothed_image_batch[i, 0]  # Current image
            threshold = thresholds[i]

            # Segment image into foreground and background based on current threshold
            foreground_mask = smoothed_image >= threshold
            background_mask = smoothed_image < threshold

            # Calculate average gray values of foreground and background
            mean_foreground = smoothed_image[foreground_mask].mean() if foreground_mask.any() else threshold
            mean_background = smoothed_image[background_mask].mean() if background_mask.any() else threshold

            # Calculate new threshold
            new_threshold = (mean_foreground + mean_background) / 2.0
            new_thresholds[i] = new_threshold

        # Check if threshold change is less than tolerance
        changes = torch.abs(new_thresholds - thresholds)
        if torch.all(changes < tolerance):
            break
        thresholds = new_thresholds

    # Use final threshold to suppress noise
    segmented_image_batch = torch.zeros_like(smoothed_image_batch)
    for i in range(batch_size):
        segmented_image_batch[i, 0] = torch.where(smoothed_image_batch[i, 0] >= thresholds[i], 
                                                  smoothed_image_batch[i, 0], 
                                                  torch.tensor(0.0, device=device))

    return segmented_image_batch


def plot_curve(X, Y=None, xlabel=None, ylabel=None, legend=None, 
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:', 'c--', 'y-.'), 
         figsize=(3.5, 2.5), axes=None):
    """
    d2l.plot 核心实现
    - 如果传了 Y=None，假设 X 是时间序列，自动生成 x 轴
    - 支持同时绘制多条曲线（X/Y 可以是列表）
    """
    # 1. 准备画布
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)
        axes = axes  # 单图情况
    
    # 2. 数据标准化（确保是列表格式，支持多曲线）
    def has_one_axis(X):  # 判断是否为标量/一维
        return (hasattr(X, "ndim") and X.ndim == 1) or \
               (isinstance(X, list) and not isinstance(X[0], list))
    
    if has_one_axis(X):
        X = [X]  # 包装成列表
    if Y is None:
        X, Y = [range(len(X[0]))], X  # 如果没有Y，X当作Y，生成时间轴
    elif has_one_axis(Y):
        Y = [Y]
    
    # 确保 X 和 Y 数量匹配（如果只有一个 X 对应多个 Y）
    if len(X) != len(Y):
        X = X * len(Y)
    
    # 3. 清空并绘制（支持动态更新）
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
    
    # 4. 设置坐标轴属性
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    
    # 5. 图例处理
    if legend:
        axes.legend(legend)
        axes.legend(legend, loc='upper right')  # d2l 默认位置
    
    # 6. 网格和布局
    axes.grid(True)
    plt.tight_layout()
    
    return fig, axes