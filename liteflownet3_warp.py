# Code taken from IRR: https://github.com/visinf/irr
# Licensed under the Apache 2.0 license (see LICENSE_IRR).

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grid(x):
    """
    生成用于grid_sample的标准化坐标网格
    
    原理:
    - 创建一个标准化的坐标网格，范围为[-1, 1]
    - 水平坐标(grid_H)对应图像的宽度维度
    - 垂直坐标(grid_V)对应图像的高度维度
    - 这个网格用作grid_sample函数的基础坐标系统
    
    Args:
        x: 输入张量 [B, C, H, W]
    
    Returns:
        grids_cuda: 标准化坐标网格 [B, 2, H, W]
                   其中第二维度0为水平坐标，1为垂直坐标
    """
    # 创建水平坐标网格: 从-1到1线性分布，对应图像宽度
    # shape: [1, 1, 1, W] -> [B, 1, H, W]
    grid_H = (
        torch.linspace(-1.0, 1.0, x.size(3))  # [W]
        .view(1, 1, 1, x.size(3))              # [1, 1, 1, W]
        .expand(x.size(0), 1, x.size(2), x.size(3))  # [B, 1, H, W]
    )
    
    # 创建垂直坐标网格: 从-1到1线性分布，对应图像高度
    # shape: [1, 1, H, 1] -> [B, 1, H, W]
    grid_V = (
        torch.linspace(-1.0, 1.0, x.size(2))  # [H]
        .view(1, 1, x.size(2), 1)              # [1, 1, H, 1]
        .expand(x.size(0), 1, x.size(2), x.size(3))  # [B, 1, H, W]
    )
    
    # 拼接水平和垂直坐标: [B, 2, H, W]
    # 第二维度: 0=水平坐标(x), 1=垂直坐标(y)
    grid = torch.cat([grid_H, grid_V], 1)  # [B, 2, H, W]
    
    # 转换到目标设备和数据类型，不需要梯度
    grids_cuda = grid.requires_grad_(False).to(dtype=x.dtype, device=x.device)
    return grids_cuda


class WarpingLayer(nn.Module):
    """
    图像扭曲层，根据光流对图像进行扭曲变换
    
    原理:
    1. 将光流从像素坐标转换为标准化坐标[-1, 1]
    2. 将光流添加到基础网格上，得到采样坐标
    3. 使用grid_sample进行双线性插值采样
    4. 生成遮挡掩码，处理超出边界的区域
    
    这是光流网络中的核心组件，用于实现基于光流的图像对齐
    """
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            flow: 光流 [B, 2, H, W], 其中第二维度0=水平流，1=垂直流
            height_im: 图像高度 (int)
            width_im: 图像宽度 (int)
            div_flow: 光流缩放因子 (float)多尺度平衡 ：确保不同金字塔层级的光流预测都在合理的数值范围内
        
        Returns:
            扭曲后的图像 [B, C, H, W]，超出边界的区域被掩码为0
        """
        flo_list = []
        
        # 将光流从像素坐标转换为标准化坐标[-1, 1]
        # 原理: grid_sample要求坐标在[-1, 1]范围内
        # 这里并不是转换到[-1, 1], 而是归一化到[0, 2], 因为get_grid是[-1, 1]
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow   # [B, H, W] 水平流
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow   # [B, H, W] 垂直流
        
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        
        # 重新组织光流维度: [2, B, H, W] -> [B, 2, H, W]
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)  # [B, 2, H, W]
        
        # 将光流添加到基础网格，得到采样坐标
        # get_grid(x): [B, 2, H, W] 基础标准化网格
        # flow_for_grid: [B, 2, H, W] 标准化光流
        # 结果: [B, 2, H, W] -> [B, H, W, 2] (grid_sample要求的格式)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)  # [B, H, W, 2]
        
        # 使用双线性插值进行图像采样
        # grid: [B, H, W, 2] 采样坐标，最后一维为(x, y)
        x_warp = F.grid_sample(x, grid, align_corners=True)  # [B, C, H, W]

        # 创建遮挡掩码，处理超出图像边界的区域
        # 原理: 对全1张量进行相同的采样，值<1的区域表示超出边界
        mask = torch.ones(x.size(), requires_grad=False).to(
            dtype=x.dtype, device=x.device
        )  # [B, C, H, W]
        mask = F.grid_sample(mask, grid, align_corners=True)  # [B, C, H, W]
        mask = (mask >= 1.0).to(dtype=x.dtype)  # [B, C, H, W] 二值掩码

        # 应用掩码，将超出边界的区域设为0
        return x_warp * mask  # [B, C, H, W]
