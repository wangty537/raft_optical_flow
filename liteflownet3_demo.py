#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteFlowNet3 演示脚本
提供简单的光流估计和可视化功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import argparse
from typing import Tuple, Optional

from liteflownet3_simple import (
    liteflownet3, liteflownet3s, 
    LiteFlowNet3, LiteFlowNet3S,
    LiteFlowNet3PseudoReg, LiteFlowNet3SPseudoReg
)
from torchvision.utils import flow_to_image
import torch
def flow_to_image_torch(flow: np.ndarray) -> np.ndarray:
    """
    将光流转换为可视化图像
    
    Args:
        flow: 光流数组 [H, W, 2], float, xy
    
    Returns:
        可视化光流图像 [H, W, 3]，值范围[0, 255] rgb format
    """
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    img = np.transpose(flow_im.numpy(), [1, 2, 0])
    print(img.shape)
    return img


def load_image(img_path: str, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    加载图像并转换为PyTorch张量
    
    Args:
        img_path: 图像路径
        target_size: 目标尺寸 (height, width)，如果为None则保持原始尺寸
    
    Returns:
        图像张量 [1, 3, H, W]，值范围[0, 1]
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # BGR转RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # 转换为张量并归一化
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
    
    return img_tensor





def load_model(model_path: str, model_type: str = 'auto') -> torch.nn.Module:
    """
    加载LiteFlowNet3模型
    
    Args:
        model_path: 模型权重路径
        model_type: 模型类型 ('liteflownet3', 'liteflownet3s', 'auto')
    
    Returns:
        加载的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 自动检测模型类型
    if model_type == 'auto':
        if 'liteflownet3s' in model_path.lower():
            model_type = 'liteflownet3s'
        else:
            model_type = 'liteflownet3'
    
    # 创建模型
    if model_type == 'liteflownet3s':
        model = LiteFlowNet3S()
    else:
        model = LiteFlowNet3()
    
    # 加载权重
    if model_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 移除可能的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        # 普通PyTorch权重
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载: {model_type} from {model_path}")
    return model


def estimate_flow_and_visualize(img1_path: str, img2_path: str, model_path: str, 
                               output_path: str = 'flow_visualization.png',
                               target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    估计光流并生成可视化图像
    
    Args:
        img1_path: 第一张图像路径
        img2_path: 第二张图像路径
        model_path: 模型权重路径
        output_path: 输出可视化图像路径
        target_size: 目标尺寸 (height, width)，如果为None则使用原始尺寸
    
    Returns:
        光流数组 [H, W, 2]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载图像
    print("加载图像...")
    img1 = load_image(img1_path, target_size) # rgb 13hw 0-1
    img2 = load_image(img2_path, target_size)
    
    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    print(f"图像尺寸: {img1.shape[2]}x{img1.shape[3]}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path)
    
    # 估计光流
    print("估计光流...")
    with torch.no_grad():
        # 准备输入 - 正确的格式是 [batch, 2, 3, H, W]
        images = torch.stack([img1, img2], dim=1)  # [1, 2, 3, H, W]
        input_dict = {'images': images}
        
        # 前向推理
        output = model(input_dict)
        if isinstance(output, dict):
            if 'flows' in output:
                flow = output['flows'][0, 0]  # 取第一个时间步的光流
            elif 'flow' in output:
                flow = output['flow']
            else:
                flow = list(output.values())[0]  # 取第一个输出
        else:
            flow = output
    
    # 转换为numpy数组
    print(f"光流张量形状: {flow.shape}")
    
    # 处理不同的光流输出格式
    if len(flow.shape) == 5:  # [B, T, C, H, W]
        flow_tensor = flow[0, 0]  # 取第一个batch和时间步
    elif len(flow.shape) == 4:  # [B, C, H, W]
        flow_tensor = flow[0]  # 取第一个batch
    elif len(flow.shape) == 3:  # [C, H, W]
        flow_tensor = flow
    else:
        raise ValueError(f"不支持的光流张量形状: {flow.shape}")
    
    flow_np = flow_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
    
    print(f"光流统计信息:")
    print(f"  形状: {flow_np.shape}")
    print(f"  范围: [{flow_np.min():.2f}, {flow_np.max():.2f}]")
    print(f"  平均幅度: {np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2).mean():.2f}")
    
    # 创建可视化
    print("生成可视化...")

    flow_img = flow_to_image_torch(flow_np) 
    cv2.imwrite(output_path, flow_img[...,::-1])
    
    return flow_np


def main():
    """
    命令行接口
    """
    parser = argparse.ArgumentParser(description='LiteFlowNet3 光流估计演示')
    parser.add_argument('--img1', default='demo-frames/frame_0016.png', help='第一张图像路径')
    parser.add_argument('--img2', default='demo-frames/frame_0017.png', help='第二张图像路径')
    parser.add_argument('--model', default='liteflownet3s-sintel-89793e34.ckpt', help='模型权重路径')
    parser.add_argument('-o', '--output', default='flow_vis_liteflownet3.png', 
                       help='输出可视化图像路径')
    parser.add_argument('--size', nargs=2, type=int, metavar=('H', 'W'),
                       help='目标图像尺寸 (高度 宽度)')
    
    args = parser.parse_args()
    print(f"使用参数: {args}")
    
    # 检查输入文件
    for path in [args.img1, args.img2, args.model]:
        if not Path(path).exists():
            print(f"错误: 文件不存在 {path}")
            return
    
    target_size = tuple(args.size) if args.size else None
    
    try:
        # 估计光流并可视化
        flow = estimate_flow_and_visualize(
            args.img1, args.img2, args.model, 
            args.output, target_size
        )
        print("\n光流估计完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    
    main()