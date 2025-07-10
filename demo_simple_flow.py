#!/usr/bin/env python3
"""
简单光流网络演示脚本

这个脚本演示了如何使用训练好的简单光流网络进行推理
支持图像对和视频序列的光流估计
"""

import os
import argparse
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from simple_flow_net import create_simple_flow_model

# 尝试导入光流可视化工具
try:
    from core.utils.flow_viz import flow_to_image
except ImportError:
    print("警告: 无法导入flow_viz，将使用简单的可视化方法")
    flow_to_image = None


def load_image(image_path, device):
    """
    加载并预处理图像
    """
    image = Image.open(image_path).convert('RGB')
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, image


def simple_flow_to_image(flow):
    """
    简单的光流可视化函数
    将光流转换为RGB图像
    """
    # flow shape: [H, W, 2]
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    
    # 归一化
    flow_magnitude = flow_magnitude / (np.max(flow_magnitude) + 1e-8)
    flow_angle = (flow_angle + np.pi) / (2 * np.pi)  # [0, 1]
    
    # 创建HSV图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[:, :, 0] = flow_angle  # Hue
    hsv[:, :, 1] = 1.0         # Saturation
    hsv[:, :, 2] = flow_magnitude  # Value
    
    # 转换为RGB
    import colorsys
    rgb = np.zeros_like(hsv)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            rgb[i, j] = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])
    
    return (rgb * 255).astype(np.uint8)


def visualize_flow(flow_np):
    """
    可视化光流
    """
    if flow_to_image is not None:
        # 使用专业的光流可视化工具
        flow_img = flow_to_image(flow_np)
    else:
        # 使用简单的可视化方法
        flow_img = simple_flow_to_image(flow_np)
    
    return flow_img


def estimate_flow_pair(model, img1_path, img2_path, output_path=None, device='cuda'):
    """
    估计一对图像的光流
    """
    print(f"处理图像对: {img1_path} -> {img2_path}")
    
    # 加载图像
    img1_tensor, img1_pil = load_image(img1_path, device)
    img2_tensor, img2_pil = load_image(img2_path, device)
    
    # 确保图像尺寸相同
    if img1_tensor.shape != img2_tensor.shape:
        print("警告: 图像尺寸不同，将调整到相同尺寸")
        h, w = min(img1_tensor.shape[2], img2_tensor.shape[2]), min(img1_tensor.shape[3], img2_tensor.shape[3])
        img1_tensor = F.interpolate(img1_tensor, size=(h, w), mode='bilinear', align_corners=True)
        img2_tensor = F.interpolate(img2_tensor, size=(h, w), mode='bilinear', align_corners=True)
    
    # 推理
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        flow_preds = model(img1_tensor, img2_tensor)
        inference_time = time.time() - start_time
    
    # 获取最高分辨率的光流预测
    flow_pred = flow_preds[-1]
    
    # 转换为numpy
    flow_np = flow_pred[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
    
    print(f"推理时间: {inference_time:.3f}s")
    print(f"光流范围: x=[{flow_np[:,:,0].min():.2f}, {flow_np[:,:,0].max():.2f}], "
          f"y=[{flow_np[:,:,1].min():.2f}, {flow_np[:,:,1].max():.2f}]")
    
    # 可视化
    flow_img = visualize_flow(flow_np)
    
    # 创建结果图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(img1_pil)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_pil)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    # 光流可视化
    axes[1, 0].imshow(flow_img)
    axes[1, 0].set_title('Optical Flow')
    axes[1, 0].axis('off')
    
    # 光流幅度
    flow_magnitude = np.sqrt(flow_np[:,:,0]**2 + flow_np[:,:,1]**2)
    im = axes[1, 1].imshow(flow_magnitude, cmap='hot')
    axes[1, 1].set_title('Flow Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 保存结果
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果保存到: {output_path}")
        
        # 同时保存光流数组
        flow_save_path = output_path.replace('.png', '_flow.npy')
        np.save(flow_save_path, flow_np)
        print(f"光流数据保存到: {flow_save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return flow_np


def estimate_flow_sequence(model, image_dir, output_dir=None, device='cuda'):
    """
    估计图像序列的光流
    """
    # 获取图像列表
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) < 2:
        print("错误: 需要至少2张图像")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 逐对处理图像
    for i in range(len(image_files) - 1):
        img1_path = os.path.join(image_dir, image_files[i])
        img2_path = os.path.join(image_dir, image_files[i + 1])
        
        if output_dir:
            output_path = os.path.join(output_dir, f'flow_{i:04d}_{i+1:04d}.png')
        else:
            output_path = None
        
        flow_np = estimate_flow_pair(model, img1_path, img2_path, output_path, device)
        
        print(f"完成 {i+1}/{len(image_files)-1}")
    
    print("序列处理完成!")


def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = create_simple_flow_model().to(device)
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("模型加载成功")
    else:
        print(f"警告: 检查点文件不存在 {checkpoint_path}，使用随机初始化的模型")
    
    return model


def create_demo_images(output_dir='./demo_images'):
    """
    创建演示图像（如果没有真实图像的话）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建简单的移动图案
    size = 256
    
    # 第一张图像：圆形
    img1 = np.zeros((size, size, 3), dtype=np.uint8)
    center1 = (size//2 - 20, size//2)
    radius = 30
    y, x = np.ogrid[:size, :size]
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 <= radius**2
    img1[mask1] = [255, 0, 0]  # 红色圆形
    
    # 第二张图像：移动后的圆形
    img2 = np.zeros((size, size, 3), dtype=np.uint8)
    center2 = (size//2 + 20, size//2)
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 <= radius**2
    img2[mask2] = [255, 0, 0]  # 红色圆形
    
    # 保存图像
    Image.fromarray(img1).save(os.path.join(output_dir, 'img1.png'))
    Image.fromarray(img2).save(os.path.join(output_dir, 'img2.png'))
    
    print(f"演示图像创建完成: {output_dir}")
    return os.path.join(output_dir, 'img1.png'), os.path.join(output_dir, 'img2.png')


def main():
    parser = argparse.ArgumentParser(description='简单光流网络演示')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='liteflownet3s-sintel-89793e34.ckpt',
                       help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    # 输入参数
    parser.add_argument('--img1', type=str, default='demo-frames/frame_0017.png', help='第一张图像路径')
    parser.add_argument('--img2', type=str, default='demo-frames/frame_0018.png', help='第二张图像路径')
    parser.add_argument('--image_dir', type=str, default=None, help='图像序列目录')
    parser.add_argument('--demo', action='store_true', help='使用演示图像')
    
    # 输出参数
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录（用于序列）')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    if args.demo:
        # 演示模式
        print("演示模式：创建演示图像")
        img1_path, img2_path = create_demo_images()
        output_path = args.output or 'demo_flow_result.png'
        estimate_flow_pair(model, img1_path, img2_path, output_path, args.device)
        
    elif args.image_dir:
        # 序列模式
        print(f"序列模式：处理目录 {args.image_dir}")
        estimate_flow_sequence(model, args.image_dir, args.output_dir, args.device)
        
    elif args.img1 and args.img2:
        # 图像对模式
        print("图像对模式")
        estimate_flow_pair(model, args.img1, args.img2, args.output, args.device)
        
    else:
        print("错误: 请指定输入图像")
        print("使用 --demo 进行演示，或指定 --img1 和 --img2，或指定 --image_dir")
        return
    
    print("演示完成!")


if __name__ == '__main__':
    main()