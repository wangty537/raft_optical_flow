#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteFlowNet3 无监督训练脚本
使用光度一致性损失、平滑性损失和遮挡处理进行无监督光流学习
"""

import os
import glob
import argparse
from pathlib import Path
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from core import datasets
from liteflownet3_simple import LiteFlowNet3Simple


class Logger:
    """日志记录类，同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # 创建日志文件目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 打开日志文件
        self.log = open(log_file, 'a', encoding='utf-8')
        
        # 记录开始时间
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log.write(f"\n{'='*50}\n")
        self.log.write(f"训练开始时间: {start_time}\n")
        self.log.write(f"{'='*50}\n")
        self.log.flush()
    
    def write(self, message):
        """同时写入控制台和文件"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """关闭日志文件"""
        if hasattr(self, 'log') and not self.log.closed:
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log.write(f"\n{'='*50}\n")
            self.log.write(f"训练结束时间: {end_time}\n")
            self.log.write(f"{'='*50}\n")
            self.log.close()
    
    def log_args(self, args):
        """记录命令行参数"""
        self.write("\n训练参数:\n")
        self.write("-" * 30 + "\n")
        for key, value in vars(args).items():
            self.write(f"{key}: {value}\n")
        self.write("-" * 30 + "\n\n")


def print(*args, **kwargs):
    """重写print函数，使其同时输出到控制台和日志文件"""
    import builtins
    builtins.print(*args, **kwargs)
    if hasattr(sys.stdout, 'log'):
        # 如果stdout被重定向到Logger，则已经记录到文件了
        pass


def flow_to_rgb(flow):
    """将光流转换为RGB图像用于可视化"""
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.minimum(v * 4, 255)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def warp_image(img, flow):
    """
    使用光流对图像进行扭曲
    Args:
        img: 输入图像 [B, C, H, W]
        flow: 光流 [B, 2, H, W]
    Returns:
        warped_img: 扭曲后的图像 [B, C, H, W]
    """
    B, C, H, W = img.shape
    
    # 创建网格
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img.device)
    
    # 添加光流
    vgrid = grid + flow
    
    # 归一化到[-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # 使用grid_sample进行扭曲
    warped_img = F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped_img


def compute_occlusion_mask(flow_fw, flow_bw):
    """
    计算遮挡掩码
    Args:
        flow_fw: 前向光流 [B, 2, H, W]
        flow_bw: 后向光流 [B, 2, H, W]
    Returns:
        occlusion_mask: 遮挡掩码 [B, 1, H, W]
    """
    # 使用前向光流扭曲后向光流
    warped_flow_bw = warp_image(flow_bw, flow_fw)
    
    # 计算前向-后向一致性
    flow_diff = flow_fw + warped_flow_bw
    flow_mag = torch.sqrt(torch.sum(flow_fw**2, dim=1, keepdim=True)) + \
               torch.sqrt(torch.sum(warped_flow_bw**2, dim=1, keepdim=True))
    
    # 计算相对误差
    occlusion = torch.sqrt(torch.sum(flow_diff**2, dim=1, keepdim=True)) > \
                0.01 * flow_mag + 0.5
    
    # 返回非遮挡区域的掩码
    return (~occlusion).float()


def compute_smoothness_loss(flow, img):
    """
    计算平滑性损失
    Args:
        flow: 光流 [B, 2, H, W]
        img: 参考图像 [B, 3, H, W]
    Returns:
        smoothness_loss: 平滑性损失
    """
    # 计算图像梯度
    img_grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    img_grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    
    # 计算光流梯度
    flow_grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    flow_grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    
    # 计算权重（边缘保持）
    weights_x = torch.exp(-torch.mean(img_grad_x, dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(img_grad_y, dim=1, keepdim=True))
    
    # 加权平滑性损失
    smoothness_x = torch.mean(weights_x * torch.sum(flow_grad_x, dim=1, keepdim=True))
    smoothness_y = torch.mean(weights_y * torch.sum(flow_grad_y, dim=1, keepdim=True))
    
    return smoothness_x + smoothness_y


def compute_photometric_loss(img1, img2, flow):
    """
    计算光度一致性损失
    Args:
        img1: 第一帧图像 [B, 3, H, W]
        img2: 第二帧图像 [B, 3, H, W]
        flow: 从img1到img2的光流 [B, 2, H, W]
    Returns:
        photometric_loss: 光度一致性损失
    """
    # 使用光流扭曲第二帧图像
    warped_img2 = warp_image(img2, flow)
    
    # 计算L1损失
    photometric_loss = torch.mean(torch.abs(img1 - warped_img2))
    
    return photometric_loss


class UnsupervisedLoss(nn.Module):
    """无监督光流损失函数"""
    def __init__(self, alpha_photo=1.0, alpha_smooth=0.1, alpha_consist=0.1):
        super().__init__()
        self.alpha_photo = alpha_photo      # 光度一致性损失权重
        self.alpha_smooth = alpha_smooth    # 平滑性损失权重
        self.alpha_consist = alpha_consist  # 前向-后向一致性损失权重
    
    def forward(self, img1, img2, flow_fw, flow_bw=None):
        """
        计算无监督损失
        Args:
            img1: 第一帧图像 [B, 3, H, W]
            img2: 第二帧图像 [B, 3, H, W]
            flow_fw: 前向光流 (img1->img2) [B, 2, H, W]
            flow_bw: 后向光流 (img2->img1) [B, 2, H, W], 可选
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 1. 光度一致性损失
        photo_loss = compute_photometric_loss(img1, img2, flow_fw)
        
        # 2. 平滑性损失
        smooth_loss = compute_smoothness_loss(flow_fw, img1)
        
        # 3. 前向-后向一致性损失（如果提供后向光流）
        consist_loss = 0.0
        if flow_bw is not None:
            # 计算遮挡掩码
            occlusion_mask = compute_occlusion_mask(flow_fw, flow_bw)
            
            # 在非遮挡区域计算一致性损失
            warped_flow_bw = warp_image(flow_bw, flow_fw)
            consist_loss = torch.mean(occlusion_mask * torch.abs(flow_fw + warped_flow_bw))
        
        # 总损失
        total_loss = (self.alpha_photo * photo_loss + 
                     self.alpha_smooth * smooth_loss + 
                     self.alpha_consist * consist_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'photometric': photo_loss.item(),
            'smoothness': smooth_loss.item(),
            'consistency': consist_loss.item() if isinstance(consist_loss, torch.Tensor) else consist_loss
        }
        
        return total_loss, loss_dict


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, writer, global_step):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_photo_loss = 0.0
    epoch_smooth_loss = 0.0
    epoch_consist_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        # 解包数据 (img1, img2, flow, valid) - 无监督训练不使用flow和valid
        img1, img2, _, _ = batch
        
        # 数据移到GPU
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # 前向传播 - 计算前向光流
        optimizer.zero_grad()
        
        # 构建前向输入：[B, 2, 3, H, W]
        images_fw = torch.stack([img1, img2], dim=1)
        flow_fw = model(images_fw)
        
        # 构建后向输入：[B, 2, 3, H, W]
        images_bw = torch.stack([img2, img1], dim=1)
        flow_bw = model(images_bw)
        
        # 计算无监督损失
        loss, loss_dict = loss_fn(img1, img2, flow_fw, flow_bw)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 更新统计
        epoch_loss += loss_dict['total']
        epoch_photo_loss += loss_dict['photometric']
        epoch_smooth_loss += loss_dict['smoothness']
        epoch_consist_loss += loss_dict['consistency']
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss_dict["total"]:.4f}',
            'Photo': f'{loss_dict["photometric"]:.4f}',
            'Smooth': f'{loss_dict["smoothness"]:.4f}',
            'Consist': f'{loss_dict["consistency"]:.4f}'
        })
        
        # 可视化第一个批次的光流
        if batch_idx == 0 and global_step[0] % 10 == 0:
            # 将光流转换为RGB图像
            flow_rgb = flow_to_rgb(flow_fw[0].detach().cpu().permute(1, 2, 0).numpy())
            flow_rgb = torch.from_numpy(flow_rgb).permute(2, 0, 1) / 255.0
            writer.add_image('Train/Flow_Forward', flow_rgb, global_step[0])
            
            flow_bw_rgb = flow_to_rgb(flow_bw[0].detach().cpu().permute(1, 2, 0).numpy())
            flow_bw_rgb = torch.from_numpy(flow_bw_rgb).permute(2, 0, 1) / 255.0
            writer.add_image('Train/Flow_Backward', flow_bw_rgb, global_step[0])
            
            # 可视化扭曲后的图像
            warped_img2 = warp_image(img2, flow_fw)
            writer.add_image('Train/Img1_Original', img1[0], global_step[0])
            writer.add_image('Train/Img2_Warped', warped_img2[0], global_step[0])
        
        global_step[0] += 1
    
    avg_loss = epoch_loss / num_batches
    avg_photo_loss = epoch_photo_loss / num_batches
    avg_smooth_loss = epoch_smooth_loss / num_batches
    avg_consist_loss = epoch_consist_loss / num_batches
    
    # 记录到TensorBoard
    writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Photometric_Loss', avg_photo_loss, epoch)
    writer.add_scalar('Train/Smoothness_Loss', avg_smooth_loss, epoch)
    writer.add_scalar('Train/Consistency_Loss', avg_consist_loss, epoch)
    
    return avg_loss, avg_photo_loss, avg_smooth_loss, avg_consist_loss


def validate(model, val_loader, loss_fn, device, epoch, writer):
    """
    验证模型性能
    """
    model.eval()
    val_loss = 0.0
    val_photo_loss = 0.0
    val_smooth_loss = 0.0
    val_consist_loss = 0.0
    num_batches = len(val_loader)
    
    # === 收集所有EPE值用于详细统计 ===
    epe_list = []  # 存储所有像素的EPE值
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # 解包数据
            img1, img2, flow_gt, valid = batch
            
            # 将数据移到GPU
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow_gt = flow_gt.to(device)
            valid = valid.to(device)
            
            # 前向传播
            images_fw = torch.stack([img1, img2], dim=1)
            flow_fw = model(images_fw)
            
            images_bw = torch.stack([img2, img1], dim=1)
            flow_bw = model(images_bw)
            
            # 计算无监督损失
            loss, loss_dict = loss_fn(img1, img2, flow_fw, flow_bw)
            
            val_loss += loss_dict['total']
            val_photo_loss += loss_dict['photometric']
            val_smooth_loss += loss_dict['smoothness']
            val_consist_loss += loss_dict['consistency']
            
            # === 计算EPE评估指标（使用真实光流标签） ===
            # 逐样本计算EPE，避免批次间的平均化
            for i in range(flow_fw.shape[0]):
                # 计算单个样本的EPE: sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)
                flow_pred_i = flow_fw[i]  # shape=[2, H, W]
                flow_gt_i = flow_gt[i]    # shape=[2, H, W]
                valid_i = valid[i]        # shape=[H, W]
                
                # 计算每个像素的EPE
                epe = torch.sum((flow_pred_i - flow_gt_i)**2, dim=0).sqrt()  # shape=[H, W]
                
                # 只在有效像素处计算EPE
                valid_mask = valid_i >= 0.5
                if valid_mask.sum() > 0:
                    epe_valid = epe[valid_mask]
                    epe_list.append(epe_valid.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss_dict["total"]:.4f}',
                'Photo': f'{loss_dict["photometric"]:.4f}'
            })
            
            # 可视化第一个批次的光流
            if batch_idx == 0:
                flow_rgb = flow_to_rgb(flow_fw[0].detach().cpu().permute(1, 2, 0).numpy())
                flow_rgb = torch.from_numpy(flow_rgb).permute(2, 0, 1) / 255.0
                writer.add_image('Val/Flow_Forward', flow_rgb, epoch)
                
                # 可视化真实光流
                flow_gt_rgb = flow_to_rgb(flow_gt[0].detach().cpu().permute(1, 2, 0).numpy())
                flow_gt_rgb = torch.from_numpy(flow_gt_rgb).permute(2, 0, 1) / 255.0
                writer.add_image('Val/Flow_GT', flow_gt_rgb, epoch)
                
                # 可视化扭曲效果
                warped_img2 = warp_image(img2, flow_fw)
                writer.add_image('Val/Img1_Original', img1[0], epoch)
                writer.add_image('Val/Img2_Warped', warped_img2[0], epoch)
    
    avg_loss = val_loss / num_batches
    avg_photo_loss = val_photo_loss / num_batches
    avg_smooth_loss = val_smooth_loss / num_batches
    avg_consist_loss = val_consist_loss / num_batches
    
    # === 计算各种评估指标 ===
    if len(epe_list) > 0:
        # 将所有EPE值合并
        epe_all = np.concatenate(epe_list)  # shape=[total_valid_pixels]
        
        avg_epe = np.mean(epe_all)                    # 平均端点误差
        px1_acc = np.mean(epe_all < 1.0)              # 1像素准确率
        px3_acc = np.mean(epe_all < 3.0)              # 3像素准确率  
        px5_acc = np.mean(epe_all < 5.0)              # 5像素准确率
    else:
        avg_epe = float('inf')
        px1_acc = px3_acc = px5_acc = 0.0
    
    # 记录到TensorBoard
    writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
    writer.add_scalar('Val/Photometric_Loss', avg_photo_loss, epoch)
    writer.add_scalar('Val/Smoothness_Loss', avg_smooth_loss, epoch)
    writer.add_scalar('Val/Consistency_Loss', avg_consist_loss, epoch)
    writer.add_scalar('Val/EPE', avg_epe, epoch)
    writer.add_scalar('Val/1px_Acc', px1_acc, epoch)
    writer.add_scalar('Val/3px_Acc', px3_acc, epoch)
    writer.add_scalar('Val/5px_Acc', px5_acc, epoch)
    
    print(f"Validation Epoch {epoch+1} - Total Loss: {avg_loss:.4f}, "
          f"Photo: {avg_photo_loss:.4f}, Smooth: {avg_smooth_loss:.4f}, "
          f"Consist: {avg_consist_loss:.4f}")
    print(f"EPE: {avg_epe:.4f}, 1px: {px1_acc:.4f}, 3px: {px3_acc:.4f}, 5px: {px5_acc:.4f}")
    
    return avg_loss, avg_photo_loss, avg_smooth_loss, avg_consist_loss, avg_epe


def main():
    parser = argparse.ArgumentParser(description='LiteFlowNet3 Unsupervised Training')
    parser.add_argument('--data_dir', type=str, default='/home/redpine/share2/dataset/MPI-Sintel-complete/', help='Sintel数据集路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[384, 512], help='裁剪尺寸')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备')
    parser.add_argument('--log_dir', type=str, default='./logs_unsupervised', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./logs_unsupervised/checkpoints', help='保存目录')
    parser.add_argument('--resume', action='store_true', default=False, help='从检查点恢复训练')
    parser.add_argument('--pretrain', type=str, default='liteflownet3s-kitti-5dffb261.ckpt', help='预训练模型路径')
    
    # 无监督损失权重参数
    parser.add_argument('--alpha_photo', type=float, default=1.0, help='光度一致性损失权重')
    parser.add_argument('--alpha_smooth', type=float, default=0.1, help='平滑性损失权重')
    parser.add_argument('--alpha_consist', type=float, default=0.1, help='前向-后向一致性损失权重')
    
    args = parser.parse_args()
    
    # 初始化日志记录器
    log_file = os.path.join(args.log_dir, 'log.txt')
    logger = Logger(log_file)
    
    # 重定向stdout到logger
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        # 记录训练参数
        logger.log_args(args)
    
        # 创建目录
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 设备
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        
        # 创建数据集
        from core import datasets
        aug_params = {'crop_size': args.crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = datasets.MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = datasets.MpiSintel(aug_params, split='training', dstype='final')        

        train_dataset = 2*sintel_clean + 2*sintel_final 
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

        val_dataset_sintel_clean = datasets.MpiSintel_val(split='training', dstype='clean')
        val_loader_sintel_clean = DataLoader(val_dataset_sintel_clean, batch_size=1, 
                                            shuffle=False, num_workers=2, pin_memory=True)
        
        print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset_sintel_clean)}')
        
        # 创建模型
        model = LiteFlowNet3Simple().to(device)
        loss_fn = UnsupervisedLoss(alpha_photo=args.alpha_photo, 
                                  alpha_smooth=args.alpha_smooth, 
                                  alpha_consist=args.alpha_consist)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)
        
        # 训练状态
        start_epoch = 0
        best_loss = float('inf')
        global_step = [0]
        
        # 加载模型权重
        if args.resume:
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                print(f'从检查点恢复训练: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint['best_loss']
                print(f'恢复到第 {start_epoch} 轮，最佳损失: {best_loss:.4f}')
            else:
                print(f'警告: 检查点文件不存在 {checkpoint_path}，将从头开始训练')
        elif args.pretrain and os.path.exists(args.pretrain):
            print(f'加载预训练模型: {args.pretrain}')
            try:
                # 加载预训练权重
                checkpoint = torch.load(args.pretrain, map_location=device)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('model.'):
                            new_state_dict[k[6:]] = v
                        else:
                            new_state_dict[k] = v
                    model.load_state_dict(new_state_dict, strict=False)
                else:
                    pretrained_dict = checkpoint
                
                # 过滤不匹配的键
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print('预训练模型加载成功')
            except Exception as e:
                print(f'警告: 预训练模型加载失败: {e}，将使用随机初始化')
        
        # TensorBoard
        writer = SummaryWriter(log_dir=args.log_dir)
        
        print('开始无监督训练...')
        print(f'损失权重 - 光度: {args.alpha_photo}, 平滑: {args.alpha_smooth}, 一致性: {args.alpha_consist}')
    
        for epoch in range(start_epoch, args.epochs):
            # 训练
            train_loss, train_photo, train_smooth, train_consist = train_epoch(
                model, train_loader, optimizer, loss_fn, device, epoch, writer, global_step)
            
            # 验证
            val_loss, val_photo, val_smooth, val_consist, val_epe = validate(
                model, val_loader_sintel_clean, loss_fn, device, epoch, writer)
            
            # 更新学习率
            scheduler.step()
            
            # 记录学习率
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch+1}/{args.epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Photo: {train_photo:.4f}, '
                  f'Smooth: {train_smooth:.4f}, Consist: {train_consist:.4f}')
            print(f'  Val - Total: {val_loss:.4f}, Photo: {val_photo:.4f}, '
                  f'Smooth: {val_smooth:.4f}, Consist: {val_consist:.4f}')
            
            # 保存最佳模型（基于验证损失）
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f'  新的最佳模型! 验证损失: {val_loss:.4f}')
            
            # 定期保存检查点
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print('无监督训练完成!')
        writer.close()
    
    except Exception as e:
        print(f'训练过程中发生错误: {e}')
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # 恢复原始stdout并关闭日志
        sys.stdout = original_stdout
        logger.close()
        print(f'日志已保存到: {log_file}')


if __name__ == '__main__':
    main()