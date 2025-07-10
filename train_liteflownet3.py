#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteFlowNet3 简化训练脚本
不依赖外部数据集类，自实现Sintel数据集加载
"""

import os
import glob
import argparse
from pathlib import Path

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
import matplotlib.pyplot as plt

from core import datasets
from liteflownet3_simple import liteflownet3s
import sys
from datetime import datetime


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


class SequenceLoss(nn.Module):
    """多尺度序列损失函数，用于LiteFlowNet3训练"""
    def __init__(self, gamma=0.8, max_flow=400.0, weights=None):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        # 默认权重：从粗到细的金字塔权重
        self.weights = weights if weights is not None else [0.32, 0.08, 0.02, 0.01, 0.005]
    
    def forward(self, flow_preds, flow_gt, valid):
        """计算多尺度序列损失
        Args:
            flow_preds: 多尺度预测光流列表或单个张量
                       如果是列表: [flow_1/32, flow_1/16, flow_1/8, flow_1/4, flow_1/1]
                       如果是张量: [B, 2, H, W] (最终尺度)
            flow_gt: 真实光流 [B, 2, H, W] 
            valid: 有效性掩码 [B, H, W]
        """
        # 如果输入是单个张量，转换为列表
        if isinstance(flow_preds, torch.Tensor):
            flow_preds = [flow_preds]
        
        # 排除无效像素和极大位移
        mag = torch.sum(flow_gt**2, dim=1).sqrt()  # [B, H, W]
        valid = (valid >= 0.5) & (mag < self.max_flow)  # [B, H, W]
        
        total_loss = 0.0
        
        for i, flow_pred in enumerate(flow_preds):
            # 获取当前尺度的权重
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            
            # 将真实光流和掩码下采样到当前预测尺度
            if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
                # 下采样真实光流
                scale_factor = flow_pred.shape[-1] / flow_gt.shape[-1]
                flow_gt_scaled = F.interpolate(flow_gt, size=flow_pred.shape[-2:], 
                                             mode='bilinear', align_corners=False)
                flow_gt_scaled = flow_gt_scaled * scale_factor  # 缩放光流值
                
                # 下采样有效性掩码
                valid_scaled = F.interpolate(valid.float().unsqueeze(1), 
                                           size=flow_pred.shape[-2:], 
                                           mode='nearest').squeeze(1) > 0.5
            else:
                flow_gt_scaled = flow_gt
                valid_scaled = valid
            
            # 计算当前尺度的损失
            flow_loss = (flow_pred - flow_gt_scaled).abs()  # [B, 2, H, W]
            
            # 应用有效性掩码
            valid_expanded = valid_scaled.unsqueeze(1)  # [B, 1, H, W]
            masked_loss = (valid_expanded * flow_loss).sum() / (valid_expanded.sum() + 1e-8)
            
            # 加权累加
            total_loss += weight * masked_loss
        
        return total_loss


def end_point_error(flow_pred, flow_gt, valid_mask):
    """计算端点误差
    Args:
        flow_pred: 预测光流 [B, 2, H, W]
        flow_gt: 真实光流 [B, 2, H, W]
        valid_mask: 有效性掩码 [B, H, W]
    """
    epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1))
    epe = epe * valid_mask
    return epe.mean()


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, writer, global_step):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_epe = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        # 解包数据 (img1, img2, flow, valid)
        img1, img2, flow, valid = batch
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        # 数据移到GPU
        img1 = img1.to(device)
        img2 = img2.to(device)
        flow = flow.to(device)
        valid = valid.to(device)
        
        # 构建模型输入：[B, 2, 3, H, W]
        images = torch.stack([img1, img2], dim=1)  # [B, 2, 3, H, W]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model({"images": images})  # 传递字典格式输入
        
        # 提取多尺度光流预测用于损失计算
        if "flow_preds" in outputs and model.training:
            # 将多尺度预测转换为真实尺度
            # 处理DataParallel包装的模型
            div_flow = model.module.div_flow if hasattr(model, 'module') else model.div_flow
            flow_preds = [pred * div_flow for pred in outputs["flow_preds"]]  # 多尺度预测列表
            flow_preds.append(outputs["flows"].squeeze(1))  # 最终预测已经是真实尺度
           
        else:
            flow_preds = outputs["flows"].squeeze(1)  # 最终预测，移除时间维度
        # for pred in reversed(flow_preds):
        #     print('train:',pred.shape, pred.dtype, pred.mean())
        # print('flow:', flow.shape, flow.dtype, flow.mean())
        # 保存多尺度光流可视化图像
        if batch_idx == 1:
            plt.figure(figsize=(15, 10))
            plt.subplot(231)
            plt.imshow(flow_to_rgb(flow_preds[0][0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Flow Pred Scale 1/32')
            plt.axis('off')
            plt.subplot(232)
            plt.imshow(flow_to_rgb(flow_preds[1][0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Flow Pred Scale 1/16')
            plt.axis('off')
            plt.subplot(233)
            plt.imshow(flow_to_rgb(flow_preds[2][0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Flow Pred Scale 1/8')
            plt.axis('off')
            plt.subplot(234)
            plt.imshow(flow_to_rgb(flow_preds[3][0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Flow Pred Scale 1/4')
            plt.axis('off')
            plt.subplot(235)
            plt.imshow(flow_to_rgb(flow_preds[4][0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Flow Pred Full Scale')
            plt.axis('off')
            plt.subplot(236)
            plt.imshow(flow_to_rgb(flow[0].detach().cpu().numpy().transpose(1,2,0)))
            plt.title('Ground Truth Flow')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'flow_visualization_epoch_{epoch+1}_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()  # 关闭图像以释放内存


        
        # 计算多尺度序列损失
        loss = loss_fn(reversed(flow_preds), flow, valid)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算EPE - 使用与真实光流相同尺寸的预测

        final_flow = outputs["flows"].squeeze(1)
        epe = end_point_error(final_flow, flow, valid)
        
        # 更新统计
        epoch_loss += loss.item()
        epoch_epe += epe.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{epoch_loss/(batch_idx+1):.4f}',
            'EPE': f'{epoch_epe/(batch_idx+1):.4f}'
        })
        
        # 可视化第一个批次的光流
        if batch_idx == 0 and global_step[0] % 10 == 0:
            # 将光流转换为RGB图像
            flow_rgb = flow_to_rgb(final_flow[0].detach().cpu().permute(1, 2, 0).numpy())
            flow_rgb = torch.from_numpy(flow_rgb).permute(2, 0, 1) / 255.0
            writer.add_image('Train/Flow', flow_rgb, global_step[0])
        
        global_step[0] += 1
    
    avg_loss = epoch_loss / num_batches
    avg_epe = epoch_epe / num_batches
    
    # 记录到TensorBoard
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/EPE', avg_epe, epoch)
    
    return avg_loss, avg_epe


def validate(model, val_loader, loss_fn, device, epoch, writer):
    """
    验证模型性能，参考RAFT evaluate.py的评估方式
    
    计算详细的光流评估指标，包括EPE和多阈值像素准确率
    
    Args:
        model: 待验证的模型
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        device: 计算设备
        epoch: 当前训练轮次
        writer: TensorBoard写入器
        
    Returns:
        avg_loss: 平均验证损失
        avg_epe: 平均端点误差
        metrics: 详细评估指标字典
    """
    model.eval()
    val_loss = []
    num_batches = len(val_loader)
    
    # === 收集所有EPE值用于详细统计 ===
    epe_list = []  # 存储所有像素的EPE值
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # === 1. 数据预处理 ===
            # 解包数据 (img1, img2, flow, valid)
            img1, img2, flow, valid = batch
            img1 = img1 / 255.0
            img2 = img2 / 255.0
            # 将数据移到GPU
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow = flow.to(device)
            valid = valid.to(device)
            
            # 构建模型输入：[B, 2, 3, H, W]
            images = torch.stack([img1, img2], dim=1)  # [B, 2, 3, H, W]
            
            # === 2. 模型前向传播 ===
            outputs = model({"images": images})  # 传递字典格式输入
            
            # 提取光流预测
          
            flow_pred = outputs["flows"].squeeze(1)  # 最终预测，移除时间维度
            flow_preds = [flow_pred]
            
            # === 3. 计算损失 ===
            loss = loss_fn(flow_preds, flow, valid)
            val_loss.append(loss.item())
            
            # === 4. 计算端点误差(EPE) ===
            # 参考evaluate.py的EPE计算方式
            # 逐像素计算EPE，不进行平均化
            epe = torch.sqrt(torch.sum((flow_pred - flow) ** 2, dim=1))  # [B, H, W]
            epe = epe * valid  # 应用有效性掩码
                
            # 展平并添加到列表中，只保留有效像素的EPE值
            for b in range(epe.shape[0]):
                valid_epe = epe[b][valid[b] > 0.5]  # 只取有效像素
                if len(valid_epe) > 0:
                    epe_list.append(valid_epe.detach().cpu().numpy())
            
            # === 5. 更新进度条显示 ===
            # 计算当前的平均EPE用于显示
            if epe_list:
                current_epe = np.concatenate(epe_list).mean()
            else:
                current_epe = 0.0
                
            pbar.set_postfix({
                'Loss': f'{np.array(val_loss).mean():.4f}',
                'EPE': f'{current_epe:.4f}'
            })
            
            # === 6. 可视化第一个批次的光流 ===
            if batch_idx == 0:
                # 将光流转换为RGB图像进行可视化
                flow_rgb = flow_to_rgb(flow_pred[0].detach().cpu().permute(1, 2, 0).numpy())
                flow_rgb = torch.from_numpy(flow_rgb).permute(2, 0, 1) / 255.0
                writer.add_image('Val/Flow_Pred', flow_rgb, epoch)
                
                # 同时可视化真实光流
                flow_gt_rgb = flow_to_rgb(flow[0].detach().cpu().permute(1, 2, 0).numpy())
                flow_gt_rgb = torch.from_numpy(flow_gt_rgb).permute(2, 0, 1) / 255.0
                writer.add_image('Val/Flow_GT', flow_gt_rgb, epoch)
    
    # === 7. 计算详细评估指标 ===
    # 将所有EPE值合并，参考evaluate.py的统计方式
    epe_all = np.concatenate(epe_list)  # shape=[total_pixels]
    
    # 计算各种评估指标
    avg_epe = np.mean(epe_all)                    # 平均端点误差
    px1_acc = np.mean(epe_all < 1.0)              # 1像素准确率
    px3_acc = np.mean(epe_all < 3.0)              # 3像素准确率  
    px5_acc = np.mean(epe_all < 5.0)              # 5像素准确率
    avg_loss = np.mean(val_loss)                  # 平均损失
    
    # 构建详细指标字典
    metrics = {
        'epe': avg_epe,
        '1px': px1_acc,
        '3px': px3_acc,
        '5px': px5_acc,
        'loss': avg_loss
    }
    
    # === 8. 记录到TensorBoard ===
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/EPE', avg_epe, epoch)
    writer.add_scalar('Val/1px_Acc', px1_acc, epoch)
    writer.add_scalar('Val/3px_Acc', px3_acc, epoch)
    writer.add_scalar('Val/5px_Acc', px5_acc, epoch)
    
    # === 9. 打印验证结果 ===
    print(f"Validation Epoch {epoch+1} - Loss: {avg_loss:.4f}, EPE: {avg_epe:.4f}, "
          f"1px: {px1_acc:.4f}, 3px: {px3_acc:.4f}, 5px: {px5_acc:.4f}")
    
    return avg_loss, avg_epe, metrics


def main():
    parser = argparse.ArgumentParser(description='LiteFlowNet3 Simple Training')
    parser.add_argument('--data_dir', type=str, default='/home/redpine/share2/dataset/MPI-Sintel-complete/', help='Sintel数据集路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[384, 512], help='裁剪尺寸')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1,2], help='GPU设备')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./logs/checkpoints', help='保存目录')
    parser.add_argument('--resume', action='store_true', default=False, help='从检查点恢复训练')
    #parser.add_argument('--pretrain', type=str, default='/home/redpine/share11/code/RAFT-master/liteflownet3s-sintel-89793e34.ckpt', help='预训练模型路径')
    parser.add_argument('--pretrain', type=str, default=None, help='预训练模型路径')

    
    args = parser.parse_args()
      # 初始化日志记录器
    log_file = os.path.join(args.log_dir, 'log.txt')
    logger = Logger(log_file)
    
    # 重定向stdout到logger
    original_stdout = sys.stdout
    sys.stdout = logger

    # 记录训练参数
    logger.log_args(args)

    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集
    aug_params = {'crop_size': args.crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    #things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
    sintel_clean = datasets.MpiSintel(aug_params, split='training', dstype='clean', preload_data=True, repeat=5)
    sintel_final = datasets.MpiSintel(aug_params, split='training', dstype='final', preload_data=True, repeat=5)        

    train_dataset = sintel_clean + sintel_final 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    val_dataset_sintel_clean = datasets.MpiSintel_val(split='training', dstype='clean',repeat=1)
    val_dataset_sintel_final = datasets.MpiSintel_val(split='training', dstype='final',repeat=1)
    val_loader_sintel_clean = DataLoader(val_dataset_sintel_clean, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    val_loader_sintel_final = DataLoader(val_dataset_sintel_final, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset_sintel_clean), len(val_dataset_sintel_final)}')
    
    # 创建模型
    model = liteflownet3s().to(device)
    if len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu)
    loss_fn = SequenceLoss(gamma=0.8, max_flow=400.0)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, (args.epochs-10) // 4), gamma=0.5)
    
    # 训练状态
    start_epoch = 0
    best_epe = float('inf')
    global_step = [0]  # 使用列表以便在函数中修改
    
    # 加载模型权重
    if args.resume:
        # 从检查点恢复训练
        checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f'从检查点恢复训练: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_epe = checkpoint['best_epe']
            print(f'恢复到第 {start_epoch} 轮，最佳EPE: {best_epe:.4f}')
        else:
            print(f'警告: 检查点文件不存在 {checkpoint_path}，将从头开始训练')
    elif args.pretrain and os.path.exists(args.pretrain):
        # 加载预训练模型
        print(f'加载预训练模型: {args.pretrain}')
        try:
            # 尝试加载Lightning格式的检查点
            checkpoint = torch.load(args.pretrain, map_location=device)
            if 'state_dict' in checkpoint:
                # Lightning格式
                state_dict = checkpoint['state_dict']
                # 移除'model.'前缀（如果存在）
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)
                print('预训练模型加载成功1')
            else:
                # 普通PyTorch格式
                model.load_state_dict(checkpoint, strict=False)
                print('预训练模型加载成功2')
            
        except Exception as e:
            print(f'警告: 预训练模型加载失败: {e}，将使用随机初始化')
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    print('开始训练...')
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_epe = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, writer, global_step)
        
        # 验证
        val_loss, val_epe, val_metrics = validate(model, val_loader_sintel_clean, loss_fn, device, epoch, writer)
        #val_loss2, val_epe2, val_metrics2 = validate(model, val_loader_sintel_final, loss_fn, device, epoch, writer)
        
        # 更新学习率
        scheduler.step()
        
        # 记录学习率
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        
        print(f'Epoch {epoch+1}/{args.epochs}:  Train Loss: {train_loss:.4f}, Train EPE: {train_epe:.4f}, Val Loss: {val_loss:.4f}, Val EPE: {val_epe:.4f}')
        #print(f'  Val Loss: {val_loss:.4f}, Val EPE: {val_epe:.4f}')
        #print(f'  Val Loss2: {val_loss2:.4f}, Val EPE2: {val_epe2:.4f}')
        
        # 保存最佳模型
        if val_epe < best_epe:
            best_epe = val_epe
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_epe': best_epe,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  新的最佳模型! EPE: {val_epe:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_epe': best_epe,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print('训练完成!')
    writer.close()


if __name__ == '__main__':
    main()