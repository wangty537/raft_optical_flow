#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无监督SimpleFlowNet训练脚本
基于train_simple_flow.py修改，使用无监督损失函数训练光流网络
"""

import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from simple_flow_net import SimpleFlowNet
from core import datasets
import evaluate
from utils import flow_viz
from utils.utils import InputPadder


from torchvision.utils import flow_to_image
def flow_to_rgb(flow):

    """
    将光流转换为可视化图像
    
    Args:
        flow: 光流数组 [H, W, 2], float, xy
    
    Returns:
        可视化光流图像 [H, W, 3]，值范围[0, 255] rgb format
    """
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    rgb = np.transpose(flow_im.numpy(), [1, 2, 0])
    #print(img.shape)
    return rgb


def warp_image(img, flow):
    """
    使用光流扭曲图像
    Args:
        img: 输入图像 [B, C, H, W]
        flow: 光流 [B, 2, H_flow, W_flow]
    Returns:
        扭曲后的图像 [B, C, H_flow, W_flow]
    """
    B, C, H_img, W_img = img.size()
    _, _, H_flow, W_flow = flow.size()
    
    # 如果图像和光流尺寸不匹配，先将图像调整到光流尺寸
    if H_img != H_flow or W_img != W_flow:
        img = F.interpolate(img, size=(H_flow, W_flow), mode='bilinear', align_corners=False)
    
    # 创建网格（基于光流的尺寸）
    xx = torch.arange(0, W_flow).view(1, -1).repeat(H_flow, 1)
    yy = torch.arange(0, H_flow).view(-1, 1).repeat(1, W_flow)
    xx = xx.view(1, 1, H_flow, W_flow).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H_flow, W_flow).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if img.is_cuda:
        grid = grid.to(img.device)
    
    # 添加光流
    vgrid = grid + flow
    
    # 归一化到[-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W_flow-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H_flow-1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(img, vgrid, align_corners=True)
    
    return output


def compute_occlusion_mask(flow_fw, flow_bw):
    """
    计算遮挡掩码
    Args:
        flow_fw: 前向光流 [B, 2, H, W]
        flow_bw: 后向光流 [B, 2, H, W]
    Returns:
        遮挡掩码 [B, 1, H, W]
    """
    # 使用前向光流扭曲后向光流
    warped_flow_bw = warp_image(flow_bw, flow_fw)
    
    # 计算前向-后向一致性
    flow_diff = flow_fw + warped_flow_bw
    flow_mag = torch.sqrt(torch.sum(flow_fw**2, dim=1, keepdim=True) + 1e-8)
    
    # 遮挡检测
    occlusion = torch.sqrt(torch.sum(flow_diff**2, dim=1, keepdim=True)) > 0.01 * flow_mag + 0.5
    
    return (~occlusion).float() #1表示可见，0表示遮挡


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
def compute_edge_aware_loss( flow, image):
    """
    计算边缘感知损失 - 在图像边缘处允许光流不连续
    
    原理：
    结合图像梯度信息调整光流平滑性约束。在图像边缘（高梯度）处，
    减少平滑性约束，允许光流不连续；在同质区域（低梯度）处，
    保持强平滑性约束。
    
    数学表达：
    edge_weight = exp(-|∇I|)  # 图像梯度越大，权重越小
    loss = edge_weight * |∇flow|  # 加权的光流梯度
    
    Args:
        flow: 光流预测 [B, 2, H, W]
        image: 输入图像 [B, 3, H, W]
    Returns:
        edge_loss: 边缘感知损失标量
    """
    # 确保图像和光流尺寸匹配
    if image.shape[-2:] != flow.shape[-2:]:
        # 将图像下采样到光流的分辨率
        image = F.interpolate(image, size=flow.shape[-2:], mode='bilinear', align_corners=False)
    
    # 计算图像梯度 - 用于检测边缘
    # image_gray: [B, 1, H, W] - 灰度图像
    image_gray = torch.mean(image, dim=1, keepdim=True)
    # img_dx: [B, 1, H, W-1] - 图像x方向梯度
    img_dx = image_gray[:, :, :, 1:] - image_gray[:, :, :, :-1]
    # img_dy: [B, 1, H-1, W] - 图像y方向梯度
    img_dy = image_gray[:, :, 1:, :] - image_gray[:, :, :-1, :]
    
    # 计算光流梯度
    # flow_dx: [B, 2, H, W-1] - 光流x方向梯度
    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    # flow_dy: [B, 2, H-1, W] - 光流y方向梯度
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    
    # 计算边缘权重 - 图像梯度越大，平滑性约束越小
    # edge_weight_x: [B, 1, H, W-1] - x方向边缘权重
    edge_weight_x = torch.exp(-torch.abs(img_dx))
    # edge_weight_y: [B, 1, H-1, W] - y方向边缘权重
    edge_weight_y = torch.exp(-torch.abs(img_dy))
    
    # 边缘感知平滑性损失 - 在边缘处减少平滑性约束
    # 在图像边缘（高梯度）处，权重接近0，平滑性约束很小
    # 在同质区域（低梯度）处，权重接近1，保持平滑性约束
    edge_loss = torch.mean(edge_weight_x * torch.abs(flow_dx)) + \
                torch.mean(edge_weight_y * torch.abs(flow_dy))
    
    return edge_loss

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
    """多尺度无监督光流损失函数"""
    def __init__(self, alpha_photo=1.0, alpha_smooth=0.1, alpha_consist=0.1, scale_weights=None):
        super().__init__()
        self.alpha_photo = alpha_photo      # 光度一致性损失权重
        self.alpha_smooth = alpha_smooth    # 平滑性损失权重
        self.alpha_consist = alpha_consist  # 前向-后向一致性损失权重
        
        # 多尺度损失权重，默认从粗到细递增
        if scale_weights is None:
            self.scale_weights = [0.32, 0.08, 0.02]  # 对应1/8, 1/4, 1/2分辨率
        else:
            self.scale_weights = scale_weights
    
    def forward(self, img1, img2, flow_preds_fw, flow_preds_bw=None):
        """
        计算多尺度无监督损失
        Args:
            img1: 第一帧图像 [B, 3, H, W]
            img2: 第二帧图像 [B, 3, H, W]
            flow_preds_fw: 前向光流预测列表 (img1->img2) [list of [B, 2, H_i, W_i]]
            flow_preds_bw: 后向光流预测列表 (img2->img1) [list of [B, 2, H_i, W_i]], 可选
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        total_photo_loss = 0.0
        total_smooth_loss = 0.0
        total_consist_loss = 0.0
        
        # 确保权重数量与预测尺度数量匹配
        num_scales = len(flow_preds_fw)
        if len(self.scale_weights) != num_scales:
            # 如果权重数量不匹配，使用均匀权重
            weights = [1.0 / num_scales] * num_scales
        else:
            weights = self.scale_weights
        
        # 对每个尺度计算损失
        for i, (flow_fw, weight) in enumerate(zip(flow_preds_fw, weights)):
            # 获取对应的后向光流（如果存在）
            flow_bw = None
            if flow_preds_bw is not None and i < len(flow_preds_bw):
                flow_bw = flow_preds_bw[i]
            
            # 将图像下采样到对应的光流分辨率
            scale_factor = flow_fw.shape[-1] / img1.shape[-1]  # 计算缩放因子
            
            if scale_factor != 1.0:
                # 下采样图像到光流分辨率
                img1_scaled = F.interpolate(img1, size=flow_fw.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                img2_scaled = F.interpolate(img2, size=flow_fw.shape[-2:], 
                                          mode='bilinear', align_corners=False)
            else:
                img1_scaled = img1
                img2_scaled = img2
            
            # 1. 光度一致性损失
            photo_loss = compute_photometric_loss(img1_scaled, img2_scaled, flow_fw)
            if flow_bw is not None:
                photo_loss += compute_photometric_loss(img2_scaled, img1_scaled, flow_bw)
            total_photo_loss += weight * photo_loss
            
            # 2. 平滑性损失
            #smooth_loss = compute_smoothness_loss(flow_fw, img1_scaled)
            smooth_loss = compute_edge_aware_loss(flow_fw, img1_scaled) 
            if flow_bw is not None:
                smooth_loss += compute_edge_aware_loss(flow_bw, img2_scaled)
            total_smooth_loss += weight * smooth_loss
            
            # 3. 前向-后向一致性损失（如果提供后向光流）
            if flow_bw is not None:
                # 计算遮挡掩码
                occlusion_mask = compute_occlusion_mask(flow_fw, flow_bw)
                
                # 在非遮挡区域计算一致性损失
                warped_flow_bw = warp_image(flow_bw, flow_fw)
                consist_loss = torch.mean(occlusion_mask * torch.abs(flow_fw + warped_flow_bw))
                total_consist_loss += weight * consist_loss
        
        # 总损失
        total_loss = (self.alpha_photo * total_photo_loss + 
                     self.alpha_smooth * total_smooth_loss + 
                     self.alpha_consist * total_consist_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'photometric': total_photo_loss.item(),
            'smoothness': total_smooth_loss.item(),
            'consistency': total_consist_loss.item() if isinstance(total_consist_loss, torch.Tensor) else total_consist_loss
        }
        
        return total_loss, loss_dict


class SimpleFlowTrainer:
    def __init__(self, args):
        self.args = args
        
        # 设备设置
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        
        # 创建模型
        self.model = SimpleFlowNet().to(self.device)
        
        # 创建多尺度无监督损失函数
        scale_weights = getattr(args, 'scale_weights', None)
        self.loss_fn = UnsupervisedLoss(
            alpha_photo=args.alpha_photo,
            alpha_smooth=args.alpha_smooth, 
            alpha_consist=args.alpha_consist,
            scale_weights=scale_weights
        )
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=args.log_dir)
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 加载检查点或预训练模型
        self.load_checkpoint()
        
        # 创建数据集
        self.train_loader, self.val_loader = self.get_data_loaders()
        
    def get_data_loaders(self):
        """创建数据加载器"""
        # 数据增强参数
        aug_params = {
            'crop_size': self.args.crop_size, 
            'min_scale': -0.2, 
            'max_scale': 0.6, 
            'do_flip': True
        }
        
        # 训练数据集
        sintel_clean = datasets.MpiSintel(aug_params, split='training', dstype='clean', preload_data=True, repeat=5)
        sintel_final = datasets.MpiSintel(aug_params, split='training', dstype='final', preload_data=True, repeat=5)
        train_dataset = 1*sintel_clean + 1*sintel_final
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            pin_memory=False, 
            shuffle=True, 
            num_workers=4, 
            drop_last=True
        )
        
        # 验证数据集
        val_dataset = datasets.MpiSintel_val(split='training', dstype='clean')
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}')
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_photo_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_consist_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 解包数据 (img1, img2, flow, valid) - 无监督训练不使用flow和valid
            img1, img2, flow, valid = batch
            
            # 数据移到GPU并归一化到[0,1]范围
            img1 = img1.to(self.device) / 255.0
            img2 = img2.to(self.device) / 255.0
            
            # 前向传播 - 计算前向光流
            self.optimizer.zero_grad()
            
            # 计算前向光流：img1 -> img2
            flow_preds_fw = self.model(img1, img2)
            
            # 计算后向光流：img2 -> img1
            flow_preds_bw = self.model(img2, img1)
            
            # 计算多尺度无监督损失
            loss, loss_dict = self.loss_fn(img1, img2, flow_preds_fw, flow_preds_bw)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 更新统计
            epoch_loss += loss_dict['total']
            epoch_photo_loss += loss_dict['photometric']
            epoch_smooth_loss += loss_dict['smoothness']
            epoch_consist_loss += loss_dict['consistency']
            
            # 可视化第一个批次的光流
            if batch_idx == 0 == 0:
                # 获取最高分辨率的光流用于可视化
                flow_fw = flow_preds_fw[-1]  # 最高分辨率前向光流
                flow_bw = flow_preds_bw[-1]  # 最高分辨率后向光流
                
                # 保存光流图像
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # 预测光流的三个尺度
                for i, flow_pred in enumerate(flow_preds_fw[:3]):
                    flow_np = flow_pred[0].detach().cpu().permute(1, 2, 0).numpy()
                    flow_rgb = flow_to_rgb(flow_np)
                    axes[0, i].imshow(flow_rgb)
                    axes[0, i].set_title(f'Forward Flow Scale {i+1}')
                    axes[0, i].axis('off')
                
                # 后向光流
                # flow_bw_np = flow_bw[0].detach().cpu().permute(1, 2, 0).numpy()
                # flow_bw_rgb = flow_to_rgb(flow_bw_np)
                # axes[1, 0].imshow(flow_bw_rgb)
                # axes[1, 0].set_title('Backward Flow')
                # axes[1, 0].axis('off')

                flow_bw_np = flow[0].detach().cpu().permute(1, 2, 0).numpy()
                flow_bw_rgb = flow_to_rgb(flow_bw_np)
                axes[1, 0].imshow(flow_bw_rgb)
                axes[1, 0].set_title('gt Flow')
                axes[1, 0].axis('off')
                
                # 原始图像和扭曲图像
                img1_np = img1[0].detach().cpu().permute(1, 2, 0).numpy()
                img1_np = np.clip(img1_np, 0, 1)
                axes[1, 1].imshow(img1_np)
                axes[1, 1].set_title('Image 1')
                axes[1, 1].axis('off')
                
                warped_img2 = warp_image(img2, flow_fw)
                warped_img2_np = warped_img2[0].detach().cpu().permute(1, 2, 0).numpy()
                warped_img2_np = np.clip(warped_img2_np, 0, 1)
                axes[1, 2].imshow(warped_img2_np)
                axes[1, 2].set_title('Warped Image 2')
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{self.args.log_dir}/train_flow_epoch_{epoch+1}_step_{self.global_step}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # TensorBoard记录
                flow_rgb_tensor = torch.from_numpy(flow_to_rgb(flow_fw[0].detach().cpu().permute(1, 2, 0).numpy())).permute(2, 0, 1) / 255.0
                self.writer.add_image('Train/Flow_Forward', flow_rgb_tensor, self.global_step)
                
                flow_bw_rgb_tensor = torch.from_numpy(flow_bw_rgb).permute(2, 0, 1) / 255.0
                self.writer.add_image('Train/Flow_Backward', flow_bw_rgb_tensor, self.global_step)
                
                self.writer.add_image('Train/Img1_Original', img1[0], self.global_step)
                self.writer.add_image('Train/Img2_Warped', warped_img2[0], self.global_step)
            
            self.global_step += 1
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss_dict["total"]:.4f}, '
                      f'Photo: {loss_dict["photometric"]:.4f}, '
                      f'Smooth: {loss_dict["smoothness"]:.4f}, '
                      f'Consist: {loss_dict["consistency"]:.4f}')
        
        avg_loss = epoch_loss / num_batches
        avg_photo_loss = epoch_photo_loss / num_batches
        avg_smooth_loss = epoch_smooth_loss / num_batches
        avg_consist_loss = epoch_consist_loss / num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Photometric_Loss', avg_photo_loss, epoch)
        self.writer.add_scalar('Train/Smoothness_Loss', avg_smooth_loss, epoch)
        self.writer.add_scalar('Train/Consistency_Loss', avg_consist_loss, epoch)
        
        return avg_loss, avg_photo_loss, avg_smooth_loss, avg_consist_loss
    
    def validate(self, epoch):
        """验证模型性能"""
        self.model.eval()
        val_loss = 0.0
        val_photo_loss = 0.0
        val_smooth_loss = 0.0
        val_consist_loss = 0.0
        num_batches = len(self.val_loader)
        
        # 收集所有EPE值用于详细统计
        epe_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 解包数据
                img1, img2, flow_gt, valid = batch
                
                # 将数据移到GPU并归一化到[0,1]范围
                img1 = img1.to(self.device) / 255.0
                img2 = img2.to(self.device) / 255.0
                flow_gt = flow_gt.to(self.device)
                valid = valid.to(self.device)
                
                # 前向传播
                flow_preds_fw = self.model(img1, img2)
                
                flow_preds_bw = self.model(img2, img1)
                
                # 计算多尺度无监督损失
                loss, loss_dict = self.loss_fn(img1, img2, flow_preds_fw, flow_preds_bw)
                
                # 获取最高分辨率的光流用于EPE计算
                flow_fw = flow_preds_fw[-1]
                
                val_loss += loss_dict['total']
                val_photo_loss += loss_dict['photometric']
                val_smooth_loss += loss_dict['smoothness']
                val_consist_loss += loss_dict['consistency']
                
                # 计算EPE评估指标（使用真实光流标签）
                for i in range(flow_fw.shape[0]):
                    flow_pred_i = flow_fw[i]
                    flow_gt_i = flow_gt[i]
                    valid_i = valid[i]
                    
                    # 如果预测光流和真实光流尺寸不匹配，上采样预测光流
                    if flow_pred_i.shape[-2:] != flow_gt_i.shape[-2:]:
                        flow_pred_i = F.interpolate(
                            flow_pred_i.unsqueeze(0), 
                            size=flow_gt_i.shape[-2:], 
                            mode='bilinear', 
                            align_corners=True
                        ).squeeze(0)
                        # 调整光流数值以匹配尺寸变化
                        scale_h = flow_gt_i.shape[-2] / flow_fw.shape[-2]
                        scale_w = flow_gt_i.shape[-1] / flow_fw.shape[-1]
                        flow_pred_i[0] *= scale_w  # x方向光流
                        flow_pred_i[1] *= scale_h  # y方向光流
                    
                    # 计算每个像素的EPE
                    epe = torch.sum((flow_pred_i - flow_gt_i)**2, dim=0).sqrt()
                    
                    # 只在有效像素处计算EPE
                    valid_mask = valid_i >= 0.5
                    if valid_mask.sum() > 0:
                        epe_valid = epe[valid_mask]
                        epe_list.append(epe_valid.cpu().numpy())
                
                # 可视化第一个批次的光流
                if batch_idx == 0:
                    # 获取最高分辨率的后向光流用于可视化
                    flow_bw = flow_preds_bw[-1]
                    
                    flow_rgb = flow_to_rgb(flow_fw[0].detach().cpu().permute(1, 2, 0).numpy())
                    flow_rgb_tensor = torch.from_numpy(flow_rgb).permute(2, 0, 1) / 255.0
                    self.writer.add_image('Val/Flow_Forward', flow_rgb_tensor, epoch)
                    
                    # 可视化真实光流
                    flow_gt_rgb = flow_to_rgb(flow_gt[0].detach().cpu().permute(1, 2, 0).numpy())
                    flow_gt_rgb_tensor = torch.from_numpy(flow_gt_rgb).permute(2, 0, 1) / 255.0
                    self.writer.add_image('Val/Flow_GT', flow_gt_rgb_tensor, epoch)
                    
                    # 可视化扭曲效果
                    warped_img2 = warp_image(img2, flow_fw)
                    self.writer.add_image('Val/Img1_Original', img1[0], epoch)
                    self.writer.add_image('Val/Img2_Warped', warped_img2[0], epoch)
        
        avg_loss = val_loss / num_batches
        avg_photo_loss = val_photo_loss / num_batches
        avg_smooth_loss = val_smooth_loss / num_batches
        avg_consist_loss = val_consist_loss / num_batches
        
        # 计算各种评估指标
        if len(epe_list) > 0:
            epe_all = np.concatenate(epe_list)
            avg_epe = np.mean(epe_all)
            px1_acc = np.mean(epe_all < 1.0)
            px3_acc = np.mean(epe_all < 3.0)
            px5_acc = np.mean(epe_all < 5.0)
        else:
            avg_epe = float('inf')
            px1_acc = px3_acc = px5_acc = 0.0
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Photometric_Loss', avg_photo_loss, epoch)
        self.writer.add_scalar('Val/Smoothness_Loss', avg_smooth_loss, epoch)
        self.writer.add_scalar('Val/Consistency_Loss', avg_consist_loss, epoch)
        self.writer.add_scalar('Val/EPE', avg_epe, epoch)
        self.writer.add_scalar('Val/1px_Acc', px1_acc, epoch)
        self.writer.add_scalar('Val/3px_Acc', px3_acc, epoch)
        self.writer.add_scalar('Val/5px_Acc', px5_acc, epoch)
        
        print(f"Validation Epoch {epoch+1} - Total Loss: {avg_loss:.4f}, "
              f"Photo: {avg_photo_loss:.4f}, Smooth: {avg_smooth_loss:.4f}, "
              f"Consist: {avg_consist_loss:.4f}")
        print(f"EPE: {avg_epe:.4f}, 1px: {px1_acc:.4f}, 3px: {px3_acc:.4f}, 5px: {px5_acc:.4f}")
        
        return avg_loss, avg_photo_loss, avg_smooth_loss, avg_consist_loss, avg_epe
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_model.pth'))
            print(f'新的最佳模型已保存! 验证损失: {self.best_loss:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def load_checkpoint(self):
        """加载检查点"""
        if self.args.resume:
            checkpoint_path = os.path.join(self.args.save_dir, 'latest_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print(f'从检查点恢复训练: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['best_loss']
                self.global_step = checkpoint['global_step']
                print(f'恢复到第 {self.start_epoch} 轮，最佳损失: {self.best_loss:.4f}')
            else:
                print(f'警告: 检查点文件不存在 {checkpoint_path}，将从头开始训练')
        elif self.args.pretrain and os.path.exists(self.args.pretrain):
            print(f'加载预训练模型: {self.args.pretrain}')
            try:
                checkpoint = torch.load(self.args.pretrain, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                print('预训练模型加载成功')
            except Exception as e:
                print(f'警告: 预训练模型加载失败: {e}，将使用随机初始化')
    
    def train(self):
        """主训练循环"""
        print('开始无监督训练...')
        print(f'损失权重 - 光度: {self.args.alpha_photo}, 平滑: {self.args.alpha_smooth}, 一致性: {self.args.alpha_consist}')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_photo, train_smooth, train_consist = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_photo, val_smooth, val_consist, val_epe = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录学习率
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Photo: {train_photo:.4f}, '
                  f'Smooth: {train_smooth:.4f}, Consist: {train_consist:.4f}')
            print(f'  Val - Total: {val_loss:.4f}, Photo: {val_photo:.4f}, '
                  f'Smooth: {val_smooth:.4f}, Consist: {val_consist:.4f}')
            
            # 保存最佳模型（基于验证损失）
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
        
        print('无监督训练完成!')
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='SimpleFlowNet Unsupervised Training')
    parser.add_argument('--data_dir', type=str, default='/home/redpine/share2/dataset/MPI-Sintel-complete/', help='Sintel数据集路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256], help='裁剪尺寸')
    parser.add_argument('--gpu', type=int, default=2, help='GPU设备')
    parser.add_argument('--log_dir', type=str, default='./logs_simple_flow_unsupervised', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./logs_simple_flow_unsupervised/checkpoints', help='保存目录')
    parser.add_argument('--resume', action='store_true', default=False, help='从检查点恢复训练')
    parser.add_argument('--pretrain', type=str, default='/home/redpine/share11/code/RAFT-master/checkpoints_simple_flow/best.pth', help='预训练模型路径')
    
    # 无监督损失权重参数
    parser.add_argument('--alpha_photo', type=float, default=1.0, help='光度一致性损失权重')
    parser.add_argument('--alpha_smooth', type=float, default=0.1, help='平滑性损失权重')
    parser.add_argument('--alpha_consist', type=float, default=0.5, help='前向-后向一致性损失权重')
    parser.add_argument('--scale_weights', type=float, nargs='+', default=None, 
                       help='多尺度损失权重列表，例如: --scale_weights 0.32 0.08 0.02')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = SimpleFlowTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()