#!/usr/bin/env python3
"""
简单光流网络训练脚本

这个脚本演示了如何训练简单高效的光流估计网络
包含完整的训练循环、验证和模型保存功能
"""

import os
import sys
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from simple_flow_net import create_simple_flow_model, create_simple_flow_loss
import matplotlib.pyplot as plt
# 导入数据集（假设使用现有的数据集类）
try:
    from core import datasets
except ImportError:
    print("警告: 无法导入数据集模块，请确保core/datasets.py存在")
    datasets = None
from torchvision.utils import flow_to_image
def flow_to_rgb(flow):
    """将光流转换为RGB图像用于可视化"""
    # h, w = flow.shape[:2]
    # fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # ang = np.arctan2(fy, fx) + np.pi
    # v = np.sqrt(fx * fx + fy * fy)
    
    # hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # hsv[:, :, 0] = ang * (180 / np.pi / 2)
    # hsv[:, :, 1] = 255
    # hsv[:, :, 2] = np.minimum(v * 4, 255)
    
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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

class SimpleFlowTrainer:
    """
    简单光流网络训练器
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_simple_flow_model(
            input_channels=3, 
            feature_dim=args.feature_dim
        ).to(self.device)
        
        # 创建损失函数
        self.loss_fn = create_simple_flow_loss(
            weights=args.loss_weights,
            smooth_weight=args.smooth_weight,
            edge_weight=args.edge_weight
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=args.lr_step, 
            gamma=args.lr_gamma
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=args.log_dir)
        
        # 训练状态
        self.start_epoch = 0
        self.best_epe = float('inf')
        self.global_step = 0
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_datasets(self):
        """
        创建训练和验证数据集
        """
        if datasets is None:
            # 如果没有数据集模块，创建虚拟数据集用于演示
            print("使用虚拟数据集进行演示")
            return self.create_dummy_datasets()
        
        # 数据增强参数
        aug_params = {
            'crop_size': self.args.crop_size, 
            'min_scale': -0.2, 
            'max_scale': 0.6, 
            'do_flip': True
        }
        
        # 训练数据集
        if self.args.dataset == 'sintel':
            train_dataset = datasets.MpiSintel(
                aug_params, 
                split='training', 
                dstype='clean',
                preload_data=True,
                repeat=5
            ) + datasets.MpiSintel(
                aug_params, 
                split='training', 
                dstype='final',
                preload_data=True,
                repeat=5
            )
            val_dataset = datasets.MpiSintel_val(
                split='training', 
                dstype='clean'
            )
        elif self.args.dataset == 'chairs':
            train_dataset = datasets.FlyingChairs(
                aug_params, 
                split='training'
            )
            val_dataset = datasets.FlyingChairs(
                split='validation'
            )
        else:
            raise ValueError(f"不支持的数据集: {self.args.dataset}")
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def create_dummy_datasets(self):
        """
        创建虚拟数据集用于演示
        """
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000, image_size=(256, 256)):
                self.size = size
                self.image_size = image_size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                H, W = self.image_size
                img1 = torch.randn(3, H, W) * 255
                img2 = torch.randn(3, H, W) * 255
                flow = torch.randn(2, H, W) * 10
                valid = torch.ones(H, W)
                return img1, img2, flow, valid
        
        train_dataset = DummyDataset(size=1000)
        val_dataset = DummyDataset(size=100)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_epe = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据预处理
            img1, img2, flow_gt, valid = batch
            img1 = img1.to(self.device) / 255.0
            img2 = img2.to(self.device) / 255.0
            flow_gt = flow_gt.to(self.device)
            valid = valid.to(self.device)
            
            # 前向传播
            flow_preds = self.model(img1, img2)
            # 保存光流图像
            if batch_idx == 1:
                plt.figure(figsize=(12, 8))
                
                # SimpleFlowNet返回3个尺度的光流预测: [1/8, 1/4, 1/2]
                plt.subplot(221)
                plt.imshow(flow_to_rgb(flow_preds[0][0].detach().cpu().numpy().transpose(1,2,0)))
                plt.title('Flow Pred Scale 1/8 (Coarse)')
                plt.axis('off')
                
                plt.subplot(222)
                plt.imshow(flow_to_rgb(flow_preds[1][0].detach().cpu().numpy().transpose(1,2,0)))
                plt.title('Flow Pred Scale 1/4 (Medium)')
                plt.axis('off')
                
                plt.subplot(223)
                plt.imshow(flow_to_rgb(flow_preds[2][0].detach().cpu().numpy().transpose(1,2,0)))
                plt.title('Flow Pred Scale 1/2 (Fine)')
                plt.axis('off')
                
                plt.subplot(224)
                plt.imshow(flow_to_rgb(flow_gt[0].detach().cpu().numpy().transpose(1,2,0)))
                plt.title('Ground Truth Flow')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'flow_simple_visualization_epoch_{epoch+1}_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
                plt.close()  # 关闭图像以释放内存

            # 计算损失
            total_loss, loss_dict = self.loss_fn(
                flow_preds, flow_gt, valid, img1
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 统计
            epoch_loss += total_loss.item()
            if 'epe' in loss_dict:
                epoch_epe += loss_dict['epe'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'EPE loss': f'{epoch_epe/(batch_idx+1):.4f}'
            })
            
            # TensorBoard记录
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss', total_loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Train/{key.capitalize()}', value.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches
        avg_epe = epoch_epe / num_batches
        
        return avg_loss, avg_epe
    
    def validate(self, val_loader, epoch):
        """
        验证模型
        """
        self.model.eval()
        val_loss = 0.0
        epe_loss = 0.0
        epe_list = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
            
            for batch_idx, batch in enumerate(pbar):
                # 数据预处理
                img1, img2, flow_gt, valid = batch
                img1 = img1.to(self.device) / 255.0
                img2 = img2.to(self.device) / 255.0
                flow_gt = flow_gt.to(self.device)
                valid = valid.to(self.device)
                
                # 前向传播
                flow_preds = self.model(img1, img2)
                
                # 计算损失
                total_loss, loss_dict = self.loss_fn(
                    flow_preds, flow_gt, valid, img1
                )
                
                val_loss += total_loss.item()
                epe_loss += loss_dict['epe']
                
                # 计算EPE
                flow_pred = flow_preds[-1]  # 最高分辨率预测
                
                # 确保flow_pred与flow_gt尺寸匹配
                if flow_pred.shape[2:] != flow_gt.shape[2:]:
                    flow_pred = F.interpolate(
                        flow_pred, 
                        size=flow_gt.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # 调整光流值以匹配新的尺寸
                    scale_h = flow_gt.shape[2] / flow_preds[-1].shape[2]
                    scale_w = flow_gt.shape[3] / flow_preds[-1].shape[3]
                    flow_pred[:, 0] *= scale_w
                    flow_pred[:, 1] *= scale_h
                
                epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1))
                epe = epe * valid
                
                # 收集有效像素的EPE
                for b in range(epe.shape[0]):
                    valid_epe = epe[b][valid[b] > 0.5]
                    if len(valid_epe) > 0:
                        epe_list.append(valid_epe.cpu().numpy())
                
                # 更新进度条
                if epe_list:
                    current_epe = np.concatenate(epe_list).mean()
                else:
                    current_epe = 0.0
                
                pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f},{epe_loss/(batch_idx+1):.4f}',
                    'EPE': f'avg:{current_epe:.4f}, cur:{valid_epe.cpu().numpy().mean():.4f}'
                })
        
        # 计算平均指标
        avg_loss = val_loss / len(val_loader)
        if epe_list:
            epe_all = np.concatenate(epe_list)
            avg_epe = np.mean(epe_all)
            px1_acc = np.mean(epe_all < 1.0)
            px3_acc = np.mean(epe_all < 3.0)
            px5_acc = np.mean(epe_all < 5.0)
        else:
            avg_epe = float('inf')
            px1_acc = px3_acc = px5_acc = 0.0
        
        # TensorBoard记录
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/EPE', avg_epe, epoch)
        self.writer.add_scalar('Val/1px_Acc', px1_acc, epoch)
        self.writer.add_scalar('Val/3px_Acc', px3_acc, epoch)
        self.writer.add_scalar('Val/5px_Acc', px5_acc, epoch)
        
        return avg_loss, avg_epe, {
            'epe': avg_epe,
            '1px': px1_acc,
            '3px': px3_acc,
            '5px': px5_acc
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_epe': self.best_epe,
            'global_step': self.global_step,
            'args': self.args
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'latest.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'best.pth'))
        
        # 定期保存
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(self.args.save_dir, f'epoch_{epoch+1}.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        """
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return
        
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_epe = checkpoint['best_epe']
        self.global_step = checkpoint['global_step']
        
        print(f"恢复到第 {self.start_epoch} 轮，最佳EPE: {self.best_epe:.4f}")
    
    def train(self):
        """
        主训练循环
        """
        # 创建数据集
        train_loader, val_loader = self.create_datasets()
        
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        
        # 加载检查点
        if self.args.resume:
            self.load_checkpoint(os.path.join(self.args.save_dir, 'latest.pth'))
        
        print("开始训练...")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_epe = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_epe, val_metrics = self.validate(val_loader, epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录学习率
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{self.args.epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, EPE: {train_epe:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, EPE: {val_epe:.4f}, "
                  f"1px: {val_metrics['1px']:.4f}, 3px: {val_metrics['3px']:.4f}, 5px: {val_metrics['5px']:.4f}")
            
            # 保存最佳模型
            is_best = val_epe < self.best_epe
            if is_best:
                self.best_epe = val_epe
                print(f"  新的最佳模型! EPE: {val_epe:.4f}")
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
        
        print("训练完成!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='简单光流网络训练')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='sintel', 
                       choices=['sintel', 'chairs', 'dummy'], help='数据集类型')
    parser.add_argument('--data_dir', type=str, default='/home/redpine/share2/dataset/MPI-Sintel-complete/', help='数据集路径')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256], help='裁剪尺寸')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=64, help='特征维度')
    
    # 损失函数参数
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.32, 0.08, 0.02], 
                       help='多尺度损失权重')
    parser.add_argument('--smooth_weight', type=float, default=0.1, help='平滑性损失权重')
    parser.add_argument('--edge_weight', type=float, default=0.1, help='边缘感知损失权重')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--lr_step', type=int, default=50, help='学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='学习率衰减因子')
    
    # 系统参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 保存参数
    parser.add_argument('--log_dir', type=str, default='./logs_simple_flow', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_simple_flow', help='保存目录')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = SimpleFlowTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()