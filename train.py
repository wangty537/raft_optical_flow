from __future__ import print_function, division
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
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core.raft import RAFT
import evaluate
from core import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """
    RAFT序列损失函数：对所有迭代预测的光流进行加权损失计算
    
    RAFT采用迭代优化策略，每次迭代都会输出一个光流预测。
    该损失函数对所有迭代的预测进行监督，后面的迭代给予更高权重。
    
    Args:
        flow_preds: 光流预测序列，列表长度为迭代次数 [iter1, iter2, ..., iterN]
                   每个元素shape=[N, 2, H, W]
        flow_gt: 真实光流 shape=[N, 2, H, W]
        valid: 有效像素掩码 shape=[N, H, W]，值为0或1
        gamma: 指数衰减因子，控制不同迭代的权重分布
        max_flow: 最大光流阈值，过滤异常大的位移
        
    Returns:
        flow_loss: 加权平均损失值 (标量)
        metrics: 评估指标字典，包含EPE和像素准确率
    """

    n_predictions = len(flow_preds)  # 迭代次数，通常为12
    flow_loss = 0.0

    # === 1. 构建有效像素掩码 ===
    # 排除无效像素和异常大的位移
    mag = torch.sum(flow_gt**2, dim=1).sqrt()  # 计算光流幅值 shape=[N, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)  # 有效像素掩码 shape=[N, H, W]
    # valid: True表示有效像素，False表示无效像素

    # === 2. 计算序列损失 ===
    # 对每个迭代的预测计算加权损失
    for i in range(n_predictions):
        # 计算当前迭代的权重：后面的迭代权重更大
        # 例如：gamma=0.8, n_predictions=12时
        # 第1次迭代权重: 0.8^11 ≈ 0.086
        # 第12次迭代权重: 0.8^0 = 1.0
        i_weight = gamma**(n_predictions - i - 1)
        
        # 计算L1损失（绝对值误差）
        i_loss = (flow_preds[i] - flow_gt).abs()  # shape=[N, 2, H, W]
        
        # 应用有效像素掩码并计算加权损失
        # valid[:, None]将掩码从[N, H, W]扩展为[N, 1, H, W]以匹配光流维度
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    # === 3. 计算评估指标 ===
    # 使用最后一次迭代的预测计算端点误差(EPE)
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()  # shape=[N, H, W]
    # 只考虑有效像素的EPE
    epe = epe.view(-1)[valid.view(-1)]  # 展平并过滤，shape=[num_valid_pixels]

    # 计算各种评估指标
    metrics = {
        'epe': epe.mean().item(),                    # 平均端点误差
        '1px': (epe < 1).float().mean().item(),      # 1像素准确率
        '3px': (epe < 3).float().mean().item(),      # 3像素准确率  
        '5px': (epe < 5).float().mean().item(),      # 5像素准确率
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            # === 混合精度训练的梯度缩放和优化步骤 ===
            # 1. 缩放损失并反向传播
            # 由于FP16精度较低，小梯度可能下溢为0，因此需要放大损失来避免梯度消失
            scaler.scale(loss).backward()
            
            # 2. 取消梯度缩放
            # 在梯度裁剪前需要将梯度恢复到正常尺度，否则裁剪阈值会不准确
            scaler.unscale_(optimizer)                
            
            # 3. 梯度裁剪
            # 限制梯度范数不超过args.clip（默认1.0），防止梯度爆炸
            # 这对于循环神经网络（如RAFT中的GRU）特别重要
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            # 4. 执行优化步骤
            # scaler会检查梯度是否包含inf/nan，如果有则跳过这次更新
            scaler.step(optimizer)
            
            # 5. 更新学习率
            # 使用OneCycleLR调度器动态调整学习率
            scheduler.step()
            
            # 6. 更新缩放因子
            # 根据本次是否出现inf/nan来调整下次的缩放因子
            # 如果梯度正常，可能增加缩放因子；如果异常，则减少缩放因子
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default='sintel',help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', default='raft-small.pth', type=str, help="restore checkpoint")
    parser.add_argument('--small', default=True, type=bool, help='use small model')
    parser.add_argument('--validation', default='sintel', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)

    #python -u train.py --name raft-sintel --stage sintel --validation sintel --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
