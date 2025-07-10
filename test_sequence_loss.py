#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多尺度序列损失函数
"""

import torch
import torch.nn.functional as F
from train_liteflownet3 import SequenceLoss

def test_sequence_loss():
    """测试SequenceLoss函数"""
    print("测试多尺度序列损失函数...")
    
    # 创建损失函数
    loss_fn = SequenceLoss(gamma=0.8, max_flow=400.0)
    
    # 模拟多尺度光流预测 (从粗到细)
    B, H, W = 2, 384, 512
    
    # 多尺度预测列表 (模拟LiteFlowNet3的输出)
    flow_preds = [
        torch.randn(B, 2, H//32, W//32),  # 1/32尺度
        torch.randn(B, 2, H//16, W//16),  # 1/16尺度  
        torch.randn(B, 2, H//8, W//8),    # 1/8尺度
        torch.randn(B, 2, H//4, W//4),    # 1/4尺度
        torch.randn(B, 2, H, W),          # 原始尺度
    ]
    
    # 真实光流和有效性掩码
    flow_gt = torch.randn(B, 2, H, W)
    valid = torch.ones(B, H, W)
    
    # 测试1: 多尺度输入
    print("测试1: 多尺度光流预测列表")
    loss_multi = loss_fn(flow_preds, flow_gt, valid)
    print(f"多尺度损失: {loss_multi.item():.6f}")
    
    # 测试2: 单尺度输入
    print("\n测试2: 单尺度光流预测")
    loss_single = loss_fn(flow_preds[-1], flow_gt, valid)
    print(f"单尺度损失: {loss_single.item():.6f}")
    
    # 测试3: 验证损失计算是否合理
    print("\n测试3: 验证损失值范围")
    assert loss_multi > 0, "损失应该为正值"
    assert loss_single > 0, "损失应该为正值"
    print("✓ 损失值范围正常")
    
    # 测试4: 验证梯度计算
    print("\n测试4: 验证梯度计算")
    for i, flow_pred in enumerate(flow_preds):
        flow_pred.requires_grad_(True)
    
    loss_multi = loss_fn(flow_preds, flow_gt, valid)
    loss_multi.backward()
    
    for i, flow_pred in enumerate(flow_preds):
        assert flow_pred.grad is not None, f"第{i}层光流预测应该有梯度"
        assert not torch.isnan(flow_pred.grad).any(), f"第{i}层梯度不应包含NaN"
    
    print("✓ 梯度计算正常")
    
    print("\n✅ 所有测试通过！多尺度序列损失函数工作正常。")

if __name__ == "__main__":
    test_sequence_loss()