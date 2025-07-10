#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LiteFlowNet3训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from liteflownet3_simple import liteflownet3s
from train_liteflownet3 import SequenceLoss

def test_training_flow():
    """测试完整的训练流程"""
    print("测试LiteFlowNet3训练流程...")
    
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = liteflownet3s().to(device)
    model.train()
    
    # 创建损失函数和优化器
    loss_fn = SequenceLoss(gamma=0.8, max_flow=400.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 模拟训练数据
    B, H, W = 2, 384, 512
    img1 = torch.randn(B, 3, H, W).to(device)
    img2 = torch.randn(B, 3, H, W).to(device)
    flow_gt = torch.randn(B, 2, H, W).to(device)
    valid = torch.ones(B, H, W).to(device)
    
    # 构建模型输入
    images = torch.stack([img1, img2], dim=1)  # [B, 2, 3, H, W]
    
    print("\n测试前向传播...")
    
    # 前向传播
    optimizer.zero_grad()
    outputs = model({"images": images})
    
    print(f"模型输出键: {list(outputs.keys())}")
    
    # 提取多尺度光流预测
    if "flow_preds" in outputs and model.training:
        flow_preds = outputs["flow_preds"]
        print(f"多尺度预测数量: {len(flow_preds)}")
        for i, flow_pred in enumerate(flow_preds):
            print(f"  层级 {i}: {flow_pred.shape}")
    else:
        flow_preds = outputs["flows"].squeeze(1)
        print(f"单尺度预测: {flow_preds.shape}")
    
    print("\n测试损失计算...")
    
    # 计算损失
    loss = loss_fn(flow_preds, flow_gt, valid)
    print(f"训练损失: {loss.item():.6f}")
    
    print("\n测试反向传播...")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
            param_count += 1
    grad_norm = grad_norm ** 0.5
    
    print(f"梯度范数: {grad_norm:.6f}")
    print(f"有梯度的参数数量: {param_count}")
    
    # 优化器步骤
    optimizer.step()
    
    print("\n测试推理模式...")
    
    # 测试推理模式
    model.eval()
    with torch.no_grad():
        outputs_eval = model({"images": images})
        print(f"推理模式输出键: {list(outputs_eval.keys())}")
        if "flows" in outputs_eval:
            print(f"推理光流形状: {outputs_eval['flows'].shape}")
    
    print("\n✅ 训练流程测试完成！所有步骤正常工作。")

if __name__ == "__main__":
    test_training_flow()