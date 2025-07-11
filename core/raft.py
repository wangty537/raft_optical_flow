import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

# 尝试导入混合精度训练的autocast，如果PyTorch版本过低则使用dummy版本
try:
    autocast = torch.cuda.amp.autocast
except:
    # PyTorch < 1.6版本的dummy autocast
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    """
    RAFT (Recurrent All-Pairs Field Transforms) 光流估计网络
    
    RAFT通过迭代更新的方式估计光流，主要包含三个核心组件：
    1. 特征编码器(Feature Encoder): 提取图像特征
    2. 上下文编码器(Context Encoder): 提取上下文信息
    3. 更新模块(Update Block): 迭代优化光流估计
    
    Args:
        args: 配置参数对象，包含模型配置信息
    """
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        # 根据模型大小配置不同的参数
        if args.small:
            # 小模型配置 - 适用于资源受限的场景
            self.hidden_dim = hdim = 96      # GRU隐藏层维度
            self.context_dim = cdim = 64     # 上下文特征维度
            args.corr_levels = 4             # 相关性金字塔层数
            args.corr_radius = 3             # 相关性搜索半径
        else:
            # 标准模型配置 - 更高精度
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        # 设置默认参数
        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # 构建网络组件：特征网络、上下文网络和更新模块
        if args.small:
            # 小模型网络结构
            # 特征编码器：输入图像 -> 特征图 [N, 128, H/8, W/8]
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            # 上下文编码器：输入图像 -> 上下文特征 [N, 160, H/8, W/8] (96+64)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            # 更新模块：迭代更新光流
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            # 标准模型网络结构
            # 特征编码器：输入图像 -> 特征图 [N, 256, H/8, W/8]
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            # 上下文编码器：输入图像 -> 上下文特征 [N, 256, H/8, W/8] (128+128)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            # 更新模块：迭代更新光流
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        """
        冻结所有BatchNorm层，将其设置为评估模式
        在微调或特定训练策略中使用，防止BatchNorm参数更新
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """
        初始化光流坐标网格
        
        光流表示为两个坐标网格的差值：flow = coords1 - coords0
        - coords0: 源坐标网格（固定不变）
        - coords1: 目标坐标网格（会被迭代更新）
        
        Args:
            img: 输入图像 shape=[N, C, H, W]
            
        Returns:
            coords0: 源坐标网格 shape=[N, 2, H//8, W//8]
            coords1: 目标坐标网格 shape=[N, 2, H//8, W//8]
        """
        N, C, H, W = img.shape
        # 创建1/8分辨率的坐标网格（RAFT在1/8分辨率下计算光流）
        coords0 = coords_grid(N, H//8, W//8, device=img.device)  # shape: [N, 2, H//8, W//8]
        coords1 = coords_grid(N, H//8, W//8, device=img.device)  # shape: [N, 2, H//8, W//8]

        # 光流计算公式：flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """
        使用可学习的凸组合将光流从1/8分辨率上采样到原始分辨率
        
        该方法比简单的双线性插值更精确，通过学习的掩码权重
        对3x3邻域进行加权组合来实现上采样
        
        Args:
            flow: 低分辨率光流 shape=[N, 2, H//8, W//8]
            mask: 上采样掩码 shape=[N, 64*9, H//8, W//8]
            
        Returns:
            up_flow: 上采样后的光流 shape=[N, 2, H, W]
        """
        N, _, H, W = flow.shape  # flow shape: [N, 2, H//8, W//8]
        
        # 重塑掩码：[N, 64*9, H//8, W//8] -> [N, 1, 9, 8, 8, H//8, W//8]
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        # 在9个邻域方向上进行softmax归一化
        mask = torch.softmax(mask, dim=2)  # shape: [N, 1, 9, 8, 8, H//8, W//8]

        # 将光流放大8倍并展开3x3邻域
        up_flow = F.unfold(8 * flow, [3,3], padding=1)  # shape: [N, 2*9, H//8*W//8]
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)     # shape: [N, 2, 9, 1, 1, H//8, W//8]

        # 使用掩码权重进行加权求和
        up_flow = torch.sum(mask * up_flow, dim=2)       # shape: [N, 2, 8, 1, 8, H//8, W//8]
        # 重新排列维度以获得正确的空间布局
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)      # shape: [N, 2, 8, 8, H//8, W//8]
        # 重塑为最终的上采样结果
        return up_flow.reshape(N, 2, 8*H, 8*W)           # shape: [N, 2, H, W]


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """
        RAFT前向传播：估计两帧图像之间的光流
        
        Args:
            image1: 第一帧图像 shape=[N, 3, H, W], 像素值范围[0, 255]
            image2: 第二帧图像 shape=[N, 3, H, W], 像素值范围[0, 255]
            iters: 迭代更新次数，默认12次
            flow_init: 初始光流（可选）shape=[N, 2, H//8, W//8]
            upsample: 是否上采样到原始分辨率
            test_mode: 测试模式，如果为True只返回最终结果
            
        Returns:
            训练模式: 所有迭代的光流预测列表，每个元素shape=[N, 2, H, W]
            测试模式: (低分辨率光流, 高分辨率光流) 元组
        """

        # === 1. 图像预处理 ===
        # 将图像从[0,255]归一化到[-1,1]范围
        image1 = 2 * (image1 / 255.0) - 1.0  # shape: [N, 3, H, W]
        image2 = 2 * (image2 / 255.0) - 1.0  # shape: [N, 3, H, W]

        # 确保内存连续性，提高计算效率
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # 获取网络维度参数
        hdim = self.hidden_dim    # GRU隐藏层维度
        cdim = self.context_dim   # 上下文特征维度

        # === 2. 特征提取 ===
        # 使用特征编码器提取两帧图像的特征
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])  # shape: [N, 256, H//8, W//8] (标准模型)
        
        # 转换为float32确保数值稳定性
        fmap1 = fmap1.float()  # shape: [N, 256, H//8, W//8]
        fmap2 = fmap2.float()  # shape: [N, 256, H//8, W//8]
        
        # === 3. 构建相关性函数 ===
        # 根据配置选择相关性计算方法
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # === 4. 上下文特征提取 ===
        # 使用上下文编码器提取第一帧的上下文信息
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)  # shape: [N, hdim+cdim, H//8, W//8]
            # 分离隐藏状态和输入特征
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # net: GRU隐藏状态 shape=[N, hdim, H//8, W//8]
            # inp: 输入特征 shape=[N, cdim, H//8, W//8]
            net = torch.tanh(net)  # 激活函数
            inp = torch.relu(inp)  # 激活函数

        # === 5. 初始化光流坐标 ===
        coords0, coords1 = self.initialize_flow(image1)
        # coords0: 源坐标网格 shape=[N, 2, H//8, W//8] (固定)
        # coords1: 目标坐标网格 shape=[N, 2, H//8, W//8] (会被更新)

        # 如果提供了初始光流，则添加到coords1
        if flow_init is not None:
            coords1 = coords1 + flow_init  # shape: [N, 2, H//8, W//8]

        # === 6. 迭代优化光流 ===
        flow_predictions = []  # 存储每次迭代的光流预测
        
        for itr in range(iters):
            # 阻断梯度，防止通过迭代历史传播梯度
            coords1 = coords1.detach()  # shape: [N, 2, H//8, W//8]
            
            # 计算当前坐标的相关性特征
            corr = corr_fn(coords1)  # shape: [N, levels*(2*radius+1)^2, H//8, W//8]

            # 计算当前光流
            flow = coords1 - coords0  # shape: [N, 2, H//8, W//8]
            
            # 使用更新模块预测光流增量
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                # net: 更新后的隐藏状态 shape=[N, hdim, H//8, W//8]
                # up_mask: 上采样掩码 shape=[N, 64*9, H//8, W//8] (可选)
                # delta_flow: 光流增量 shape=[N, 2, H//8, W//8]

            # 更新坐标：F(t+1) = F(t) + Δ(t)
            coords1 = coords1 + delta_flow  # shape: [N, 2, H//8, W//8]

            # === 7. 上采样到原始分辨率 ===
            if up_mask is None:
                # 使用简单的8倍上采样
                flow_up = upflow8(coords1 - coords0)  # shape: [N, 2, H, W]
            else:
                # 使用学习的上采样方法
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)  # shape: [N, 2, H, W]
            
            # 保存当前迭代的预测结果
            flow_predictions.append(flow_up)

        # === 8. 返回结果 ===
        if test_mode:
            # 测试模式：返回低分辨率和高分辨率光流
            return coords1 - coords0, flow_up  # ([N, 2, H//8, W//8], [N, 2, H, W])
        else:
            # 训练模式：返回所有迭代的预测结果
            return flow_predictions  # List[Tensor], 每个元素shape=[N, 2, H, W]
