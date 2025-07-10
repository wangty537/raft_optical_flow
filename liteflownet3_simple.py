from typing import Any, Dict, List, Optional, Union, Tuple


from liteflownet3_correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler

import torch
import torch.nn as nn
import torch.nn.functional as F


from liteflownet3_warp import WarpingLayer

import numpy as np

from liteflownet3_util import InputPadder, InputScaler, bgr_val_as_tensor
class FeatureExtractor(nn.Module):
    """
    特征提取器：构建多尺度特征金字塔
    原理：通过多层卷积网络提取不同分辨率的特征，用于后续的光流估计
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # LeakyReLU激活函数，负斜率为0.1
        leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # 构建6层特征提取网络，每层输出不同分辨率的特征
        self.convs = nn.ModuleList(
            [
                # Level 0: 输入3通道RGB图像，输出32通道特征，保持分辨率
                # Shape: (B*2, 3, H, W) -> (B*2, 32, H, W)
                nn.Sequential(nn.Conv2d(3, 32, 7, 1, 3), leaky_relu),
                
                # Level 1: 下采样2倍，输出32通道特征
                # Shape: (B*2, 32, H, W) -> (B*2, 32, H/2, W/2)
                nn.Sequential(
                    nn.Conv2d(32, 32, 3, 2, 1),  # 步长2进行下采样
                    leaky_relu,
                    nn.Conv2d(32, 32, 3, 1, 1),  # 保持分辨率的卷积
                    leaky_relu,
                    nn.Conv2d(32, 32, 3, 1, 1),  # 保持分辨率的卷积
                    leaky_relu,
                ),
                
                # Level 2: 继续下采样2倍，输出64通道特征
                # Shape: (B*2, 32, H/2, W/2) -> (B*2, 64, H/4, W/4)
                nn.Sequential(
                    nn.Conv2d(32, 64, 3, 2, 1),  # 步长2进行下采样
                    leaky_relu,
                    nn.Conv2d(64, 64, 3, 1, 1),  # 保持分辨率的卷积
                    leaky_relu,
                ),
                
                # Level 3: 继续下采样2倍，输出96通道特征
                # Shape: (B*2, 64, H/4, W/4) -> (B*2, 96, H/8, W/8)
                nn.Sequential(
                    nn.Conv2d(64, 96, 3, 2, 1),  # 步长2进行下采样
                    leaky_relu,
                    nn.Conv2d(96, 96, 3, 1, 1),  # 保持分辨率的卷积
                    leaky_relu,
                ),
                
                # Level 4: 继续下采样2倍，输出128通道特征
                # Shape: (B*2, 96, H/8, W/8) -> (B*2, 128, H/16, W/16)
                nn.Sequential(nn.Conv2d(96, 128, 3, 2, 1), leaky_relu),
                
                # Level 5: 继续下采样2倍，输出192通道特征
                # Shape: (B*2, 128, H/16, W/16) -> (B*2, 192, H/32, W/32)
                nn.Sequential(nn.Conv2d(128, 192, 3, 2, 1), leaky_relu),
            ]
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播：提取多尺度特征金字塔
        
        Args:
            images: 输入图像对，Shape: (B, 2, 3, H, W)
        
        Returns:
            features: 特征金字塔列表，从高分辨率到低分辨率
        """
        features = []

        # 将图像对重塑为批次维度，Shape: (B, 2, 3, H, W) -> (B*2, 3, H, W)
        x = images.view(-1, *images.shape[2:])
        
        # 逐层提取特征
        for i, conv in enumerate(self.convs):
            x = conv(x)  # 通过卷积层提取特征
            
            # 从第3层开始保存特征（跳过前两层，因为分辨率太高）
            if i > 1:
                # 恢复图像对的维度，Shape: (B*2, C, H', W') -> (B, 2, C, H', W')
                features.append(x.view(*images.shape[:2], *x.shape[1:]))

        # 返回反向排列的特征列表（从低分辨率到高分辨率）
        # 这样便于从粗到细的光流估计
        return features[::-1]


class FlowFieldDeformation(nn.Module):
    """
    光流场变形模块：通过自相关和置信度信息对光流进行精细化调整
    原理：利用特征的自相关性来检测和修正光流场中的不一致性
    """
    def __init__(self, level: int) -> None:
        super(FlowFieldDeformation, self).__init__()

        # 根据金字塔层级设置不同的patch大小和预测核大小
        # 高层级使用更大的感受野来捕获更大的运动
        patch_size = [None, 5, 7, 9][level]  # level 1,2,3对应patch_size 5,7,9
        pred_kernel_size = [None, 3, 5, 5][level]  # level 1,2,3对应kernel_size 3,5,5

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # 上采样置信度图，Shape: (B, 1, H/2, W/2) -> (B, 1, H, W)
        self.up_conf = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        # 上采样光流图，使用分组卷积分别处理x和y方向
        # Shape: (B, 2, H/2, W/2) -> (B, 2, H, W)
        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)

        # 空间相关采样器：计算特征的自相关
        # 使用膨胀卷积增大感受野，dilation_patch=2
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=patch_size, padding=0, stride=1, dilation_patch=2
        )

        # 特征处理网络：处理自相关特征和置信度
        # 输入通道数：patch_size^2（自相关） + 1（置信度）
        self.feat_net = nn.Sequential(
            nn.Conv2d(patch_size**2 + 1, 128, 3, 1, 1),  # Shape: (B, patch_size^2+1, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),  # Shape: (B, 128, H, W) -> (B, 64, H, W)
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),   # Shape: (B, 64, H, W) -> (B, 32, H, W)
            self.leaky_relu,
        )

        # 位移预测器：预测光流的修正量
        # Shape: (B, 32, H, W) -> (B, 2, H, W)
        self.disp_pred = nn.Conv2d(32, 2, pred_kernel_size, 1, pred_kernel_size // 2)

        # 置信度预测器：预测新的置信度
        # Shape: (B, 32, H, W) -> (B, 1, H, W)
        self.conf_pred = nn.Sequential(
            nn.Conv2d(32, 1, pred_kernel_size, 1, pred_kernel_size // 2), 
            nn.Sigmoid()  # 将置信度限制在[0,1]范围内
        )

        # 图像扭曲层：用于光流变形
        self.warp = WarpingLayer()

    def forward(
        self, feats: torch.Tensor, flow: torch.Tensor, conf: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：对光流进行变形修正
        
        Args:
            feats: 特征对，Shape: (B, 2, C, H, W)
            flow: 输入光流，Shape: (B, 2, H/2, W/2)
            conf: 输入置信度，Shape: (B, 1, H/2, W/2)
        
        Returns:
            flow: 修正后的光流，Shape: (B, 2, H, W)
            conf: 新的置信度，Shape: (B, 1, H, W)
        """
        # 上采样置信度图到当前分辨率
        # Shape: (B, 1, H/2, W/2) -> (B, 1, H, W)
        conf = self.up_conf(conf)
        
        # 上采样光流图到当前分辨率
        # Shape: (B, 2, H/2, W/2) -> (B, 2, H, W)
        flow = self.up_flow(flow)

        # 计算第一帧特征的自相关
        # 自相关可以检测特征的一致性和可靠性
        # Shape: (B, C, H, W) -> (B, patch_size^2, H, W)
        self_corr = self.leaky_relu(self.corr(feats[:, 0], feats[:, 0]))
        
        # 重塑自相关张量的维度
        # Shape: (B, 1, patch_size^2, H, W) -> (B, patch_size^2, H, W)
        self_corr = self_corr.view(
            self_corr.shape[0], -1, self_corr.shape[3], self_corr.shape[4]
        )
        
        # 归一化自相关，除以特征通道数
        self_corr = self_corr / feats.shape[2]

        # 拼接自相关特征和置信度
        # Shape: (B, patch_size^2+1, H, W)
        x = torch.cat([self_corr, conf], dim=1)
        
        # 通过特征网络处理
        # Shape: (B, patch_size^2+1, H, W) -> (B, 32, H, W)
        x = self.feat_net(x)

        # 预测位移修正量
        # Shape: (B, 32, H, W) -> (B, 2, H, W)
        disp = self.disp_pred(x)

        # 使用位移对光流进行变形修正
        # 这里使用光流本身作为变形场来修正自己
        flow = self.warp(flow, disp, flow.shape[-2], flow.shape[-1], 1.0)

        # 预测新的置信度
        # Shape: (B, 32, H, W) -> (B, 1, H, W)
        conf = self.conf_pred(x)

        return flow, conf


class CostVolumeModulation(nn.Module):
    """
    代价体调制模块：通过学习的调制参数改善代价体的质量
    原理：使用标量调制和偏移调制来增强代价体中有用的匹配信息，抑制噪声
    """
    def __init__(self, level: int, num_levels: int = 4, div_flow: float = 20.0) -> None:
        super().__init__()

        # 根据层级设置输入维度：特征通道数 + 代价体通道数(81) + 置信度通道数(1)
        # level 1: 32+81+1=114, level 2: 64+81+1=146, level 3: 96+81+1=178
        input_dims = [None, 210, 178, 146][level]
        
        # 计算光流缩放因子，用于将光流转换到当前层级的尺度
        self.mult = [div_flow / 2 ** (num_levels - i + 1) for i in range(num_levels)][
            level
        ]

        # 空间相关采样器：计算特征间的相关性
        # patch_size=9产生9x9=81个相关值
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1
        )

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # 特征处理网络：处理拼接的特征、代价体和置信度
        self.feat_net = nn.Sequential(
            nn.Conv2d(input_dims, 128, 3, 1, 1),  # Shape: (B, input_dims, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),          # Shape: (B, 128, H, W) -> (B, 64, H, W)
            self.leaky_relu,
        )

        # 标量调制网络：学习每个代价体通道的权重
        # 输出81个通道，对应9x9代价体的每个位置
        self.mod_scalar_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),    # Shape: (B, 64, H, W) -> (B, 32, H, W)
            self.leaky_relu, 
            nn.Conv2d(32, 81, 1, 1, 0)     # Shape: (B, 32, H, W) -> (B, 81, H, W)
        )

        # 偏移调制网络：学习每个代价体通道的偏移
        # 输出81个通道，对应9x9代价体的每个位置
        self.mod_offset_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),    # Shape: (B, 64, H, W) -> (B, 32, H, W)
            self.leaky_relu, 
            nn.Conv2d(32, 81, 1, 1, 0)     # Shape: (B, 32, H, W) -> (B, 81, H, W)
        )

        # 图像扭曲层：用于特征对齐
        self.warp = WarpingLayer()

    def forward(
        self, feats: torch.Tensor, flow: torch.Tensor, conf: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：生成调制后的代价体
        
        Args:
            feats: 特征对，Shape: (B, 2, C, H, W)
            flow: 光流，Shape: (B, 2, H, W)
            conf: 置信度，Shape: (B, 1, H, W)
        
        Returns:
            corr: 调制后的代价体，Shape: (B, 81, H, W)
        """
        # 使用光流对第二帧特征进行扭曲对齐
        # 光流需要按层级缩放到正确的尺度
        # Shape: (B, C, H, W)
        warped_feat2 = self.warp(
            feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult
        )

        # 计算第一帧特征与扭曲后第二帧特征的相关性
        # Shape: (B, C, H, W) x (B, C, H, W) -> (B, 1, 81, H, W)
        corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
        
        # 重塑代价体维度
        # Shape: (B, 1, 81, H, W) -> (B, 81, H, W)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        
        # 归一化代价体，除以特征通道数
        corr = corr / feats.shape[2]

        # 拼接第一帧特征、代价体和置信度
        # Shape: (B, C+81+1, H, W)
        x = torch.cat([feats[:, 0], corr, conf], dim=1)
        
        # 通过特征网络处理
        # Shape: (B, C+81+1, H, W) -> (B, 64, H, W)
        x = self.feat_net(x)

        # 预测标量调制参数
        # Shape: (B, 64, H, W) -> (B, 81, H, W)
        mod_scalar = self.mod_scalar_net(x)
        
        # 预测偏移调制参数
        # Shape: (B, 64, H, W) -> (B, 81, H, W)
        mod_offset = self.mod_offset_net(x)

        # 应用调制：标量乘法 + 偏移加法
        # 这样可以增强有用的匹配信息，抑制噪声
        # Shape: (B, 81, H, W)
        corr = mod_scalar * corr + mod_offset

        return corr


class Matching(nn.Module):
    """
    匹配模块：基于代价体预测光流
    原理：使用卷积网络从代价体中回归光流，支持从粗到细的层次化预测
   
    """
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0, #光流网络通常预测的是归一化后的相对位移（值域较小，如 [-1, 1]），而非真实的像素位移。div_flow 的作用是将网络输出的光流值重新缩放回真实尺度。
        use_s_version: bool = False,
    ) -> None:
        super(Matching, self).__init__()

        # 根据层级设置光流预测的卷积核大小
        # 高层级使用更大的核来捕获更大的运动
        flow_kernel_size = [3, 3, 5, 5][level]  # level 0,1,2,3对应kernel_size 3,3,5,5
        
        # 计算光流缩放因子
        self.mult = [div_flow / 2 ** (num_levels - i + 1) for i in range(num_levels)][
            level
        ]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # 只在level=1且非S版本时进行光流上采样
        # Shape: (B, 2, H/2, W/2) -> (B, 2, H, W)
        if level == 1 and not use_s_version:
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        else:
            self.up_flow = None

        # 只在前两层(level 0,1)计算相关性，后续层使用调制后的代价体
        if level < 2:
            self.corr = SpatialCorrelationSampler(
                kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1
            )
        else:
            self.corr = None

        # 光流预测网络：从代价体回归光流
        # 输入固定为81通道的代价体(9x9相关窗口)
        self.flow_net = nn.Sequential(
            nn.Conv2d(81, 128, 3, 1, 1),    # Shape: (B, 81, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),   # Shape: (B, 128, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 96, 3, 1, 1),    # Shape: (B, 128, H, W) -> (B, 96, H, W)
            self.leaky_relu,
            nn.Conv2d(96, 64, 3, 1, 1),     # Shape: (B, 96, H, W) -> (B, 64, H, W)
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),     # Shape: (B, 64, H, W) -> (B, 32, H, W)
            self.leaky_relu,
            # 最终预测2通道光流(x,y方向)
            nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size // 2),  # Shape: (B, 32, H, W) -> (B, 2, H, W)
        )

        # 图像扭曲层：用于特征对齐
        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: Optional[torch.Tensor],
        corr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播：从代价体预测光流
        
        Args:
            feats: 特征对，Shape: (B, 2, C, H, W)
            flow: 上一层的光流(可选)，Shape: (B, 2, H', W') 或 None
            corr: 调制后的代价体(可选)，Shape: (B, 81, H, W) 或 None
        
        Returns:
            new_flow: 预测的光流，Shape: (B, 2, H, W)
        """
        # 如果需要，对输入光流进行上采样
        if self.up_flow is not None:
            # Shape: (B, 2, H/2, W/2) -> (B, 2, H, W)
            flow = self.up_flow(flow)

        # 如果没有提供调制后的代价体，则计算原始代价体
        if corr is None:
            # 默认使用第二帧特征
            warped_feat2 = feats[:, 1]
            
            # 如果有光流，则先对第二帧特征进行扭曲对齐
            if flow is not None:
                # Shape: (B, C, H, W)
                warped_feat2 = self.warp(
                    feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult
                )

            # 计算第一帧特征与(扭曲后)第二帧特征的相关性
            # Shape: (B, C, H, W) x (B, C, H, W) -> (B, 1, 81, H, W)
            corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
            
            # 重塑代价体维度
            # Shape: (B, 1, 81, H, W) -> (B, 81, H, W)
            corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
            
            # 归一化代价体
            corr = corr / feats.shape[2]

        # 从代价体预测光流增量
        # Shape: (B, 81, H, W) -> (B, 2, H, W)
        new_flow = self.flow_net(corr)
        
        # 如果有输入光流，则将预测的增量加到输入光流上(残差连接)
        if flow is not None:
            new_flow = flow + new_flow
            
        return new_flow


class SubPixel(nn.Module):
    def __init__(self, level: int, num_levels: int = 4, div_flow: float = 20.0) -> None:
        super(SubPixel, self).__init__()

        inputs_dims = [386, 258, 194, 130][level]
        flow_kernel_size = [3, 3, 5, 5][level]
        self.mult = [div_flow / 2 ** (num_levels - i + 1) for i in range(num_levels)][
            level
        ]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.feat_net = nn.Sequential(
            nn.Conv2d(inputs_dims, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 96, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(96, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
        )

        self.flow_net = nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size // 2)

        self.warp = WarpingLayer()

    def forward(self, feats: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        feat_warped = self.warp(
            feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult
        )
        x = torch.cat([feats[:, 0], feat_warped, flow], dim=1)
        x = self.feat_net(x)
        new_flow = self.flow_net(x)
        new_flow = flow + new_flow
        return new_flow, x


class Regularization(nn.Module):
    """
    正则化模块：通过自适应平滑对光流进行正则化处理
    
    原理：
    1. 图像一致性检查：计算扭曲后图像与目标图像的差异
    2. 特征融合：结合图像差异、去均值光流和特征信息
    3. 距离权重计算：学习每个邻域像素的权重分布
    4. 自适应平滑：基于学习的权重对光流进行加权平均
    5. 置信度预测：估计光流的可靠性
    """
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0,
        use_s_version: bool = False,
    ) -> None:
        super(Regularization, self).__init__()

        # 各层级的输入维度：[图像差异(1) + 去均值光流(2) + 特征维度]
        # level 0: 1+2+192=195, level 1: 1+2+128=131, level 2: 1+2+96=99, level 3: 1+2+64=67
        inputs_dims = [195, 131, 99, 67][level]
        
        # 光流平滑的邻域窗口大小
        flow_kernel_size = [3, 3, 5, 5][level]  # 高层级使用更大的平滑窗口
        
        # 置信度预测的卷积核大小
        conf_kernel_size = [3, 3, 5, None][level]  # level 3不预测置信度
        
        # 光流缩放因子
        self.mult = [div_flow / 2 ** (num_levels - i + 1) for i in range(num_levels)][
            level
        ]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # 特征预处理：高层级需要降维
        if level < 2:
            # 前两层：直接使用原始特征
            self.feat_conv = nn.Sequential()
        else:
            # 后两层：特征降维到128维
            # Shape: (B, inputs_dims-3, H, W) -> (B, 128, H, W)
            self.feat_conv = nn.Sequential(
                nn.Conv2d(inputs_dims - 3, 128, 1, 1, 0), self.leaky_relu
            )
            inputs_dims = 131  # 更新输入维度：1+2+128=131

        # 特征处理网络：提取用于距离计算的特征
        self.feat_net = nn.Sequential(
            nn.Conv2d(inputs_dims, 128, 3, 1, 1),    # Shape: (B, inputs_dims, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),            # Shape: (B, 128, H, W) -> (B, 128, H, W)
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),             # Shape: (B, 128, H, W) -> (B, 64, H, W)
            self.leaky_relu,
            nn.Conv2d(64, 64, 3, 1, 1),              # Shape: (B, 64, H, W) -> (B, 64, H, W)
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),              # Shape: (B, 64, H, W) -> (B, 32, H, W)
            self.leaky_relu,
            nn.Conv2d(32, 32, 3, 1, 1),              # Shape: (B, 32, H, W) -> (B, 32, H, W)
            self.leaky_relu,
        )

        # 距离权重预测器：计算邻域像素的相似性权重
        if level < 2:
            # 前两层：使用标准3x3卷积
            # Shape: (B, 32, H, W) -> (B, kernel_size^2, H, W)
            self.dist = nn.Conv2d(32, flow_kernel_size**2, 3, 1, 1)
        else:
            # 后两层：使用可分离卷积减少参数量
            # Shape: (B, 32, H, W) -> (B, kernel_size^2, H, W)
            self.dist = nn.Sequential(
                # 垂直方向卷积
                nn.Conv2d(
                    32,
                    flow_kernel_size**2,
                    (flow_kernel_size, 1),
                    1,
                    (flow_kernel_size // 2, 0),
                ),
                # 水平方向卷积
                nn.Conv2d(
                    flow_kernel_size**2,
                    flow_kernel_size**2,
                    (1, flow_kernel_size),
                    1,
                    (0, flow_kernel_size // 2),
                ),
            )

        # 邻域展开操作：将邻域像素展开为通道维度
        self.unfold = nn.Unfold(flow_kernel_size, padding=flow_kernel_size // 2)

        # 置信度预测器：某些层级不预测置信度
        if (level == 0 and not use_s_version) or level == 3:
            self.conf_pred = None
        else:
            # Shape: (B, 32, H, W) -> (B, 1, H, W)
            self.conf_pred = nn.Sequential(
                nn.Conv2d(32, 1, conf_kernel_size, 1, conf_kernel_size // 2),
                nn.Sigmoid(),  # 输出0-1之间的置信度
            )

        # 图像扭曲层：用于图像对齐
        self.warp = WarpingLayer()

    def forward(
        self, images: torch.Tensor, feats: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：对光流进行自适应正则化平滑
        
        Args:
            images: 图像对，Shape: (B, 2, 3, H, W)
            feats: 特征对，Shape: (B, 2, C, H, W)
            flow: 输入光流，Shape: (B, 2, H, W)
        
        Returns:
            flow: 正则化后的光流，Shape: (B, 2, H, W)
            conf: 置信度图(可选)，Shape: (B, 1, H, W) 或 None
            x: 特征表示，Shape: (B, 32, H, W)
        """
        # 1. 图像一致性检查：用光流扭曲第二帧图像
        # Shape: (B, 3, H, W)
        img2_warped = self.warp(
            images[:, 1], flow, images.shape[-2], images.shape[-1], 1.0 / self.mult
        )
        
        # 计算扭曲后图像与第一帧的L2差异
        # Shape: (B, 3, H, W) -> (B, 1, H, W)
        img_diff_norm = torch.norm(images[:, 0] - img2_warped, p=2, dim=1, keepdim=True)

        # 2. 光流去均值：移除全局运动偏移
        # 计算光流的空间均值
        # Shape: (B, 2, H, W) -> (B, 2, H*W) -> (B, 2, 1, 1)
        flow_mean = flow.view(*flow.shape[:2], -1).mean(dim=-1)[..., None, None]
        
        # 去除均值，保留局部运动模式
        # Shape: (B, 2, H, W)
        flow_nomean = flow - flow_mean
        
        # 3. 特征预处理：降维或直接使用
        # Shape: (B, C, H, W) -> (B, C', H, W)
        feat = self.feat_conv(feats[:, 0])
        
        # 4. 特征融合：连接图像差异、去均值光流和特征
        # Shape: (B, 1+2+C', H, W)
        x = torch.cat([img_diff_norm, flow_nomean, feat], dim=1)
        
        # 5. 特征处理：提取用于距离计算的特征
        # Shape: (B, inputs_dims, H, W) -> (B, 32, H, W)
        x = self.feat_net(x)
        
        # 6. 距离权重计算：学习邻域像素的相似性权重
        # Shape: (B, 32, H, W) -> (B, kernel_size^2, H, W)
        dist = self.dist(x)
        
        # 转换为负平方距离
        dist = dist.square().neg()
        
        # 数值稳定的softmax：先减去最大值再指数化
        # Shape: (B, kernel_size^2, H, W)
        dist = (dist - dist.max(dim=1, keepdim=True)[0]).exp()
        
        # 归一化权重：确保权重和为1
        # Shape: (B, 1, H, W)
        div = dist.sum(dim=1, keepdim=True)

        # 7. 光流平滑：对x和y分量分别进行加权平均
        
        # 处理x分量
        # 展开邻域：Shape: (B, 1, H, W) -> (B, kernel_size^2, H, W)
        reshaped_flow_x = self.unfold(flow[:, :1])
        reshaped_flow_x = reshaped_flow_x.view(
            *reshaped_flow_x.shape[:2], *flow.shape[2:4]
        )
        
        # 加权平均：Shape: (B, kernel_size^2, H, W) -> (B, 1, H, W)
        flow_smooth_x = (reshaped_flow_x * dist).sum(dim=1, keepdim=True) / div

        # 处理y分量
        # 展开邻域：Shape: (B, 1, H, W) -> (B, kernel_size^2, H, W)
        reshaped_flow_y = self.unfold(flow[:, 1:2])
        reshaped_flow_y = reshaped_flow_y.view(
            *reshaped_flow_y.shape[:2], *flow.shape[2:4]
        )
        
        # 加权平均：Shape: (B, kernel_size^2, H, W) -> (B, 1, H, W)
        flow_smooth_y = (reshaped_flow_y * dist).sum(dim=1, keepdim=True) / div

        # 8. 合并平滑后的光流分量
        # Shape: (B, 2, H, W)
        flow = torch.cat([flow_smooth_x, flow_smooth_y], dim=1)

        # 9. 置信度预测(可选)
        conf = None
        if self.conf_pred is not None:
            # Shape: (B, 32, H, W) -> (B, 1, H, W)
            conf = self.conf_pred(x)

        return flow, conf, x


class PseudoSubpixel(nn.Module):
    def __init__(self) -> None:
        super(PseudoSubpixel, self).__init__()

        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)

        self.flow_net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.Conv2d(32, 2, 7, 1, 3)
        )

    def forward(self, sub_feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return self.up_flow(flow) + self.flow_net(sub_feat)


class PseudoRegularization(nn.Module):
    def __init__(self) -> None:
        super(PseudoRegularization, self).__init__()

        self.feat_net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.Conv2d(32, 49, (7, 1), 1, (3, 0)),
            nn.Conv2d(49, 49, (1, 7), 1, (0, 3)),
        )

        self.unfold = nn.Unfold(7, padding=3)

    def forward(self, reg_feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        dist = self.feat_net(reg_feat)
        dist = dist.square().neg()
        dist = (dist - dist.max(dim=1, keepdim=True)[0]).exp()
        div = dist.sum(dim=1, keepdim=True)

        reshaped_flow_x = self.unfold(flow[:, :1])
        reshaped_flow_x = reshaped_flow_x.view(
            *reshaped_flow_x.shape[:2], *flow.shape[2:4]
        )
        flow_smooth_x = (reshaped_flow_x * dist).sum(dim=1, keepdim=True) / div

        reshaped_flow_y = self.unfold(flow[:, 1:2])
        reshaped_flow_y = reshaped_flow_y.view(
            *reshaped_flow_y.shape[:2], *flow.shape[2:4]
        )
        flow_smooth_y = (reshaped_flow_y * dist).sum(dim=1, keepdim=True) / div

        flow = torch.cat([flow_smooth_x, flow_smooth_y], dim=1)

        return flow


class LiteFlowNet3(nn.Module):
    """
    LiteFlowNet3主网络：轻量级光流估计网络第三版
    
    原理：
    1. 特征提取：构建多尺度特征金字塔
    2. 从粗到细的光流估计：
       - 在每个尺度上进行光流预测
       - 使用光流场变形和代价体调制提升精度
       - 通过残差连接逐步细化光流
    3. 多尺度监督：输出所有尺度的光流用于训练
    """
    pretrained_checkpoints = {
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3-sintel-d985929f.ckpt"
    }

    def __init__(
        self,
        div_flow: float = 20.0,
        use_pseudo_regularization: bool = False,
        use_s_version: bool = False,
        **kwargs,
    ):
        super(LiteFlowNet3, self).__init__()

        self.div_flow = div_flow  # 光流缩放因子
        self.use_pseudo_regularization = use_pseudo_regularization  # 是否使用伪正则化
        self.use_s_version = use_s_version  # 是否使用S版本(简化版)

        self.num_levels = 4  # 金字塔层数
        self.output_stride = 32  # 图像尺寸需要是32的倍数

        # 根据版本设置最小调制层级
        if use_s_version:
            self.min_mod_level = 1  # S版本从第1层开始调制
        else:
            self.min_mod_level = 2  # 标准版本从第2层开始调制

        # 特征提取器：构建多尺度特征金字塔
        self.feature_net = FeatureExtractor() # 4,8,16,32
        
        # 光流场变形模块：通过自相关和置信度信息对光流进行warp
        self.deformation_nets = nn.ModuleList(
            [
                FlowFieldDeformation(i)
                for i in range(self.min_mod_level, self.num_levels)
            ]
        )
        
        # 代价体调制模块：对corr进行调整。应用调制：标量乘法 + 偏移加法
        self.modulation_nets = nn.ModuleList(
            [
                CostVolumeModulation(i, self.num_levels, self.div_flow)
                for i in range(self.min_mod_level, self.num_levels)
            ]
        )
        
        # 匹配模块：基于代价体预测光流
        self.matching_nets = nn.ModuleList(
            [
                Matching(i, self.num_levels, self.div_flow, self.use_s_version)
                for i in range(self.num_levels)
            ]
        )
        
        # concat warp后的特征, 预测光流
        self.subpixel_nets = nn.ModuleList(
            [
                SubPixel(i, self.num_levels, self.div_flow)
                for i in range(self.num_levels)
            ]
        )
        
        # 正则化模块：根据warp后的特征与目标特征差异, 预测邻域flow的weight, 重新计算flow
        self.regularization_nets = nn.ModuleList(
            [
                Regularization(i, self.num_levels, self.div_flow, self.use_s_version)
                for i in range(self.num_levels)
            ]
        )

        # 根据是否使用伪正则化设置最终上采样层
        if self.use_pseudo_regularization:
            self.pseudo_subpixel = PseudoSubpixel()
            self.pseudo_regularization = PseudoRegularization()
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        else:
            self.up_flow = nn.ConvTranspose2d(2, 2, 8, 4, 2, bias=False, groups=2)

 
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播：从粗到细的光流估计
        
        Args:
            inputs: 输入字典，包含"images"键，Shape: (B, 2, 3, H, W)
        
        Returns:
            outputs: 输出字典，包含:
                - "flows": 最终光流，Shape: (B, 1, 2, H, W)
                - "confs": 置信度图，Shape: (B, 1, 1, H, W)
                - "flow_preds": 训练时的多尺度光流预测列表
                - "conf_preds": 训练时的多尺度置信度预测列表
        """
        # 图像预处理：BGR转RGB，归一化，调整分辨率
        # 原始实现对im1和im2使用不同的BGR均值，这里简化为相同
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=[-0.454253, -0.434631, -0.411618],  # BGR均值
            bgr_mult=1.0,
            bgr_to_rgb=True,
            resize_mode="interpolation",
            interpolation_mode="bilinear",
            interpolation_align_corners=False,
        )

        # 特征提取：构建多尺度特征金字塔
        # feats_pyr[i] Shape: (B, 2, C_i, H_i, W_i) # 4,8,16,32
        feats_pyr = self.feature_net(images)
        
        # 创建对应的图像金字塔，用于正则化模块
        # images_pyr[i] Shape: (B, 2, 3, H_i, W_i) # 缩小4,8,16,32倍
        images_pyr = self._create_images_pyr(images, feats_pyr)

        flow_preds = []  # 存储每个层级的光流预测
        conf_preds = []  # 存储每个层级的置信度预测
        flow = None      # 当前光流
        conf = None      # 当前置信度
        corr = None      # 调制后的代价体

        # 从粗到细的光流估计(level 0最粗，level 3最细)
        for i in range(self.num_levels):
            # 如果达到调制层级，进行光流场变形和代价体调制
            if i >= self.min_mod_level:
                # 1. 光流场变形：基于自相关和置信度精细化光流
                flow, conf = self.deformation_nets[i - self.min_mod_level](
                    feats_pyr[i], flow, conf
                )
                if conf is not None:
                    conf_preds.append(conf)
                
                # 2. 代价体调制：改善代价体质量
                corr = self.modulation_nets[i - self.min_mod_level](
                    feats_pyr[i], flow, conf
                )
            
            # 3. 匹配：基于(调制后的)代价体预测光流
            flow = self.matching_nets[i](feats_pyr[i], flow, corr)
            
            # 4. 亚像素细化：进一步提升光流精度
            flow, sub_feat = self.subpixel_nets[i](feats_pyr[i], flow)
            
            # 5. 正则化：平滑光流场，提升一致性
            flow, conf, reg_feat = self.regularization_nets[i](
                images_pyr[i], feats_pyr[i], flow
            )
            
            # 保存当前层级的预测结果
            flow_preds.append(flow)
            if conf is not None:
                conf_preds.append(conf)

        # 最终处理：上采样到原始分辨率
        if self.use_pseudo_regularization:
            # 使用伪正则化：额外的亚像素细化和正则化
            flow = self.pseudo_subpixel(sub_feat, flow)
            flow = self.pseudo_regularization(reg_feat, flow)
            flow = self.up_flow(flow)  # 2倍上采样
        else:
            # 标准处理：直接4倍上采样
            flow = self.up_flow(flow)
        
        # 恢复光流的真实尺度
        flow = flow * self.div_flow
        
        # 后处理：恢复到原始图像分辨率
        flow = self.postprocess_predictions(flow, image_resizer, is_flow=True)

        # 处理置信度：上采样并后处理
        conf = F.interpolate(
            conf_preds[-1], scale_factor=4, mode="bilinear", align_corners=False
        )
        conf = self.postprocess_predictions(conf, image_resizer, is_flow=False)

        # 构建输出字典
        outputs = {}
        if self.training:
            # 训练时：返回所有层级的预测用于多尺度监督
            outputs["flow_preds"] = flow_preds
            outputs["conf_preds"] = conf_preds
            outputs["flows"] = flow[:, None]  # 添加时间维度
            outputs["confs"] = conf[:, None]  # 添加时间维度
        else:
            # 推理时：只返回最终结果
            outputs["flows"] = flow[:, None]
            outputs["confs"] = conf[:, None]
        return outputs

    def _create_images_pyr(
        self, images: torch.Tensor, feats_pyr: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        创建图像金字塔：将原始图像下采样到与特征金字塔相同的分辨率
        
        原理：正则化模块需要原始图像信息来计算光流的平滑性约束，
              因此需要将图像调整到与特征相同的分辨率
        
        Args:
            images: 原始图像，Shape: (B, 2, 3, H, W)
            feats_pyr: 特征金字塔列表，feats_pyr[i] Shape: (B, 2, C_i, H_i, W_i)
        
        Returns:
            images_pyr: 图像金字塔列表，images_pyr[i] Shape: (B, 2, 3, H_i, W_i)
        """
        batch_size = images.shape[0]
        images = images.view(-1, *images.shape[2:]).detach()  # 展平批次和时间维度
        
        # 为每个特征层创建对应分辨率的图像
        images_pyr = [
            F.interpolate(
                images,
                size=feats_pyr[i].shape[-2:],  # 使用特征层的空间尺寸
                mode="bilinear",
                align_corners=False,
            )
            for i in range(len(feats_pyr))
        ]
        
        # 恢复批次和时间维度
        # Shape: (B*2, 3, H_i, W_i) -> (B, 2, 3, H_i, W_i)
        images_pyr = [im.view(batch_size, -1, *im.shape[1:]) for im in images_pyr]
        return images_pyr
    def preprocess_images(
        self,
        images: torch.Tensor,
        stride: Optional[int] = None,
        bgr_add: Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor] = 0,
        bgr_mult: Union[
            float, Tuple[float, float, float], np.ndarray, torch.Tensor
        ] = 1,
        bgr_to_rgb: bool = False,
        image_resizer: Optional[Union[InputPadder, InputScaler]] = None,
        resize_mode: str = "pad",
        target_size: Optional[Tuple[int, int]] = None,
        pad_mode: str = "replicate",
        pad_value: float = 0.0,
        pad_two_side: bool = True,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = True,
    ) -> Tuple[torch.Tensor, Union[InputPadder, InputScaler]]:
        """Applies basic pre-processing to the images.

        The pre-processing is done in this order:
        1. images = images + bgr_add
        2. images = images * bgr_mult
        3. (optional) Convert BGR channels to RGB
        4. Pad or resize the input to the closest larger size multiple of self.output_stride

        Parameters
        ----------
        images : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., 3, H, W].
        bgr_add : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 0
            BGR values to be added to the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_mult : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 1
            BGR values to be multiplied by the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_to_rgb : bool, default False
            If True, flip the channels to convert from BGR to RGB.
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to resize the images.
            If not provided, a new one will be created based on the given resize_mode.
        resize_mode : str, default "pad"
            How to resize the input. Accepted values are "pad" and "interpolation".
        target_size : Optional[Tuple[int, int]], default None
            If given, the images will be resized to this size, instead of calculating a multiple of self.output_stride.
        pad_mode : str, default "replicate"
            Used if resize_mode == "pad". How to pad the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.pad.
        pad_value : float, default 0.0
            Used if resize_mode == "pad" and pad_mode == "constant". The value to fill in the padded area.
        pad_two_side : bool, default True
            Used if resize_mode == "pad". If True, half of the padding goes to left/top and the rest to right/bottom. Otherwise, all the padding goes to the bottom right.
        interpolation_mode : str, default "bilinear"
            Used if resize_mode == "interpolation". How to interpolate the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.interpolate.
        interpolation_align_corners : bool, default True
            Used if resize_mode == "interpolation". See 'align_corners' in https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

        Returns
        -------
        torch.Tensor
            A copy of the input images after applying all of the pre-processing steps.
        Union[InputPadder, InputScaler]
            An instance of InputPadder or InputScaler that was used to resize the images.
            Can be used to reverse the resizing operations.
        """
        bgr_add = bgr_val_as_tensor(
            bgr_add, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images = images + bgr_add
        bgr_mult = bgr_val_as_tensor(
            bgr_mult, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images *= bgr_mult
        if bgr_to_rgb:
            images = torch.flip(images, [-3])

        stride = self.output_stride if stride is None else stride
        if target_size is not None:
            stride = None

        if image_resizer is None:
            if resize_mode == "pad":
                image_resizer = InputPadder(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    pad_mode=pad_mode,
                    two_side_pad=pad_two_side,
                    pad_value=pad_value,
                )
            elif resize_mode == "interpolation":
                image_resizer = InputScaler(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    interpolation_mode=interpolation_mode,
                    interpolation_align_corners=interpolation_align_corners,
                )
            else:
                raise ValueError(
                    f"resize_mode must be one of (pad, interpolation). Found: {resize_mode}."
                )

        images = image_resizer.fill(images)
        images = images.contiguous()
        return images, image_resizer

    def postprocess_predictions(
        self,
        prediction: torch.Tensor,
        image_resizer: Optional[Union[InputPadder, InputScaler]],
        is_flow: bool,
    ) -> torch.Tensor:
        """Simple resizing post-processing. Just use image_resizer to revert the resizing operations.

        Parameters
        ----------
        prediction : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., C, H, W].
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to reverse the resizing done to the inputs.
            Typically, this will be the instance returned by self.preprocess_images().
        is_flow : bool
            Indicates if prediction is an optical flow prediction of not.
            Only used if image_resizer is an instance of InputScaler, in which case the flow values need to be scaled.

        Returns
        -------
        torch.Tensor
            A copy of the prediction after reversing the resizing.
        """
        if isinstance(image_resizer, InputScaler):
            return image_resizer.unfill(prediction, is_flow=is_flow)
        else:
            return image_resizer.unfill(prediction)

class LiteFlowNet3PseudoReg(LiteFlowNet3):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3-kitti-b5d32443.ckpt"
    }

    def __init__(
        self,
        div_flow: float = 20,
        use_pseudo_regularization: bool = True,
        use_s_version: bool = False,
        **kwargs,
    ):
        super().__init__(
            div_flow,
            use_pseudo_regularization,
            use_s_version,
            **kwargs,
        )


class LiteFlowNet3S(LiteFlowNet3):
    pretrained_checkpoints = {
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3s-sintel-89793e34.ckpt"
    }

    def __init__(
        self,
        div_flow: float = 20,
        use_pseudo_regularization: bool = False,
        use_s_version: bool = True,
        **kwargs,
    ):
        super().__init__(
            div_flow,
            use_pseudo_regularization,
            use_s_version,
            **kwargs,
        )


class LiteFlowNet3SPseudoReg(LiteFlowNet3):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3s-kitti-5dffb261.ckpt"
    }

    def __init__(
        self,
        div_flow: float = 20,
        use_pseudo_regularization: bool = True,
        use_s_version: bool = True,
        **kwargs,
    ):
        super().__init__(
            div_flow,
            use_pseudo_regularization,
            use_s_version,
            **kwargs,
        )


class liteflownet3(LiteFlowNet3):
    pass



class liteflownet3_pseudoreg(LiteFlowNet3PseudoReg):
    pass



class liteflownet3s(LiteFlowNet3S):
    pass


class liteflownet3s_pseudoreg(LiteFlowNet3SPseudoReg):
    pass

if __name__ == "__main__":
    import torch.onnx
    import os
    
    model = liteflownet3s()
    print(model)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # ONNX模型导出
    def export_onnx_model():
        """导出ONNX模型用于可视化"""
        model.eval()  # 设置为评估模式
        
        # 创建示例输入 (batch_size=1, 2张图片, 3通道, 高度384, 宽度512)
        dummy_images = torch.randn(1, 2, 3, 384, 512)
        dummy_input = {'images': dummy_images}  # 模型期望字典输入
        
        # 导出路径
        onnx_path = "liteflownet3s_model.onnx"
        
        print(f"正在导出ONNX模型到: {onnx_path}")
        
        # 创建包装器模型以处理字典输入输出
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, images):
                outputs = self.model({'images': images})
                return outputs['flows'], outputs['confs']
        
        wrapper_model = ONNXWrapper(model)
        
        # 导出ONNX模型
        torch.onnx.export(
            wrapper_model,                  # 包装后的模型
            dummy_images,                   # 直接传递图像张量
            onnx_path,                      # 输出文件路径
            export_params=True,             # 导出参数
            opset_version=16,               # 使用opset版本16以支持grid_sampler
            do_constant_folding=True,       # 常量折叠优化
            input_names=['images'],         # 输入名称
            output_names=['flows', 'confs'], # 输出名称
            dynamic_axes={
                'images': {0: 'batch_size', 2: 'height', 3: 'width'},
                'flows': {0: 'batch_size', 2: 'height', 3: 'width'},
                'confs': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"ONNX模型导出成功: {onnx_path}")
        print(f"文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        print("\n可视化方法:")
        print("1. 在线可视化: https://netron.app")
        print("2. 本地安装: pip install netron && netron liteflownet3_model.onnx")
        print("3. 命令行: python -m netron liteflownet3_model.onnx")
        
        return onnx_path
    
    # 执行导出
    try:
        export_onnx_model()
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        print("请确保安装了torch和onnx: pip install torch onnx")
        import traceback
        traceback.print_exc()