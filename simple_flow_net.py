import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleFlowNet(nn.Module):
    """
    简单高效的光流估计网络 - 基于多尺度特征匹配的光流估计
    
    网络架构设计原理：
    1. 特征提取器：使用轻量级残差网络提取多尺度特征金字塔
       - 捕获不同尺度的运动模式：细粒度小运动到大尺度运动
       - 使用BatchNorm和残差连接提高训练稳定性
    
    2. 相关性计算：计算两帧特征之间的相关性体积
       - 在9x9搜索窗口内计算特征相关性
       - 使用L2归一化提高匹配稳定性
    
    3. 光流解码器：从相关性特征解码出光流向量
       - 使用CNN学习从相关性模式到运动向量的映射
       - 支持残差学习，细化粗尺度预测
    
    4. 多尺度预测：从粗到细逐步细化光流预测
       - 粗尺度：捕获大运动，提供全局约束
       - 细尺度：细化细节，提高精度
    
    技术特点：
    - 轻量级设计：参数量小，推理速度快
    - 多尺度融合：结合不同分辨率的预测
    - 运动补偿：使用warp操作提高匹配精度
    - 残差学习：逐步细化光流预测
    """
    
    def __init__(self, input_channels=3, feature_dim=64):
        super(SimpleFlowNet, self).__init__()
        self.feature_dim = feature_dim
        
        # 特征提取器 - 编码器部分
        self.feature_extractor = FeatureExtractor(input_channels, feature_dim)
        
        # 相关性计算模块
        self.correlation = CorrelationLayer()
        
        # 光流解码器
        self.flow_decoder = FlowDecoder(feature_dim)
        
        # 上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, img1, img2):
        """
        前向传播 - 多尺度光流估计的核心算法
        
        原理：
        1. 特征提取：使用CNN提取两帧图像的多尺度特征
        2. 相关性计算：计算特征间的相关性，找到像素对应关系
        3. 光流解码：从相关性特征解码出光流向量
        4. 多尺度融合：从粗到细逐步细化光流预测
        
        Args:
            img1: 第一帧图像 [B, 3, H, W] - B:批次大小, 3:RGB通道, H,W:图像高宽
            img2: 第二帧图像 [B, 3, H, W]
        Returns:
            flow_predictions: 多尺度光流预测列表，每个元素shape为[B, 2, H_i, W_i]
                            2表示x,y方向的光流分量
        """
        # 提取多尺度特征 - 获得3个尺度的特征金字塔
        # features1/2: list of [B, C, H/2^i, W/2^i], i=1,2,3
        # 例如：[B,32,H/2,W/2], [B,64,H/4,W/4], [B,128,H/8,W/8]
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        flow_predictions = []  # 存储各尺度的光流预测
        
        # 从最低分辨率开始预测光流 - 粗到细的策略
        # i=2,1,0 对应 1/8, 1/4, 1/2 分辨率
        for i in range(len(features1)-1, -1, -1):
            # 当前尺度的特征 feat1,feat2: [B, C_i, H_i, W_i]
            feat1 = features1[i]  # 第一帧特征
            feat2 = features2[i]  # 第二帧特征
            
            # 计算相关性 - 寻找像素对应关系
            
          
            # 如果不是最低分辨率，需要利用之前的光流预测进行细化
            if i < len(features1) - 1:
                # 上采样之前的光流预测到当前分辨率
                # prev_flow: [B, 2, H_{i+1}, W_{i+1}] -> [B, 2, H_i, W_i]
                prev_flow = flow_predictions[-1]
                # 确保prev_flow与当前特征尺寸匹配 - 处理尺寸不一致问题
               
                # 计算缩放因子并调整光流值
                scale_h = feat1.shape[-2] / prev_flow.shape[-2]
                scale_w = feat1.shape[-1] / prev_flow.shape[-1]
                prev_flow = F.interpolate(prev_flow, size=feat1.shape[-2:], mode='bilinear', align_corners=False)
                prev_flow[:, 0] *= scale_w  # x方向光流按宽度缩放
                prev_flow[:, 1] *= scale_h  # y方向光流按高度缩放

                # 根据光流对第二帧特征进行warp - 运动补偿
                # feat2_warped: [B, C_i, H_i, W_i] - 按光流变形后的特征
                feat2_warped = self.warp_features(feat2, prev_flow*20)
                
                # 重新计算相关性 - 基于运动补偿后的特征
                # correlation: [B, 81, H_i, W_i]
                correlation = self.correlation(feat1, feat2_warped)
                
                # flow_input: [B, 83, H_i, W_i] (81+2=相关性+光流)
                flow_input = torch.cat([correlation, prev_flow], dim=1)
               
            else:
                # 最低分辨率时只使用相关性
                # correlation: [B, 81, H_i, W_i] (81=9x9搜索窗口)
                correlation = self.correlation(feat1, feat2)
                # flow_input: [B, 81, H_i, W_i]
                flow_input = correlation
            
            # 解码光流 - 从特征解码出光流向量
            # flow: [B, 2, H_i, W_i] - 残差光流
            flow = self.flow_decoder(flow_input)
            
            # 如果有之前的光流预测，则相加得到最终光流
            # 这是残差学习的思想：flow_final = flow_coarse + flow_residual
            if i < len(features1) - 1:
                flow = flow + prev_flow  # [B, 2, H_i, W_i]
            #print(i, feat1.shape, feat2.shape, correlation.shape, flow.shape)
            flow_predictions.append(flow)
        
        # 反转列表，使其从低分辨率到高分辨率排列
        # 循环处理顺序：1/8 -> 1/4 -> 1/2 (从粗到细)
        flow_predictions = [flow * 20 for flow in flow_predictions]
        return flow_predictions
    
    def warp_features(self, features, flow):
        """
        根据光流对特征进行warp操作 - 运动补偿的核心实现
        
        原理：
        1. 创建像素坐标网格
        2. 根据光流偏移坐标
        3. 使用双线性插值采样变形后的特征
        
        这个操作实现了"如果像素从位置A移动到位置B，那么B位置的特征应该来自A位置"
        
        Args:
            features: 待变形的特征 [B, C, H, W]
            flow: 光流向量 [B, 2, H_flow, W_flow]
        Returns:
            warped_features: 变形后的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape  # 特征的维度信息
        assert flow.shape[-2:] == (H, W), "光流尺寸与特征尺寸不匹配"
        # # 确保flow尺寸与features匹配 - 处理多尺度问题
        # if flow.shape[-2:] != (H, W):
        #     # 将光流插值到特征的尺寸
        #     flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
        #     # 调整光流值以匹配新的尺寸 - 光流值需要按比例缩放
        #     scale_h = H / flow.shape[-2] if flow.shape[-2] != H else 1.0
        #     scale_w = W / flow.shape[-1] if flow.shape[-1] != W else 1.0
        #     flow[:, 0] *= scale_w  # x方向光流缩放
        #     flow[:, 1] *= scale_h  # y方向光流缩放
        
        # 创建像素坐标网格 - 标准坐标系统
        # grid_x, grid_y: [H, W] - 每个像素的x,y坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        # grid: [2, H, W] - 坐标网格，0通道为x坐标，1通道为y坐标
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        # grid: [B, 2, H, W] - 扩展到批次维度
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).to(features.device)
        
        # 添加光流偏移 - 计算新的采样坐标
        # new_grid: [B, 2, H, W] - 偏移后的坐标
        # new_grid[b,0,i,j] = 原x坐标 + x方向光流
        # new_grid[b,1,i,j] = 原y坐标 + y方向光流
        new_grid = grid + flow
        
        # 归一化到[-1, 1] - grid_sample要求的坐标范围
        # PyTorch的grid_sample使用[-1,1]坐标系统
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0  # x坐标归一化
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0  # y坐标归一化
        
        # 调整维度顺序为grid_sample要求的格式
        # new_grid: [B, H, W, 2] - 最后一维为(x,y)坐标
        new_grid = new_grid.permute(0, 2, 3, 1)
        
        # 进行双线性插值采样 - 实现特征变形
        # warped_features: [B, C, H, W] - 按光流变形后的特征
        warped_features = F.grid_sample(features, new_grid, align_corners=True)
        
        return warped_features


class FeatureExtractor(nn.Module):
    """
    轻量级特征提取器 - 构建多尺度特征金字塔
    
    设计原理：
    使用残差连接构建深度网络，避免梯度消失问题。
    通过逐步下采样构建特征金字塔，每个尺度捕获不同大小的运动模式。
    
    网络结构：
    - 输入：RGB图像 [B, 3, H, W]
    - 1/2分辨率：[B, 32, H/2, W/2] - 高分辨率特征，捕获细粒度运动
    - 1/4分辨率：[B, 64, H/4, W/4] - 中分辨率特征，捕获中等运动
    - 1/8分辨率：[B, 128, H/8, W/8] - 低分辨率特征，捕获大尺度运动
    
    技术特点：
    - 残差连接：解决深度网络训练问题
    - BatchNorm：加速收敛，提高稳定性
    - 渐进下采样：保持特征表达能力
    """
    
    def __init__(self, input_channels, feature_dim):
        super(FeatureExtractor, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim//2, 7, stride=2, padding=3),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.res_block1 = ResidualBlock(feature_dim//2, feature_dim//2)
        self.res_block2 = ResidualBlock(feature_dim//2, feature_dim, stride=2)
        self.res_block3 = ResidualBlock(feature_dim, feature_dim)
        self.res_block4 = ResidualBlock(feature_dim, feature_dim*2, stride=2)
        self.res_block5 = ResidualBlock(feature_dim*2, feature_dim*2)
        
    def forward(self, x):
        """
        提取多尺度特征 - 构建特征金字塔
        
        原理：
        通过逐步下采样构建多尺度特征表示，每个尺度捕获不同大小的运动模式：
        - 1/2分辨率：捕获细粒度的小运动
        - 1/4分辨率：捕获中等尺度的运动
        - 1/8分辨率：捕获大尺度的运动
        
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 多尺度特征列表，包含3个尺度 [1/2, 1/4, 1/8]
                     [0]: [B, 32, H/2, W/2]   - 高分辨率特征
                     [1]: [B, 64, H/4, W/4]   - 中分辨率特征  
                     [2]: [B, 128, H/8, W/8]  - 低分辨率特征
        """
        features = []  # 存储多尺度特征
        
        # 1/2 分辨率特征提取
        # x: [B, 3, H, W] -> [B, 32, H/2, W/2]
        x = self.conv1(x)        # 7x7卷积+BN+ReLU，stride=2下采样
        x = self.res_block1(x)   # 残差块，保持通道数和分辨率
        features.append(x)       # 添加到特征列表
        
        # 1/4 分辨率特征提取  
        # x: [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
        x = self.res_block2(x)   # 残差块，stride=2下采样，通道数翻倍
        x = self.res_block3(x)   # 残差块，保持通道数和分辨率
        features.append(x)       # 添加到特征列表
        
        # 1/8 分辨率特征提取
        # x: [B, 64, H/4, W/4] -> [B, 128, H/8, W/8] 
        x = self.res_block4(x)   # 残差块，stride=2下采样，通道数翻倍
        x = self.res_block5(x)   # 残差块，保持通道数和分辨率
        features.append(x)       # 添加到特征列表
        
        return features


class ResidualBlock(nn.Module):
    """
    残差块 - 深度网络的基础构建单元
    
    设计原理：
    残差学习通过跳跃连接解决深度网络的梯度消失和退化问题。
    将学习目标从H(x)改为F(x) = H(x) - x，即学习残差映射。
    
    结构组成：
    - 主路径：Conv -> BN -> ReLU -> Conv -> BN
    - 跳跃连接：恒等映射或1x1投影（当维度不匹配时）
    - 输出：主路径 + 跳跃连接 -> ReLU
    
    优势：
    - 解决梯度消失：跳跃连接提供梯度直接传播路径
    - 避免网络退化：至少能学到恒等映射
    - 提高表达能力：学习残差比学习完整映射更容易
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同或步长不为1，需要调整shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """
        残差块前向传播 - 实现残差学习
        
        原理：
        残差学习通过跳跃连接解决深度网络的梯度消失问题。
        学习目标从H(x)变为F(x) = H(x) - x，即学习残差映射。
        
        数学表达：
        y = F(x, {Wi}) + x
        其中F(x, {Wi})是残差映射，x是恒等映射
        
        Args:
            x: 输入特征 [B, in_channels, H, W]
        Returns:
            out: 输出特征 [B, out_channels, H', W']
                 H', W'可能因stride而改变
        """
        # 计算跳跃连接 - 恒等映射或投影映射
        # residual: [B, out_channels, H', W'] - 调整后的输入
        residual = self.shortcut(x)
        
        # 第一个卷积块：Conv + BN + ReLU
        # out: [B, out_channels, H', W'] - 第一层输出
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 第二个卷积块：Conv + BN（无ReLU）
        # out: [B, out_channels, H', W'] - 第二层输出
        out = self.bn2(self.conv2(out))
        
        # 残差连接：F(x) + x
        # 将残差映射与跳跃连接相加
        out += residual
        
        # 最终激活
        out = F.relu(out)
        
        return out


class CorrelationLayer(nn.Module):
    """
    相关性计算层 - 光流估计的核心匹配模块
    
    工作原理：
    在每个像素位置，计算第一帧特征与第二帧特征在搜索窗口内的相关性。
    相关性越高，表示该位置的像素越可能对应，这是光流估计的基础。
    
    算法流程：
    1. 特征归一化：L2归一化使内积等于余弦相似度
    2. 搜索窗口：在9x9窗口内计算所有可能的匹配
    3. 相关性计算：使用特征向量内积作为相似度度量
    4. 输出组织：81个通道，每个对应一个搜索位置
    
    技术细节：
    - 搜索范围：max_displacement=4，对应9x9窗口
    - 相关性度量：归一化特征的内积（余弦相似度）
    - 输出格式：[B, 81, H, W]，81=(2*4+1)^2
    - 通道顺序：按行优先顺序排列搜索位置
    """
    
    def __init__(self, max_displacement=4):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        
    def forward(self, feat1, feat2):
        """
        计算相关性 - 寻找像素对应关系的核心算法
        
        原理：
        在每个像素位置，计算第一帧特征与第二帧特征在搜索窗口内的相关性。
        相关性越高，表示该位置的像素越可能对应。这是光流估计的基础。
        
        搜索策略：
        - 搜索窗口：9x9 (max_displacement=4)
        - 对于每个偏移(dx,dy)，计算特征的内积作为相关性度量
        - 输出81个通道，每个通道对应一个搜索位置的相关性
        
        Args:
            feat1: 第一帧特征 [B, C, H, W] - 参考特征
            feat2: 第二帧特征 [B, C, H, W] - 搜索特征
        Returns:
            correlation: 相关性特征 [B, 81, H, W]
                        81 = (2*4+1)^2 = 9x9搜索窗口
                        每个通道表示对应偏移位置的相关性强度
        """
        B, C, H, W = feat1.shape  # 获取特征维度
        
        # L2归一化特征 - 提高相关性计算的稳定性
        # 归一化后特征向量长度为1，内积等于余弦相似度
        feat1 = F.normalize(feat1, p=2, dim=1)  # [B, C, H, W]
        feat2 = F.normalize(feat2, p=2, dim=1)  # [B, C, H, W]
        
        # 计算相关性 - 遍历搜索窗口
        correlation_list = []  # 存储各偏移位置的相关性
        
        # 遍历搜索窗口：从(-4,-4)到(4,4)
        for dy in range(-self.max_displacement, self.max_displacement + 1):
            for dx in range(-self.max_displacement, self.max_displacement + 1):
                # 计算偏移后的第二帧特征
                # feat2_shifted: [B, C, H, W] - 按(dx,dy)偏移后的特征
                feat2_shifted = self.shift_feature(feat2, dx, dy)
                
                # 计算相关性：特征向量内积
                # corr: [B, 1, H, W] - 当前偏移位置的相关性图
                corr = torch.sum(feat1 * feat2_shifted, dim=1, keepdim=True)
                correlation_list.append(corr)
        
        # 拼接所有相关性通道
        # correlation: [B, 81, H, W] - 完整的相关性体积
        # 通道顺序：[(dx=-4,dy=-4), (dx=-3,dy=-4), ..., (dx=4,dy=4)]
        correlation = torch.cat(correlation_list, dim=1)
        
        return correlation
    
    def shift_feature(self, feat, dx, dy):
        """
        对特征进行偏移 - 实现相关性计算中的搜索
        
        原理：
        将特征图按指定偏移量(dx,dy)进行平移，用于计算不同位置的相关性。
        这模拟了在搜索窗口内寻找最佳匹配的过程。
        
        偏移规则：
        - 正偏移：特征向右/下移动
        - 负偏移：特征向左/上移动
        - 超出边界的区域填充为0
        
        Args:
            feat: 输入特征 [B, C, H, W]
            dx: x方向偏移量（像素）
            dy: y方向偏移量（像素）
        Returns:
            shifted_feat: 偏移后的特征 [B, C, H, W]
        """
        B, C, H, W = feat.shape
        # 创建与输入相同大小的零张量
        shifted_feat = torch.zeros_like(feat)
        
        # 计算目标区域的有效范围
        # 目标区域：偏移后特征要放置的位置
        x_start = max(0, dx)      # 目标x起始位置
        x_end = min(W, W + dx)    # 目标x结束位置
        y_start = max(0, dy)      # 目标y起始位置
        y_end = min(H, H + dy)    # 目标y结束位置
        
        # 计算源区域的有效范围
        # 源区域：原始特征中要复制的部分
        shifted_x_start = max(0, -dx)    # 源x起始位置
        shifted_x_end = min(W, W - dx)   # 源x结束位置
        shifted_y_start = max(0, -dy)    # 源y起始位置
        shifted_y_end = min(H, H - dy)   # 源y结束位置
        
        # 执行特征偏移：将源区域复制到目标区域
        # 只复制有效重叠区域，其余部分保持为0
        shifted_feat[:, :, y_start:y_end, x_start:x_end] = \
            feat[:, :, shifted_y_start:shifted_y_end, shifted_x_start:shifted_x_end]
        
        return shifted_feat


class FlowDecoder(nn.Module):
    """
    光流解码器 - 从相关性特征解码出光流向量
    
    功能原理：
    使用CNN将相关性特征和之前的光流预测解码为光流向量。
    网络学习从相关性模式到运动向量的复杂映射关系。
    
    输入处理：
    - 相关性特征：81通道，来自CorrelationLayer
    - 之前光流：2通道，来自粗尺度预测（可选）
    - 总输入：83通道（81+2）或81通道（仅相关性）
    
    网络结构：
    - 输入层：处理83或81通道输入
    - 隐藏层：逐步细化特征表示（128->64->32通道）
    - 输出层：生成2通道光流向量（x,y方向）
    
    设计考虑：
    - 渐进式细化：通道数逐步减少，特征逐步抽象
    - 残差学习：支持在粗尺度基础上学习细化
    - 轻量级设计：平衡精度和效率
    """
    
    def __init__(self, feature_dim):
        super(FlowDecoder, self).__init__()
        
        # 相关性特征的通道数 (9x9=81 for max_displacement=4)
        corr_channels = 81
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(corr_channels + 2, 128, 3, padding=1),  # +2 for previous flow
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出光流 (2个通道: x和y方向)
        self.flow_conv = nn.Conv2d(32, 2, 3, padding=1)
        
    def forward(self, x):
        """
        解码光流 - 从相关性特征解码出光流向量
        
        原理：
        使用CNN将相关性特征和之前的光流预测解码为光流向量。
        网络学习从相关性模式到运动向量的映射关系。
        
        输入处理：
        - 如果只有相关性特征(81通道)，添加零光流(2通道)作为初始估计
        - 如果有之前的光流预测，直接使用(83通道=81+2)
        
        Args:
            x: 输入特征，两种情况：
               - 相关性特征: [B, 81, H, W] (最低分辨率)
               - 相关性+光流: [B, 83, H, W] (其他分辨率)
        Returns:
            flow: 光流预测 [B, 2, H, W]
                  flow[:,0,:,:] - x方向光流分量
                  flow[:,1,:,:] - y方向光流分量
        """
        # 处理输入特征 - 确保输入包含光流信息
        if x.shape[1] == 81:  # 只有相关性特征的情况(最低分辨率)
            B, _, H, W = x.shape
            # 创建零光流作为初始估计
            # zero_flow: [B, 2, H, W] - 初始光流为0
            zero_flow = torch.zeros(B, 2, H, W, device=x.device)
            # 拼接相关性和零光流: [B, 83, H, W]
            x = torch.cat([x, zero_flow], dim=1)
        
        # 光流解码网络 - 逐步细化特征表示
        # x: [B, 83, H, W] -> [B, 128, H, W]
        x = self.conv1(x)    # 第一层：83->128通道，提取高级特征
        
        # x: [B, 128, H, W] -> [B, 64, H, W] 
        x = self.conv2(x)    # 第二层：128->64通道，进一步细化
        
        # x: [B, 64, H, W] -> [B, 32, H, W]
        x = self.conv3(x)    # 第三层：64->32通道，准备输出
        
        # x: [B, 32, H, W] -> [B, 2, H, W]
        flow = self.flow_conv(x)  # 输出层：32->2通道，生成光流向量
        
        return flow


class SimpleFlowLoss(nn.Module):
    """
    简单高效的光流损失函数 - 多项损失的组合优化
    
    损失设计原理：
    结合多种损失项，平衡光流预测的准确性、平滑性和边缘保持能力。
    每项损失针对光流估计的不同方面进行优化。
    
    损失组成：
    1. 多尺度EPE损失（主要监督信号）：
       - 目标：确保光流预测准确性
       - 度量：端点误差（End-Point Error）
       - 策略：粗尺度权重大，细尺度权重小
    
    2. 平滑性损失（正则化项）：
       - 目标：鼓励光流在同质区域平滑
       - 度量：光流梯度的L1范数
       - 作用：避免噪声，提高视觉质量
    
    3. 边缘感知损失（自适应正则化）：用这个就没必要用平滑性损失
       - 目标：在图像边缘处允许光流不连续
       - 策略：根据图像梯度调整平滑性约束
       - 效果：保持运动边界的清晰度
    
    权重策略：
    - EPE权重：[0.32, 0.08, 0.02] 对应 [1/8, 1/4, 1/2] 分辨率
    - 平滑性权重：0.1（可调节）
    - 边缘感知权重：0.1（可调节）
    """
    
    def __init__(self, weights=[0.32, 0.08, 0.02], smooth_weight=0.1, edge_weight=0.1):
        super(SimpleFlowLoss, self).__init__()
        self.weights = weights  # 多尺度权重
        self.smooth_weight = smooth_weight
        self.edge_weight = edge_weight
        
    def forward(self, flow_preds, flow_gt, valid_mask=None, image=None):
        """
        计算光流损失 - 多项损失的组合优化
        
        损失组成：
        1. 多尺度EPE损失：主要监督信号，确保光流预测准确性
        2. 平滑性损失：正则化项，鼓励光流在非边缘区域平滑
        3. 边缘感知损失：在图像边缘处允许光流不连续
        
        Args:
            flow_preds: 多尺度光流预测列表
                       [flow_1/8, flow_1/4, flow_1/2] 每个元素 [B, 2, H_i, W_i]
            flow_gt: 真实光流 [B, 2, H, W] - 全分辨率真值
            valid_mask: 有效像素掩码 [B, H, W] - 1表示有效，0表示无效
            image: 输入图像 [B, 3, H, W] - 用于边缘感知损失
        Returns:
            total_loss: 总损失标量
            loss_dict: 损失字典，包含各项损失的详细信息
        """
        total_loss = 0.0
        loss_dict = {}  # 记录各项损失
        
        # 1. 多尺度EPE损失 - End-Point Error
        # EPE = ||flow_pred - flow_gt||_2，衡量光流向量的欧氏距离误差
        epe_loss = 0.0
        for i, flow_pred in enumerate(flow_preds):
            # 将真实光流下采样到预测的对应尺度
            # scale_factor: 当前尺度相对于原图的缩放比例
            scale_factor = flow_pred.shape[-1] / flow_gt.shape[-1]
            # flow_gt_scaled: [B, 2, H_i, W_i] - 缩放到对应尺度的真值
            flow_gt_scaled = F.interpolate(flow_gt, size=flow_pred.shape[-2:], mode='bilinear', align_corners=False)
            # 光流值也需要按比例缩放（像素位移随分辨率变化）
            flow_gt_scaled = flow_gt_scaled * scale_factor
            
            # 将有效掩码下采样到对应尺度
            if valid_mask is not None:
                # valid_scaled: [B, H_i, W_i] - 对应尺度的有效掩码
                # 排除无效像素和异常大的位移
                mag = torch.sum(flow_gt**2, dim=1).sqrt()  # 计算光流幅值 shape=[N, H, W]
                valid_mask = (valid_mask >= 0.5) & (mag < 400)  # 有效像素掩码 shape=[N, H, W]
                # valid: True表示有效像素，False表示无效像素

                valid_scaled = F.interpolate(valid_mask.unsqueeze(1).float(), 
                                           size=flow_pred.shape[-2:], mode='nearest').squeeze(1)
            else:
                # 如果没有掩码
                mag = torch.sum(flow_gt**2, dim=1).sqrt()  # 计算光流幅值 shape=[N, H, W]
                valid_mask =  (mag < 400)  # 有效像素掩码 shape=[N, H, W]
                valid_scaled = F.interpolate(valid_mask.unsqueeze(1).float(), 
                                           size=flow_pred.shape[-2:], mode='nearest').squeeze(1)
                
            
            # 计算EPE：光流向量的L2距离
            # epe: [B, H_i, W_i] - 每个像素的端点误差
            epe = torch.sqrt(torch.sum((flow_pred - flow_gt_scaled) ** 2, dim=1))
            # 只在有效像素上计算损失
            epe = epe * valid_scaled
            
            # 多尺度加权：粗尺度权重大，细尺度权重小
            # weights = [0.32, 0.08, 0.02] 对应 [1/8, 1/4, 1/2] 分辨率
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            epe_loss += weight * epe.mean()  # 加权平均EPE
        
        loss_dict['epe'] = epe.mean() # 标量记录最大分辨率的flow的epe
        total_loss += epe_loss
        
        # # 2. 平滑性损失 - 鼓励光流在同质区域平滑
        # # 只对最高分辨率的预测计算，避免重复计算
        # if self.smooth_weight > 0:
        #     flow_pred_hr = flow_preds[-1]  # 最高分辨率预测 [B, 2, H/2, W/2]
        #     smooth_loss = self.compute_smoothness_loss(flow_pred_hr)
        #     loss_dict['smooth'] = smooth_loss
        #     total_loss += self.smooth_weight * smooth_loss
        
        # 3. 边缘感知损失 - 在图像边缘处允许光流不连续
        # 结合图像梯度信息，在边缘处减少平滑性约束
        if self.edge_weight > 0 and image is not None:
            edge_loss = self.compute_edge_aware_loss(flow_preds[-1], image)
            loss_dict['edge'] = edge_loss
            total_loss += self.edge_weight * edge_loss
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def compute_smoothness_loss(self, flow):
        """
        计算平滑性损失 - 鼓励光流在空间上连续
        
        原理：
        计算光流在x和y方向的梯度，使用L1范数作为平滑性度量。
        L1范数相比L2范数对异常值更鲁棒，适合处理运动边界。
        
        Args:
            flow: 光流预测 [B, 2, H, W]
        Returns:
            smooth_loss: 平滑性损失标量
        """
        # 计算光流的空间梯度
        # flow_dx: [B, 2, H, W-1] - x方向梯度（水平相邻像素差）
        flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        # flow_dy: [B, 2, H-1, W] - y方向梯度（垂直相邻像素差）
        flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # L1平滑性损失 - 鼓励光流梯度接近0
        # 分别计算x和y方向的平滑性，然后求和
        smooth_loss = torch.mean(torch.abs(flow_dx)) + torch.mean(torch.abs(flow_dy))
        
        return smooth_loss
    
    def compute_edge_aware_loss(self, flow, image):
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


def create_simple_flow_model(input_channels=3, feature_dim=64):
    """
    创建简单光流模型
    """
    model = SimpleFlowNet(input_channels, feature_dim)
    return model


def create_simple_flow_loss(weights=[0.32, 0.08, 0.02], smooth_weight=0.1, edge_weight=0.1):
    """
    创建简单光流损失函数
    """
    loss_fn = SimpleFlowLoss(weights, smooth_weight, edge_weight)
    return loss_fn


if __name__ == "__main__":
    # 测试网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_simple_flow_model().to(device)
    loss_fn = create_simple_flow_loss()
    
    # 测试数据
    B, C, H, W = 2, 3, 256, 256
    img1 = torch.randn(B, C, H, W).to(device)
    img2 = torch.randn(B, C, H, W).to(device)
    flow_gt = torch.randn(B, 2, H, W).to(device)
    valid_mask = torch.ones(B, H, W).to(device)
    
    # 前向传播
    with torch.no_grad():
        flow_preds = model(img1, img2)
        print(f"模型输出: {len(flow_preds)} 个尺度的光流预测")
        for i, flow in enumerate(flow_preds):
            print(f"尺度 {i}: {flow.shape}")
    
    # 计算损失
    total_loss, loss_dict = loss_fn(flow_preds, flow_gt, valid_mask, img1)
    print(f"\n损失计算:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")