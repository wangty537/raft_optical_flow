import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """
    光流预测头：将特征转换为光流增量
    
    这是一个简单的两层卷积网络，用于从GRU的隐藏状态预测光流增量。
    输出2个通道，分别对应x和y方向的光流分量。
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        # 第一层：特征变换
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        # 第二层：输出光流（2个通道：x和y方向）
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: 输入特征 shape=[N, input_dim, H//8, W//8]
        Returns:
            光流增量 shape=[N, 2, H//8, W//8]
        """
        return self.conv2(self.relu(self.conv1(x)))  # shape: [N, 2, H//8, W//8]

class ConvGRU(nn.Module):
    """
    卷积门控循环单元（Convolutional Gated Recurrent Unit）
    
    这是RAFT中用于迭代更新的核心组件，结合了GRU的记忆机制和卷积的空间处理能力。
    通过门控机制控制信息的流动，实现对光流的渐进式优化。
    
    GRU公式：
    - 更新门：z = σ(W_z * [h, x])
    - 重置门：r = σ(W_r * [h, x]) 
    - 候选状态：q = tanh(W_q * [r⊙h, x])
    - 新状态：h' = (1-z)⊙h + z⊙q
    """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        # 更新门：控制新信息的融入程度
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        # 重置门：控制历史信息的保留程度
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        # 候选状态：生成新的候选信息
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        """
        Args:
            h: 隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            x: 输入特征 shape=[N, input_dim, H//8, W//8]
        Returns:
            更新后的隐藏状态 shape=[N, hidden_dim, H//8, W//8]
        """
        # 拼接隐藏状态和输入特征
        hx = torch.cat([h, x], dim=1)  # shape: [N, hidden_dim+input_dim, H//8, W//8]

        # 计算更新门：决定保留多少旧信息和接受多少新信息
        z = torch.sigmoid(self.convz(hx))  # shape: [N, hidden_dim, H//8, W//8]
        # 计算重置门：决定忘记多少历史信息
        r = torch.sigmoid(self.convr(hx))  # shape: [N, hidden_dim, H//8, W//8]
        # 计算候选状态：基于重置后的历史信息和当前输入
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))  # shape: [N, hidden_dim, H//8, W//8]

        # 更新隐藏状态：线性插值融合旧状态和新候选状态
        h = (1-z) * h + z * q  # shape: [N, hidden_dim, H//8, W//8]
        return h

class SepConvGRU(nn.Module):
    """
    分离卷积门控循环单元（Separable Convolutional GRU）
    
    这是ConvGRU的改进版本，使用分离卷积（先水平后垂直）来减少计算量并提高效率。
    分离卷积将一个大的卷积核分解为两个较小的一维卷积核，在保持感受野的同时减少参数量。
    
    优势：
    1. 参数量更少：5x5卷积 → 1x5 + 5x1卷积
    2. 计算效率更高
    3. 更好的梯度流动
    """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        # 水平方向的门控卷积（1x5卷积核）
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        # 垂直方向的门控卷积（5x1卷积核）
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        """
        Args:
            h: 隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            x: 输入特征 shape=[N, input_dim, H//8, W//8]
        Returns:
            更新后的隐藏状态 shape=[N, hidden_dim, H//8, W//8]
        """
        # === 第一步：水平方向的GRU更新 ===
        hx = torch.cat([h, x], dim=1)  # shape: [N, hidden_dim+input_dim, H//8, W//8]
        z = torch.sigmoid(self.convz1(hx))  # 水平更新门
        r = torch.sigmoid(self.convr1(hx))  # 水平重置门
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))  # 水平候选状态
        h = (1-z) * h + z * q  # 水平方向更新

        # === 第二步：垂直方向的GRU更新 ===
        hx = torch.cat([h, x], dim=1)  # 使用更新后的h
        z = torch.sigmoid(self.convz2(hx))  # 垂直更新门
        r = torch.sigmoid(self.convr2(hx))  # 垂直重置门
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))  # 垂直候选状态
        h = (1-z) * h + z * q  # 垂直方向更新

        return h  # shape: [N, hidden_dim, H//8, W//8]

class SmallMotionEncoder(nn.Module):
    """
    小型运动编码器：用于小模型的运动特征编码
    
    将相关性特征和光流特征编码为运动特征，供GRU使用。
    这是轻量级版本，参数量较少，适用于资源受限的场景。
    """
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        # 计算相关性特征的通道数：层数 × (2×半径+1)²
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        
        # 相关性特征编码：1×1卷积降维
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        
        # 光流特征编码：两层卷积
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)  # 大感受野捕获运动模式
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)  # 进一步特征提取
        
        # 融合特征：相关性(96) + 光流(32) = 128 → 80
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow: 当前光流 shape=[N, 2, H//8, W//8]
            corr: 相关性特征 shape=[N, cor_planes, H//8, W//8]
        Returns:
            运动特征 shape=[N, 82, H//8, W//8] (80个编码特征 + 2个原始光流)
        """
        # 编码相关性特征
        cor = F.relu(self.convc1(corr))  # shape: [N, 96, H//8, W//8]
        
        # 编码光流特征
        flo = F.relu(self.convf1(flow))  # shape: [N, 64, H//8, W//8]
        flo = F.relu(self.convf2(flo))   # shape: [N, 32, H//8, W//8]
        
        # 拼接相关性和光流特征
        cor_flo = torch.cat([cor, flo], dim=1)  # shape: [N, 128, H//8, W//8]
        
        # 融合特征
        out = F.relu(self.conv(cor_flo))  # shape: [N, 80, H//8, W//8]
        
        # 拼接编码特征和原始光流（保留原始信息）
        return torch.cat([out, flow], dim=1)  # shape: [N, 82, H//8, W//8]

class BasicMotionEncoder(nn.Module):
    """
    基础运动编码器：用于标准模型的运动特征编码
    
    相比SmallMotionEncoder，这个版本有更多的参数和更深的网络结构，
    能够提取更丰富的运动特征，适用于对精度要求较高的场景。
    """
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        # 计算相关性特征的通道数
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        
        # 相关性特征编码：两层卷积，更深的特征提取
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)  # 1×1卷积降维
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)         # 3×3卷积空间特征
        
        # 光流特征编码：两层卷积，更大的特征容量
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)   # 大感受野
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)  # 特征精炼
        
        # 融合特征：相关性(192) + 光流(64) = 256 → 126
        # 输出126维，加上原始光流2维，总共128维
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow: 当前光流 shape=[N, 2, H//8, W//8]
            corr: 相关性特征 shape=[N, cor_planes, H//8, W//8]
        Returns:
            运动特征 shape=[N, 128, H//8, W//8] (126个编码特征 + 2个原始光流)
        """
        # 编码相关性特征（两层网络）
        cor = F.relu(self.convc1(corr))  # shape: [N, 256, H//8, W//8]
        cor = F.relu(self.convc2(cor))   # shape: [N, 192, H//8, W//8]
        
        # 编码光流特征（两层网络）
        flo = F.relu(self.convf1(flow))  # shape: [N, 128, H//8, W//8]
        flo = F.relu(self.convf2(flo))   # shape: [N, 64, H//8, W//8]

        # 拼接相关性和光流特征
        cor_flo = torch.cat([cor, flo], dim=1)  # shape: [N, 256, H//8, W//8]
        
        # 融合特征
        out = F.relu(self.conv(cor_flo))  # shape: [N, 126, H//8, W//8]
        
        # 拼接编码特征和原始光流
        return torch.cat([out, flow], dim=1)  # shape: [N, 128, H//8, W//8]

class SmallUpdateBlock(nn.Module):
    """
    小型更新块：用于轻量级RAFT模型
    
    这是RAFT迭代更新的核心组件的轻量版本，包含：
    1. 运动编码器：编码光流和相关性特征
    2. ConvGRU：更新隐藏状态
    3. 光流头：预测光流增量
    
    特点：参数量少，计算效率高，适用于资源受限场景
    """
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        # 运动特征编码器
        self.encoder = SmallMotionEncoder(args)
        # GRU更新模块：输入维度 = 上下文特征(64) + 运动特征(82) = 146
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        # 光流预测头
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        """
        Args:
            net: GRU隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            inp: 上下文特征 shape=[N, 64, H//8, W//8]
            corr: 相关性特征 shape=[N, cor_planes, H//8, W//8]
            flow: 当前光流 shape=[N, 2, H//8, W//8]
        Returns:
            net: 更新后的隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            mask: 上采样掩码 (小模型中为None)
            delta_flow: 光流增量 shape=[N, 2, H//8, W//8]
        """
        # 编码运动特征
        motion_features = self.encoder(flow, corr)  # shape: [N, 82, H//8, W//8]
        
        # 拼接上下文特征和运动特征
        inp = torch.cat([inp, motion_features], dim=1)  # shape: [N, 146, H//8, W//8]
        
        # GRU更新
        net = self.gru(net, inp)  # shape: [N, hidden_dim, H//8, W//8]
        
        # 预测光流增量
        delta_flow = self.flow_head(net)  # shape: [N, 2, H//8, W//8]

        # 小模型不使用学习的上采样掩码
        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    """
    基础更新块：用于标准RAFT模型
    
    这是RAFT迭代更新的核心组件的标准版本，包含：
    1. 运动编码器：编码光流和相关性特征
    2. SepConvGRU：使用分离卷积的GRU更新隐藏状态
    3. 光流头：预测光流增量
    4. 掩码头：预测上采样掩码（用于学习的上采样）
    
    特点：精度高，包含学习的上采样机制
    """
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        
        # 运动特征编码器
        self.encoder = BasicMotionEncoder(args)
        
        # 分离卷积GRU：输入维度 = 上下文特征(128) + 运动特征(128) = 256
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        
        # 光流预测头
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # 上采样掩码预测头：预测64个3×3卷积核的权重
        # 64*9 = 576个参数，用于8倍上采样的每个输出像素
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        """
        Args:
            net: GRU隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            inp: 上下文特征 shape=[N, 128, H//8, W//8]
            corr: 相关性特征 shape=[N, cor_planes, H//8, W//8]
            flow: 当前光流 shape=[N, 2, H//8, W//8]
            upsample: 是否使用上采样（保留参数兼容性）
        Returns:
            net: 更新后的隐藏状态 shape=[N, hidden_dim, H//8, W//8]
            mask: 上采样掩码 shape=[N, 576, H//8, W//8]
            delta_flow: 光流增量 shape=[N, 2, H//8, W//8]
        """
        # 编码运动特征
        motion_features = self.encoder(flow, corr)  # shape: [N, 128, H//8, W//8]
        
        # 拼接上下文特征和运动特征
        inp = torch.cat([inp, motion_features], dim=1)  # shape: [N, 256, H//8, W//8]

        # 分离卷积GRU更新
        net = self.gru(net, inp)  # shape: [N, hidden_dim, H//8, W//8]
        
        # 预测光流增量
        delta_flow = self.flow_head(net)  # shape: [N, 2, H//8, W//8]

        # 预测上采样掩码，缩放0.25平衡梯度
        mask = .25 * self.mask(net)  # shape: [N, 576, H//8, W//8]
        
        return net, mask, delta_flow



