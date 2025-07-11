import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    """
    相关性块：RAFT中用于计算特征相关性的核心组件
    
    这个类实现了多尺度相关性金字塔，用于在不同分辨率下计算两帧图像特征之间的相关性。
    相关性金字塔能够捕获不同尺度的运动信息，从大位移到小位移。
    
    工作原理：
    1. 计算两个特征图之间的全局相关性
    2. 构建多尺度相关性金字塔（通过平均池化降采样）
    3. 在查询时，根据给定坐标在每个尺度上采样相关性特征
    4. 在每个尺度上采样一个局部窗口（半径为radius）
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        初始化相关性块
        
        Args:
            fmap1: 第一帧特征图 shape=[N, C, H//8, W//8]
            fmap2: 第二帧特征图 shape=[N, C, H//8, W//8]
            num_levels: 金字塔层数，默认4层
            radius: 采样半径，默认4（采样9×9窗口）
        """
        self.num_levels = num_levels  # 金字塔层数
        self.radius = radius          # 采样半径
        self.corr_pyramid = []        # 存储多尺度相关性

        # === 计算全局相关性 ===
        # 计算两个特征图之间的全对全相关性
        corr = CorrBlock.corr(fmap1, fmap2)  # shape: [N, H, W, 1, H, W]

        # 重塑为适合池化的形状
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)  # shape: [N*H*W, 1, H, W]
        
        # === 构建相关性金字塔 ===
        # 第0层：原始分辨率
        self.corr_pyramid.append(corr)
        
        # 第1-3层：通过平均池化逐步降采样
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)  # 每层分辨率减半
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        根据给定坐标查询相关性特征
        
        Args:
            coords: 查询坐标 shape=[N, 2, H//8, W//8]，表示每个像素在第二帧中的对应位置
        Returns:
            相关性特征 shape=[N, num_levels*(2*radius+1)^2, H//8, W//8]
        """
        r = self.radius
        # 转换坐标维度：[N, 2, H, W] → [N, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []  # 存储每层的相关性特征
        
        # === 在每个金字塔层级采样相关性 ===
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # 当前层的相关性 shape: [N*H*W, 1, H_i, W_i]
            
            # 创建采样偏移网格：在半径r范围内的所有整数位置
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)  # [-r, ..., 0, ..., r]
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)  # [-r, ..., 0, ..., r]
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)      # shape: [2*r+1, 2*r+1, 2]

            # 计算当前层级的采样坐标
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  # 缩放到当前层级
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)                  # 偏移网格
            coords_lvl = centroid_lvl + delta_lvl  # 最终采样坐标 shape: [N*H*W, 2*r+1, 2*r+1, 2]

            # 双线性插值采样相关性
            corr = bilinear_sampler(corr, coords_lvl)  # shape: [N*H*W, 1, 2*r+1, 2*r+1]
            corr = corr.view(batch, h1, w1, -1)        # 重塑为 [N, H, W, (2*r+1)^2]
            out_pyramid.append(corr)

        # 拼接所有层级的特征
        out = torch.cat(out_pyramid, dim=-1)  # shape: [N, H, W, num_levels*(2*r+1)^2]
        # 转换回标准格式：[N, H, W, C] → [N, C, H, W]
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        计算两个特征图之间的全局相关性
        
        这个函数计算第一帧每个像素与第二帧所有像素之间的相关性（内积）。
        相关性衡量了特征的相似程度，用于后续的光流估计。
        
        Args:
            fmap1: 第一帧特征图 shape=[N, C, H, W]
            fmap2: 第二帧特征图 shape=[N, C, H, W]
        Returns:
            相关性张量 shape=[N, H, W, 1, H, W]
            其中corr[b,i,j,0,u,v]表示第一帧(i,j)位置与第二帧(u,v)位置的相关性
        """
        batch, dim, ht, wd = fmap1.shape
        
        # 重塑特征图：[N, C, H, W] → [N, C, H*W]
        fmap1 = fmap1.view(batch, dim, ht*wd)  # shape: [N, C, H*W]
        fmap2 = fmap2.view(batch, dim, ht*wd)  # shape: [N, C, H*W]
        
        # 计算相关性矩阵：每个位置与所有位置的内积
        # fmap1.transpose(1,2): [N, H*W, C]
        # fmap2: [N, C, H*W]
        # 结果: [N, H*W, H*W]
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        
        # 重塑为6维张量：[N, H*W, H*W] → [N, H, W, 1, H, W]
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        
        # 归一化：除以特征维度的平方根（类似注意力机制中的缩放）
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    """
    替代相关性块：使用CUDA优化的相关性计算
    
    这是CorrBlock的优化版本，使用自定义CUDA核心来加速相关性计算。
    相比标准版本，它不预计算全局相关性，而是在查询时动态计算，
    这样可以节省大量内存，特别是对于高分辨率图像。
    
    注意：需要编译alt_cuda_corr扩展才能使用
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        初始化替代相关性块
        
        Args:
            fmap1: 第一帧特征图 shape=[N, C, H//8, W//8]
            fmap2: 第二帧特征图 shape=[N, C, H//8, W//8]
            num_levels: 金字塔层数，默认4层
            radius: 采样半径，默认4
        """
        self.num_levels = num_levels
        self.radius = radius

        # 构建特征金字塔：存储不同分辨率的特征图对
        self.pyramid = [(fmap1, fmap2)]  # 第0层：原始分辨率
        
        # 构建多尺度特征金字塔
        for i in range(self.num_levels):
            # 对两个特征图同时进行平均池化降采样
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        """
        使用CUDA优化的相关性计算
        
        Args:
            coords: 查询坐标 shape=[N, 2, H//8, W//8]
        Returns:
            相关性特征 shape=[N, num_levels*(2*radius+1)^2, H//8, W//8]
        """
        # 转换坐标格式
        coords = coords.permute(0, 2, 3, 1)  # [N, 2, H, W] → [N, H, W, 2]
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]  # 特征维度

        corr_list = []
        # 在每个金字塔层级计算相关性
        for i in range(self.num_levels):
            r = self.radius
            
            # 获取当前层级的特征图（转换为CUDA函数需要的格式）
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()  # 始终使用第0层的fmap1
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()  # 使用第i层的fmap2

            # 调整坐标到当前层级的分辨率
            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            
            # 使用CUDA优化的相关性计算
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        # 拼接所有层级的相关性特征
        corr = torch.stack(corr_list, dim=1)  # shape: [B, num_levels, (2*r+1)^2, H, W]
        corr = corr.reshape(B, -1, H, W)      # shape: [B, num_levels*(2*r+1)^2, H, W]
        
        # 归一化
        return corr / torch.sqrt(torch.tensor(dim).float())
