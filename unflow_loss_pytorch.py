import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# 假设这些函数在其他模块中实现，这里提供占位符
# from ..ops import backward_warp, forward_warp
# from .image_warp import image_warp

# 常量定义
DISOCC_THRESH = 0.8


def length_sq(x):
    """计算向量的平方长度
    
    Args:
        x: 输入张量，形状为 [batch, height, width, channels]
    
    Returns:
        平方长度张量，形状为 [batch, height, width, 1]
    """
    return torch.sum(torch.square(x), dim=3, keepdim=True)


def image_warp(image, flow):
    """使用光流对图像进行变形
    
    使用双线性插值根据光流场对图像进行后向变形（backward warping）。
    对于每个输出像素位置，根据光流找到对应的输入像素位置并进行插值。
    
    Args:
        image: 输入图像，形状为 [batch, height, width, channels]
               类型: torch.Tensor (float32)
        flow: 光流，形状为 [batch, height, width, 2]
              类型: torch.Tensor (float32)
              flow[:,:,:,0] 是x方向位移，flow[:,:,:,1] 是y方向位移
    
    Returns:
        变形后的图像，形状为 [batch, height, width, channels]
        类型: torch.Tensor (float32)
    """
    batch, height, width, channels = image.shape
    
    # 创建网格坐标 [batch, height, width]
    grid_x = torch.arange(width, dtype=torch.float32).reshape(1, 1, width).repeat(batch, height, 1)
    grid_y = torch.arange(height, dtype=torch.float32).reshape(1, height, 1).repeat(batch, 1, width)
    
    # 根据光流计算采样位置
    flow_x = flow[:, :, :, 0]  # x方向位移
    flow_y = flow[:, :, :, 1]  # y方向位移
    
    # 计算采样坐标
    sample_x = grid_x + flow_x
    sample_y = grid_y + flow_y
    
    # 归一化坐标到[-1, 1]范围（grid_sample要求）
    sample_x_norm = 2.0 * sample_x / (width - 1) - 1.0
    sample_y_norm = 2.0 * sample_y / (height - 1) - 1.0
    
    # 组合采样网格 [batch, height, width, 2]
    sample_grid = torch.stack([sample_x_norm, sample_y_norm], dim=3)
    
    # 转换图像格式为 [batch, channels, height, width]
    image_permuted = image.permute(0, 3, 1, 2)
    
    # 使用双线性插值进行采样
    warped_image = F.grid_sample(
        image_permuted, 
        sample_grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    # 转换回原始格式 [batch, height, width, channels]
    warped_image = warped_image.permute(0, 2, 3, 1)
    
    return warped_image


def forward_warp(flow):
    """前向变形操作
    
    执行前向变形，计算每个像素在光流作用下的新位置，
    并统计每个目标位置被多少个源像素映射到。
    
    Args:
        flow: 光流张量，形状为 [batch, height, width, 2]
              类型: torch.Tensor (float32)
              flow[:,:,:,0] 是x方向位移，flow[:,:,:,1] 是y方向位移
    
    Returns:
        前向变形权重图，形状为 [batch, height, width, 1]
        类型: torch.Tensor (float32)
        值表示每个位置被映射到的次数（权重）
    """
    batch, height, width, _ = flow.shape
    
    # 创建源坐标网格
    grid_x = torch.arange(width, dtype=torch.float32).reshape(1, 1, width).repeat(batch, height, 1)
    grid_y = torch.arange(height, dtype=torch.float32).reshape(1, height, 1).repeat(batch, 1, width)
    
    # 计算目标位置
    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]
    target_x = grid_x + flow_x
    target_y = grid_y + flow_y
    
    # 初始化权重图
    weight_map = torch.zeros(batch, height, width, 1, dtype=torch.float32)
    
    # 对每个batch进行处理
    for b in range(batch):
        # 获取当前batch的目标坐标
        tx = target_x[b].flatten()
        ty = target_y[b].flatten()
        
        # 过滤出在图像边界内的坐标
        valid_mask = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        valid_tx = tx[valid_mask]
        valid_ty = ty[valid_mask]
        
        if len(valid_tx) > 0:
            # 使用双线性插值分配权重
            # 计算四个邻近像素的坐标和权重
            x0 = torch.floor(valid_tx).long()
            y0 = torch.floor(valid_ty).long()
            x1 = torch.clamp(x0 + 1, 0, width - 1)
            y1 = torch.clamp(y0 + 1, 0, height - 1)
            
            # 计算插值权重
            wx = valid_tx - x0.float()
            wy = valid_ty - y0.float()
            
            # 四个角的权重
            w00 = (1 - wx) * (1 - wy)
            w01 = (1 - wx) * wy
            w10 = wx * (1 - wy)
            w11 = wx * wy
            
            # 累加权重到对应位置
            weight_map[b, y0, x0, 0] += w00
            weight_map[b, y1, x0, 0] += w01
            weight_map[b, y0, x1, 0] += w10
            weight_map[b, y1, x1, 0] += w11
    
    return weight_map


def compute_losses(im1, im2, flow_fw, flow_bw,
                   border_mask=None,
                   mask_occlusion='',
                   data_max_distance=1):
    """计算各种损失函数
    
    Args:
        im1: 第一帧图像
        im2: 第二帧图像
        flow_fw: 前向光流
        flow_bw: 后向光流
        border_mask: 边界掩码
        mask_occlusion: 遮挡掩码类型
        data_max_distance: 数据最大距离
    
    Returns:
        包含各种损失的字典
    """
    losses = {}

    # 图像变形
    im2_warped = image_warp(im2, flow_fw)
    im1_warped = image_warp(im1, flow_bw)

    # 图像差异
    im_diff_fw = im1 - im2_warped
    im_diff_bw = im2 - im1_warped

    # 遮挡检测
    disocc_fw = (forward_warp(flow_fw) < DISOCC_THRESH).float()
    disocc_bw = (forward_warp(flow_bw) < DISOCC_THRESH).float()

    # 掩码处理
    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask

    # 光流一致性检查
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped) 
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw = 0.01 * mag_sq_fw + 0.5
    occ_thresh_bw = 0.01 * mag_sq_bw + 0.5
    
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()

    # 遮挡掩码应用
    if mask_occlusion == 'fb':
        mask_fw *= (1 - fb_occ_fw)
        mask_bw *= (1 - fb_occ_bw)
    elif mask_occlusion == 'disocc':
        mask_fw *= (1 - disocc_bw)
        mask_bw *= (1 - disocc_fw)

    occ_fw = 1 - mask_fw
    occ_bw = 1 - mask_bw

    # 计算各种损失
    losses['sym'] = (charbonnier_loss(occ_fw - disocc_bw) + 
                     charbonnier_loss(occ_bw - disocc_fw)) # 如果某个区域在前向光流中被遮挡，那么在后向光流中应该是非遮挡的

    losses['occ'] = (charbonnier_loss(occ_fw) +
                     charbonnier_loss(occ_bw))

    losses['photo'] = (photometric_loss(im_diff_fw, mask_fw) +
                       photometric_loss(im_diff_bw, mask_bw))

    losses['grad'] = (gradient_loss(im1, im2_warped, mask_fw) +
                      gradient_loss(im2, im1_warped, mask_bw))

    losses['smooth_1st'] = (smoothness_loss(flow_fw) +
                            smoothness_loss(flow_bw))

    losses['smooth_2nd'] = (second_order_loss(flow_fw) +
                            second_order_loss(flow_bw))

    losses['fb'] = (charbonnier_loss(flow_diff_fw, mask_fw) +
                    charbonnier_loss(flow_diff_bw, mask_bw))

    losses['ternary'] = (ternary_loss(im1, im2_warped, mask_fw,
                                      max_distance=data_max_distance) +
                         ternary_loss(im2, im1_warped, mask_bw,
                                      max_distance=data_max_distance))

    return losses


def ternary_loss(im1, im2_warped, mask, max_distance=1):
    """计算三元损失
    
    Args:
        im1: 第一帧图像
        im2_warped: 变形后的第二帧图像
        mask: 掩码
        max_distance: 最大距离
    
    Returns:
        三元损失值
    """
    patch_size = 2 * max_distance + 1
    
    def _ternary_transform(image):
        """三元变换"""
        # 转换为灰度图
        intensities = torch.mean(image, dim=3, keepdim=True) * 255
        
        # 创建卷积核
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        # PyTorch卷积核格式: [out_channels, in_channels, height, width]
        weights = torch.tensor(w.transpose(3, 2, 0, 1), dtype=torch.float32)
        
        # 应用卷积
        intensities_permuted = intensities.permute(0, 3, 1, 2)  # [B, C, H, W]
        patches = F.conv2d(intensities_permuted, weights, padding='same')
        patches = patches.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
        return transf_norm

    def _hamming_distance(t1, t2):
        """计算汉明距离"""
        dist = torch.square(t1 - t2)
        dist_norm = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, dim=3, keepdim=True)
        return dist_sum

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)
    dist = _hamming_distance(t1, t2)

    transform_mask = create_mask(mask, [[max_distance, max_distance],
                                        [max_distance, max_distance]])
    return charbonnier_loss(dist, mask * transform_mask)


def occlusion(flow_fw, flow_bw):
    """计算遮挡
    
    Args:
        flow_fw: 前向光流
        flow_bw: 后向光流
    
    Returns:
        前向和后向遮挡掩码
    """
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh = 0.01 * mag_sq + 0.5
    occ_fw = (length_sq(flow_diff_fw) > occ_thresh).float()
    occ_bw = (length_sq(flow_diff_bw) > occ_thresh).float()
    return occ_fw, occ_bw


def divergence(flow):
    """计算光流散度
    
    Args:
        flow: 光流张量
    
    Returns:
        散度张量
    """
    # Sobel滤波器
    filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    filter_y = np.transpose(filter_x)
    
    weight_array_x = np.zeros([1, 1, 3, 3])
    weight_array_x[0, 0, :, :] = filter_x
    weights_x = torch.tensor(weight_array_x, dtype=torch.float32)
    
    weight_array_y = np.zeros([1, 1, 3, 3])
    weight_array_y[0, 0, :, :] = filter_y
    weights_y = torch.tensor(weight_array_y, dtype=torch.float32)
    
    flow_u, flow_v = torch.split(flow, 1, dim=3)
    
    # 转换维度用于卷积
    flow_u_conv = flow_u.permute(0, 3, 1, 2)
    flow_v_conv = flow_v.permute(0, 3, 1, 2)
    
    grad_x = F.conv2d(flow_u_conv, weights_x, padding=1)
    grad_y = F.conv2d(flow_v_conv, weights_y, padding=1)
    
    # 转换回原始维度
    grad_x = grad_x.permute(0, 2, 3, 1)
    grad_y = grad_y.permute(0, 2, 3, 1)
    
    div = torch.sum(torch.cat([grad_x, grad_y], dim=3), dim=3, keepdim=True)
    return div


def norm(x, sigma):
    """高斯衰减函数
    
    Args:
        x: 输入张量
        sigma: 标准差
    
    Returns:
        归一化结果，x=0时为1.0，|x|>sigma时衰减到0
    """
    dist = Normal(0.0, sigma)
    return dist.log_prob(x).exp() / dist.log_prob(torch.tensor(0.0)).exp()


def diffusion_loss(flow, im, occ):
    """扩散损失，基于运动、强度和遮挡标签相似性加权
    受双边流滤波启发
    
    Args:
        flow: 光流
        im: 图像
        occ: 遮挡掩码
    
    Returns:
        扩散损失
    """
    def neighbor_diff(x, num_in=1):
        """计算邻域差异"""
        weights = np.zeros([8 * num_in, num_in, 3, 3])
        out_channel = 0
        for c in range(num_in):  # 遍历输入通道
            for n in [0, 1, 2, 3, 5, 6, 7, 8]:  # 遍历邻域
                weights[out_channel, c, 1, 1] = 1
                weights[out_channel, c, n // 3, n % 3] = -1
                out_channel += 1
        weights = torch.tensor(weights, dtype=torch.float32)
        
        x_conv = x.permute(0, 3, 1, 2)
        result = F.conv2d(x_conv, weights, padding=1)
        return result.permute(0, 2, 3, 1)

    # 创建8通道（每个邻域一个）差异
    occ_diff = neighbor_diff(occ)
    flow_diff = neighbor_diff(flow, 2)
    flow_diff_u, flow_diff_v = torch.split(flow_diff, flow_diff.shape[3]//2, dim=3)
    flow_diff_mag = torch.sqrt(torch.square(flow_diff_u) + torch.square(flow_diff_v))
    
    # 转换为灰度图
    intensity_gray = torch.mean(im, dim=3, keepdim=True)
    intensity_diff = torch.abs(neighbor_diff(intensity_gray))

    diff = norm(intensity_diff, 7.5 / 255) * norm(flow_diff_mag, 0.5) * occ_diff * flow_diff_mag
    return charbonnier_loss(diff)


def photometric_loss(im_diff, mask):
    """光度损失
    
    Args:
        im_diff: 图像差异
        mask: 掩码
    
    Returns:
        光度损失值
    """
    return charbonnier_loss(im_diff, mask, beta=255)


# def conv2d(x, weights):
#     """2D卷积操作
    
#     Args:
#         x: 输入张量
#         weights: 卷积核权重
    
#     Returns:
#         卷积结果
#     """
#     x_conv = x.permute(0, 3, 1, 2)
#     weights_conv = weights.permute(3, 2, 0, 1)
#     result = F.conv2d(x_conv, weights_conv, padding=1)
#     return result.permute(0, 2, 3, 1)


def _smoothness_deltas(flow):
    """计算平滑性增量
    
    Args:
        flow: 光流张量
    
    Returns:
        u方向增量、v方向增量和掩码
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat([mask_x, mask_y], dim=3)

    filter_x = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
    filter_y = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
    weight_array = np.ones([2, 1, 3, 3])
    weight_array[0, 0, :, :] = filter_x
    weight_array[1, 0, :, :] = filter_y
    weights = torch.tensor(weight_array, dtype=torch.float32)

    flow_u, flow_v = torch.split(flow, 1, dim=3)
    
    flow_u_conv = flow_u.permute(0, 3, 1, 2)
    flow_v_conv = flow_v.permute(0, 3, 1, 2)
    
    delta_u = F.conv2d(flow_u_conv, weights, padding=1).permute(0, 2, 3, 1)
    delta_v = F.conv2d(flow_v_conv, weights, padding=1).permute(0, 2, 3, 1)
    
    return delta_u, delta_v, mask


def _gradient_delta(im1, im2_warped):
    """计算梯度增量（梯度差异）
    
    使用Sobel算子计算两幅图像在x和y方向上的梯度，然后计算梯度差异。
    这个函数用于梯度损失计算，通过比较原图像和变形图像的梯度来保持边缘结构。
    
    Args:
        im1: 第一帧图像，形状为 [batch, height, width, 3]
             类型: torch.Tensor (float32)
        im2_warped: 变形后的第二帧图像，形状为 [batch, height, width, 3]
                   类型: torch.Tensor (float32)
    
    Returns:
        梯度差异，形状为 [batch, height, width, 6]
        类型: torch.Tensor (float32)
        通道顺序: [R_x, R_y, G_x, G_y, B_x, B_y]，其中_x表示x方向梯度，_y表示y方向梯度
    """
    # 定义Sobel滤波器用于计算图像梯度
    # x方向Sobel算子：检测垂直边缘（水平方向的强度变化）
    filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # shape: [3, 3]
    # y方向Sobel算子：检测水平边缘（垂直方向的强度变化）
    filter_y = np.transpose(filter_x)  # shape: [3, 3]
    
    # 创建卷积权重数组，为每个颜色通道的x和y方向梯度分别创建滤波器
    # 输出通道数为6：RGB三个通道 × 2个方向（x, y）
    weight_array = np.zeros([6, 3, 3, 3])  # shape: [out_channels=6, in_channels=3, kernel_h=3, kernel_w=3]
    
    # 为每个颜色通道设置x和y方向的Sobel滤波器
    for c in range(3):  # 遍历RGB三个通道
        # 偶数索引：x方向梯度滤波器（通道c的x方向梯度）
        weight_array[2 * c, c, :, :] = filter_x      # 输出通道 2*c 对应输入通道 c 的x方向梯度
        # 奇数索引：y方向梯度滤波器（通道c的y方向梯度）
        weight_array[2 * c + 1, c, :, :] = filter_y  # 输出通道 2*c+1 对应输入通道 c 的y方向梯度
    
    # 转换为PyTorch张量
    weights = torch.tensor(weight_array, dtype=torch.float32)  # shape: [6, 3, 3, 3]

    # 转换图像格式以适配PyTorch卷积：从 [B, H, W, C] 转为 [B, C, H, W]
    im1_conv = im1.permute(0, 3, 1, 2)              # shape: [batch, 3, height, width]
    im2_warped_conv = im2_warped.permute(0, 3, 1, 2)  # shape: [batch, 3, height, width]
    
    # 使用Sobel滤波器计算梯度
    # padding=1 确保输出尺寸与输入相同
    im1_grad = F.conv2d(im1_conv, weights, padding=1).permute(0, 2, 3, 1)
    # im1_grad shape: [batch, height, width, 6] - 6个通道对应 [R_x, R_y, G_x, G_y, B_x, B_y]
    
    im2_warped_grad = F.conv2d(im2_warped_conv, weights, padding=1).permute(0, 2, 3, 1)
    # im2_warped_grad shape: [batch, height, width, 6] - 同样的6个梯度通道
    
    # 计算梯度差异：原图像梯度 - 变形图像梯度
    diff = im1_grad - im2_warped_grad  # shape: [batch, height, width, 6]
    # diff 表示两幅图像在各个方向上的梯度差异，用于衡量边缘结构的保持程度
    
    return diff


def gradient_loss(im1, im2_warped, mask):
    """梯度损失
    
    Args:
        im1: 第一帧图像
        im2_warped: 变形后的第二帧图像
        mask: 掩码
    
    Returns:
        梯度损失值
    """
    mask_x = create_mask(im1, [[0, 0], [1, 1]])
    mask_y = create_mask(im1, [[1, 1], [0, 0]])
    gradient_mask = torch.tile(torch.cat([mask_x, mask_y], dim=3), [1, 1, 1, 3])
    diff = _gradient_delta(im1, im2_warped)
    return charbonnier_loss(diff, mask * gradient_mask)


def smoothness_loss(flow):
    """平滑性损失
    
    Args:
        flow: 光流张量
    
    Returns:
        平滑性损失值
    """
    delta_u, delta_v, mask = _smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, mask)
    loss_v = charbonnier_loss(delta_v, mask)
    return loss_u + loss_v


def _second_order_deltas(flow):
    """计算二阶增量
    
    Args:
        flow: 光流张量
    
    Returns:
        u方向二阶增量、v方向二阶增量和掩码
    """
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat([mask_x, mask_y, mask_diag, mask_diag], dim=3)

    filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
    filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
    filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
    filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
    
    weight_array = np.ones([4, 1, 3, 3])
    weight_array[0, 0, :, :] = filter_x
    weight_array[1, 0, :, :] = filter_y
    weight_array[2, 0, :, :] = filter_diag1
    weight_array[3, 0, :, :] = filter_diag2
    weights = torch.tensor(weight_array, dtype=torch.float32)

    flow_u, flow_v = torch.split(flow, 1, dim=3)
    
    flow_u_conv = flow_u.permute(0, 3, 1, 2)
    flow_v_conv = flow_v.permute(0, 3, 1, 2)
    
    delta_u = F.conv2d(flow_u_conv, weights, padding=1).permute(0, 2, 3, 1)
    delta_v = F.conv2d(flow_v_conv, weights, padding=1).permute(0, 2, 3, 1)
    
    return delta_u, delta_v, mask


def second_order_loss(flow):
    """二阶损失
    
    Args:
        flow: 光流张量
    
    Returns:
        二阶损失值
    """
    delta_u, delta_v, mask = _second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, mask)
    loss_v = charbonnier_loss(delta_v, mask)
    return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """计算广义Charbonnier损失
    
    Args:
        x: 输入张量，形状为 [batch, height, width, channels]
        mask: 掩码张量，形状为 [batch, height, width, mask_channels]
              掩码通道数必须为1或与x的通道数相同，值应为0或1
        truncate: 截断值
        alpha: alpha参数
        beta: beta参数
        epsilon: 小常数防止除零
    
    Returns:
        损失值（标量）
    """
    batch, height, width, channels = x.shape
    normalization = float(batch * height * width * channels)

    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)

    if mask is not None:
        error = error * mask

    if truncate is not None:
        error = torch.minimum(error, torch.tensor(truncate))

    return torch.sum(error) / normalization


def create_mask(tensor, paddings):
    """创建边界掩码
    
    根据指定的填充参数创建一个二进制掩码，用于标识张量中的有效区域。
    掩码在内部区域为1，在边界填充区域为0。
    
    Args:
        tensor: 输入张量，形状为 [batch, height, width, channels]
               类型: torch.Tensor
        paddings: 填充参数，格式为 [[top, bottom], [left, right]]
                 类型: List[List[int]]，例如 [[1, 1], [2, 2]] 表示上下各填充1像素，左右各填充2像素
    
    Returns:
        掩码张量，形状为 [batch, height, width, 1]
        类型: torch.Tensor (float32)
        值: 1.0表示有效区域，0.0表示填充区域
    """
    shape = tensor.shape  # [batch, height, width, channels]
    # 计算内部有效区域的尺寸
    inner_width = shape[1] - (paddings[0][0] + paddings[0][1])   # height方向的有效尺寸
    inner_height = shape[2] - (paddings[1][0] + paddings[1][1])  # width方向的有效尺寸
    
    # 创建内部全1的掩码 [inner_width, inner_height]
    inner = torch.ones([inner_width, inner_height])

    # PyTorch的pad格式为 (left, right, top, bottom)
    pad_values = (paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1])
    # 对内部掩码进行填充，得到完整的2D掩码 [height, width]
    mask2d = F.pad(inner, pad_values, mode='constant', value=0.0)
    # 扩展到3D [batch, height, width]
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    # 扩展到4D [batch, height, width, 1]
    mask4d = mask3d.unsqueeze(3)
    return mask4d.detach()


def create_border_mask(tensor, border_ratio=0.1):
    """创建边界掩码
    
    Args:
        tensor: 输入张量
        border_ratio: 边界比例
    
    Returns:
        边界掩码
    """
    batch, height, width, _ = tensor.shape
    min_dim = float(min(height, width))
    sz = int(np.ceil(min_dim * border_ratio))
    border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()


def create_outgoing_mask(flow):
    """计算界内掩码
    
    在所有光流会将像素保持在图像边界内的位置，掩码值为1；
    在光流会将像素带出图像边界的位置，掩码值为0
    
    Args:
        flow: 光流张量
    
    Returns:
        界内掩码（在边界内为1，出界为0）
    """
    batch, height, width, _ = flow.shape

    # 创建网格坐标
    grid_x = torch.arange(width).reshape(1, 1, width).repeat(batch, height, 1)
    grid_y = torch.arange(height).reshape(1, height, 1).repeat(batch, 1, width)

    flow_u, flow_v = torch.unbind(flow, dim=3)
    pos_x = grid_x.float() + flow_u
    pos_y = grid_y.float() + flow_v
    
    inside_x = torch.logical_and(pos_x <= float(width - 1), pos_x >= 0.0)
    inside_y = torch.logical_and(pos_y <= float(height - 1), pos_y >= 0.0)
    inside = torch.logical_and(inside_x, inside_y)
    
    return inside.float().unsqueeze(3)