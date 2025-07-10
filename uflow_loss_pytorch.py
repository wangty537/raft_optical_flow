import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions import Normal


def resampler(data, warp, name='resampler'):
    """在用户定义的坐标处重采样输入数据
    
    Args:
        data: 形状为 [batch_size, data_height, data_width, data_num_channels] 的张量，
              包含将被重采样的2D数据
        warp: 形状为 [batch_size, dim_0, ..., dim_n, 2] 的张量，
              包含执行重采样的坐标
        name: 操作的可选名称
    
    Returns:
        从 data 重采样的值的张量。输出张量形状为
        [batch_size, dim_0, ..., dim_n, data_num_channels]
    """
    warp_x, warp_y = torch.unbind(warp, dim=-1)
    return resampler_with_unstacked_warp(data, warp_x, warp_y, name=name)


def resampler_with_unstacked_warp(data, warp_x, warp_y, safe=True, name='resampler'):
    """在用户定义的坐标处重采样输入数据
    
    Args:
        data: 形状为 [batch_size, data_height, data_width, data_num_channels] 的张量
        warp_x: 形状为 [batch_size, dim_0, ..., dim_n] 的张量，包含x坐标
        warp_y: 与warp_x相同形状的张量，包含y坐标
        safe: 布尔值，如果为True，warp_x和warp_y将被限制在边界内
        name: 操作的可选名称
    
    Returns:
        从 data 重采样的值的张量
    """
    if not warp_x.shape == warp_y.shape:
        raise ValueError(f'warp_x and warp_y are of incompatible shapes: {warp_x.shape} vs {warp_y.shape}')
    
    warp_shape = warp_x.shape
    if warp_x.shape[0] != data.shape[0]:
        raise ValueError(f'warp_x and data must have compatible first dimension (batch size), '
                        f'but their shapes are {warp_x.shape[0]} and {data.shape[0]}')
    
    # 计算最接近warp的四个整数点
    warp_floor_x = torch.floor(warp_x)
    warp_floor_y = torch.floor(warp_y)
    
    # 计算每个点的权重
    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y
    
    warp_floor_x = warp_floor_x.long()
    warp_floor_y = warp_floor_y.long()
    warp_ceil_x = torch.ceil(warp_x).long()
    warp_ceil_y = torch.ceil(warp_y).long()
    
    left_warp_weight = 1.0 - right_warp_weight
    up_warp_weight = 1.0 - down_warp_weight
    
    # 扩展warp以包含batch索引
    warp_batch_shape = [warp_shape[0]] + [1] * (len(warp_shape) - 1)
    warp_batch = torch.arange(warp_shape[0], device=data.device).view(warp_batch_shape)
    warp_batch = warp_batch.expand_as(warp_y)
    
    left_warp_weight = left_warp_weight.unsqueeze(-1)
    down_warp_weight = down_warp_weight.unsqueeze(-1)
    up_warp_weight = up_warp_weight.unsqueeze(-1)
    right_warp_weight = right_warp_weight.unsqueeze(-1)
    
    up_left_warp = torch.stack([warp_batch, warp_floor_y, warp_floor_x], dim=-1)
    up_right_warp = torch.stack([warp_batch, warp_floor_y, warp_ceil_x], dim=-1)
    down_left_warp = torch.stack([warp_batch, warp_ceil_y, warp_floor_x], dim=-1)
    down_right_warp = torch.stack([warp_batch, warp_ceil_y, warp_ceil_x], dim=-1)
    
    def gather_nd(params, indices):
        """PyTorch版本的gather_nd"""
        if safe:
            # 安全版本：检查边界
            batch_size, height, width, channels = params.shape
            valid_mask = (
                (indices[..., 1] >= 0) & (indices[..., 1] < height) &
                (indices[..., 2] >= 0) & (indices[..., 2] < width)
            )
            # 将无效索引设置为0
            safe_indices = indices.clone()
            safe_indices[..., 1] = torch.clamp(indices[..., 1], 0, height - 1)
            safe_indices[..., 2] = torch.clamp(indices[..., 2], 0, width - 1)
            
            result = params[safe_indices[..., 0], safe_indices[..., 1], safe_indices[..., 2]]
            result = result * valid_mask.unsqueeze(-1).float()
            return result
        else:
            return params[indices[..., 0], indices[..., 1], indices[..., 2]]
    
    # 收集数据然后取加权平均得到重采样结果
    result = (
        (gather_nd(data, up_left_warp) * left_warp_weight +
         gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
        (gather_nd(data, down_left_warp) * left_warp_weight +
         gather_nd(data, down_right_warp) * right_warp_weight) * down_warp_weight
    )
    
    return result


def flow_to_warp(flow):
    """从光流场计算变形
    
    Args:
        flow: 表示光流的张量
    
    Returns:
        变形，即估计光流的端点
    """
    height, width = flow.shape[-3:-1]
    device = flow.device
    
    # 构造图像坐标网格
    i_grid = torch.linspace(0.0, height - 1.0, height, device=device)
    j_grid = torch.linspace(0.0, width - 1.0, width, device=device)
    i_grid, j_grid = torch.meshgrid(i_grid, j_grid, indexing='ij')
    grid = torch.stack([i_grid, j_grid], dim=2)
    
    # 如果需要，添加batch维度以匹配flow的形状
    if len(flow.shape) == 4:
        grid = grid.unsqueeze(0)
    
    # 将光流场添加到图像网格
    if flow.dtype != grid.dtype:
        grid = grid.to(flow.dtype)
    warp = grid + flow
    return warp


def mask_invalid(coords):
    """掩盖图像外的坐标
    
    有效 = 1，无效 = 0
    
    Args:
        coords: 4D浮点张量的图像坐标
    
    Returns:
        显示哪些坐标有效的掩码
    """
    coords_rank = len(coords.shape)
    if coords_rank != 4:
        raise NotImplementedError()
    
    max_height = float(coords.shape[-3] - 1)
    max_width = float(coords.shape[-2] - 1)
    
    mask = torch.logical_and(
        torch.logical_and(coords[:, :, :, 0] >= 0.0, coords[:, :, :, 0] <= max_height),
        torch.logical_and(coords[:, :, :, 1] >= 0.0, coords[:, :, :, 1] <= max_width)
    )
    mask = mask.float().unsqueeze(-1)
    return mask


def resample(source, coords):
    """在传递的坐标处重采样源图像
    
    Args:
        source: 要重采样的图像批次
        coords: 图像中的坐标批次
    
    Returns:
        重采样的图像
    """
    orig_source_dtype = source.dtype
    if source.dtype != torch.float32:
        source = source.float()
    if coords.dtype != torch.float32:
        coords = coords.float()
    
    coords_rank = len(coords.shape)
    if coords_rank == 4:
        # 翻转坐标顺序以匹配resampler的期望
        output = resampler(source, coords[:, :, :, [1, 0]])
        if orig_source_dtype != source.dtype:
            return output.to(orig_source_dtype)
        return output
    else:
        raise NotImplementedError()


def compute_range_map(flow, downsampling_factor=1, reduce_downsampling_bias=True, resize_output=True):
    """计算每个坐标被采样的频率
    
    Args:
        flow: 形状为 (batch_size, height, width, 2) 的浮点张量，表示密集光流场
        downsampling_factor: 整数，相对于输入分辨率的输出分辨率下采样因子
        reduce_downsampling_bias: 布尔值，是否通过填充光流场来减少图像边界附近的下采样偏差
        resize_output: 布尔值，是否将输出调整到输入分辨率
    
    Returns:
        形状为 [batch_size, height, width, 1] 的浮点张量，表示每个像素被采样的频率
    """
    input_shape = list(flow.shape)
    if len(input_shape) != 4:
        raise NotImplementedError()
    batch_size, input_height, input_width, _ = input_shape
    
    flow_height = input_height
    flow_width = input_width
    
    # 应用下采样
    output_height = input_height // downsampling_factor
    output_width = input_width // downsampling_factor
    
    if downsampling_factor > 1:
        if reduce_downsampling_bias:
            p = downsampling_factor // 2
            flow_height += 2 * p
            flow_width += 2 * p
            # 应用对称填充
            for _ in range(p):
                flow = F.pad(flow.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='reflect').permute(0, 2, 3, 1)
            coords = flow_to_warp(flow) - p
        # 更新坐标框架到下采样的坐标框架
        coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
    elif downsampling_factor == 1:
        coords = flow_to_warp(flow)
    else:
        raise ValueError('downsampling_factor must be an integer >= 1.')
    
    # 将坐标分割为整数部分和浮点偏移量用于插值
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.long()
    
    # 定义批次偏移量
    batch_range = torch.arange(batch_size, device=flow.device).view(batch_size, 1, 1)
    idx_batch_offset = batch_range.expand(batch_size, flow_height, flow_width) * output_height * output_width
    
    # 展平所有内容
    coords_floor_flattened = coords_floor.view(-1, 2)
    coords_offset_flattened = coords_offset.view(-1, 2)
    idx_batch_offset_flattened = idx_batch_offset.view(-1)
    
    # 初始化结果
    idxs_list = []
    weights_list = []
    
    # 循环遍历四个相邻像素的差异di和dj
    for di in range(2):
        for dj in range(2):
            # 计算相邻像素坐标
            idxs_i = coords_floor_flattened[:, 0] + di
            idxs_j = coords_floor_flattened[:, 1] + dj
            # 计算所有像素的平坦索引
            idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j
            
            # 只计算有效像素
            mask = torch.logical_and(
                torch.logical_and(idxs_i >= 0, idxs_i < output_height),
                torch.logical_and(idxs_j >= 0, idxs_j < output_width)
            )
            valid_indices = torch.where(mask)[0]
            valid_idxs = idxs[valid_indices]
            valid_offsets = coords_offset_flattened[valid_indices]
            
            # 根据双线性插值计算权重
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
            weights = weights_i * weights_j
            
            # 将索引和权重添加到相应列表
            idxs_list.append(valid_idxs)
            weights_list.append(weights)
    
    # 连接所有内容
    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    
    # 为每个像素求和权重并重塑结果
    counts = torch.zeros(batch_size * output_height * output_width, device=flow.device)
    counts.scatter_add_(0, idxs, weights)
    count_image = counts.view(batch_size, output_height, output_width, 1)
    
    if downsampling_factor > 1:
        # 归一化计数图像，使下采样不影响计数
        count_image /= downsampling_factor**2
        if resize_output:
            count_image = resize(count_image, input_height, input_width, is_flow=False)
    
    return count_image


def compute_warps_and_occlusion(flows, occlusion_estimation, occ_weights=None, 
                                occ_thresholds=None, occ_clip_max=None, 
                                occlusions_are_zeros=True, occ_active=None):
    """计算变形、有效变形掩码、平流图和遮挡掩码"""
    if occ_clip_max is not None:
        for key in occ_clip_max:
            if key not in ['forward_collision', 'fb_abs']:
                raise ValueError('occ_clip_max for this key is not supported')
    
    warps = dict()
    range_maps_high_res = dict()
    range_maps_low_res = dict()
    occlusion_logits = dict()
    occlusion_scores = dict()
    occlusion_masks = dict()
    valid_warp_masks = dict()
    fb_sq_diff = dict()
    fb_sum_sq = dict()
    
    for key in flows:
        i, j, t = key
        rev_key = (j, i, t)
        
        warps[key] = []
        range_maps_high_res[key] = []
        range_maps_low_res[rev_key] = []
        occlusion_masks[key] = []
        valid_warp_masks[key] = []
        fb_sq_diff[key] = []
        fb_sum_sq[key] = []
        
        for level in range(min(3, len(flows[key]))):
            flow_ij = flows[key][level]
            flow_ji = flows[rev_key][level]
            
            # 计算变形（坐标）和有效坐标掩码
            warps[key].append(flow_to_warp(flow_ij))
            valid_warp_masks[key].append(mask_invalid(warps[key][level]))
            
            # 比较前向和后向光流
            flow_ji_in_i = resample(flow_ji, warps[key][level])
            fb_sq_diff[key].append(
                torch.sum((flow_ij + flow_ji_in_i)**2, dim=-1, keepdim=True))
            fb_sum_sq[key].append(
                torch.sum(flow_ij**2 + flow_ji_in_i**2, dim=-1, keepdim=True))
            
            if level != 0:
                continue
            
            # 初始化遮挡掩码
            occlusion_mask = torch.zeros_like(flow_ij[..., :1])
            occlusion_scores['forward_collision'] = torch.zeros_like(flow_ij[..., :1])
            occlusion_scores['backward_zero'] = torch.zeros_like(flow_ij[..., :1])
            occlusion_scores['fb_abs'] = torch.zeros_like(flow_ij[..., :1])
            
            if occlusion_estimation == 'none' or (
                occ_active is not None and not occ_active[occlusion_estimation]):
                occlusion_mask = torch.zeros_like(flow_ij[..., :1])
            
            elif occlusion_estimation == 'brox':
                occlusion_mask = (fb_sq_diff[key][level] > 
                                0.01 * fb_sum_sq[key][level] + 0.5).float()
            
            elif occlusion_estimation == 'fb_abs':
                occlusion_mask = (fb_sq_diff[key][level]**0.5 > 1.5).float()
            
            elif occlusion_estimation == 'wang':
                range_maps_low_res[rev_key].append(
                    compute_range_map(flow_ji, downsampling_factor=1,
                                    reduce_downsampling_bias=False, resize_output=False))
                occlusion_mask = 1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.)
            
            elif occlusion_estimation == 'wang4':
                range_maps_low_res[rev_key].append(
                    compute_range_map(flow_ji, downsampling_factor=4,
                                    reduce_downsampling_bias=True, resize_output=True))
                occlusion_mask = 1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.)
            
            elif occlusion_estimation == 'wangthres':
                range_maps_low_res[rev_key].append(
                    compute_range_map(flow_ji, downsampling_factor=1,
                                    reduce_downsampling_bias=True, resize_output=True))
                occlusion_mask = (range_maps_low_res[rev_key][level] < 0.75).float()
            
            elif occlusion_estimation == 'wang4thres':
                range_maps_low_res[rev_key].append(
                    compute_range_map(flow_ji, downsampling_factor=4,
                                    reduce_downsampling_bias=True, resize_output=True))
                occlusion_mask = (range_maps_low_res[rev_key][level] < 0.75).float()
            
            elif occlusion_estimation == 'uflow':
                # 从前向光流的范围图计算遮挡
                if 'forward_collision' in occ_weights and (
                    occ_active is None or occ_active['forward_collision']):
                    range_maps_high_res[key].append(
                        compute_range_map(flow_ij, downsampling_factor=1,
                                        reduce_downsampling_bias=True, resize_output=True))
                    fwd_range_map_in_i = resample(range_maps_high_res[key][level], warps[key][level])
                    occlusion_scores['forward_collision'] = torch.clamp(
                        fwd_range_map_in_i, 1., occ_clip_max['forward_collision']) - 1.0
                
                # 从后向光流的范围图计算遮挡
                if 'backward_zero' in occ_weights and (
                    occ_active is None or occ_active['backward_zero']):
                    range_maps_low_res[rev_key].append(
                        compute_range_map(flow_ji, downsampling_factor=4,
                                        reduce_downsampling_bias=True, resize_output=True))
                    occlusion_scores['backward_zero'] = (
                        1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))
                
                # 从前向-后向一致性计算遮挡
                if 'fb_abs' in occ_weights and (
                    occ_active is None or occ_active['fb_abs']):
                    occlusion_scores['fb_abs'] = torch.clamp(
                        fb_sq_diff[key][level]**0.5, 0.0, occ_clip_max['fb_abs'])
                
                occlusion_logits = torch.zeros_like(flow_ij[..., :1])
                for k, v in occlusion_scores.items():
                    occlusion_logits += (v - occ_thresholds[k]) * occ_weights[k]
                occlusion_mask = torch.sigmoid(occlusion_logits)
            
            else:
                raise ValueError('Unknown value for occlusion_estimation:', occlusion_estimation)
            
            occlusion_masks[key].append(
                1. - occlusion_mask if occlusions_are_zeros else occlusion_mask)
    
    return warps, valid_warp_masks, range_maps_low_res, occlusion_masks, fb_sq_diff, fb_sum_sq


def apply_warps_stop_grad(sources, warps, level):
    """在正确的源上应用所有变形"""
    warped = dict()
    for (i, j, t) in warps:
        # 只通过变形传播梯度，不通过源传播
        warped[(i, j, t)] = resample(sources[j].detach(), warps[(i, j, t)][level])
    return warped


def upsample(img, is_flow):
    """将图像或光流场的分辨率加倍
    
    Args:
        img: 要调整大小的图像或光流场
        is_flow: 布尔值，是否相应缩放光流
    
    Returns:
        调整大小并可能缩放的图像或光流场
    """
    _, height, width, _ = img.shape
    orig_dtype = img.dtype
    if orig_dtype != torch.float32:
        img = img.float()
    
    # 转换为NCHW格式进行插值
    img_nchw = img.permute(0, 3, 1, 2)
    img_resized = F.interpolate(img_nchw, size=(int(height * 2), int(width * 2)), 
                               mode='bilinear', align_corners=False)
    img_resized = img_resized.permute(0, 2, 3, 1)
    
    if is_flow:
        # 缩放光流值以与新图像大小一致
        img_resized *= 2
    
    if img_resized.dtype != orig_dtype:
        return img_resized.to(orig_dtype)
    return img_resized


def downsample(img, is_flow):
    """将图像或光流场的分辨率减半
    
    Args:
        img: 要调整大小的图像或光流场
        is_flow: 布尔值，是否相应缩放光流
    
    Returns:
        调整大小并可能缩放的图像或光流场
    """
    _, height, width, _ = img.shape
    
    # 转换为NCHW格式进行插值
    img_nchw = img.permute(0, 3, 1, 2)
    img_resized = F.interpolate(img_nchw, size=(int(height / 2), int(width / 2)), 
                               mode='bilinear', align_corners=False)
    img_resized = img_resized.permute(0, 2, 3, 1)
    
    if is_flow:
        # 缩放光流值以与新图像大小一致
        img_resized /= 2
    return img_resized


def resize(img, height, width, is_flow, mask=None):
    """将图像或光流场调整到新分辨率
    
    Args:
        img: 要调整大小的图像或光流场，形状为 [b, h, w, c]
        height: 新分辨率的高度
        width: 新分辨率的宽度
        is_flow: 布尔值，是否相应缩放光流
        mask: 可选的掩码，每像素{0,1}标志
    
    Returns:
        调整大小并可能缩放的图像或光流场（和掩码）
    """
    def _resize(img, mask=None):
        orig_height, orig_width = img.shape[1:3]
        
        if orig_height == height and orig_width == width:
            # 如果不需要调整大小，提前返回
            if mask is not None:
                return img, mask
            else:
                return img
        
        if mask is not None:
            # 与掩码相乘，确保无效位置为零
            img = img * mask
            # 调整图像大小
            img_nchw = img.permute(0, 3, 1, 2)
            img_resized = F.interpolate(img_nchw, size=(height, width), 
                                       mode='bilinear', align_corners=False)
            img_resized = img_resized.permute(0, 2, 3, 1)
            
            # 调整掩码大小（将作为归一化权重）
            mask_nchw = mask.permute(0, 3, 1, 2)
            mask_resized = F.interpolate(mask_nchw, size=(height, width), 
                                        mode='bilinear', align_corners=False)
            mask_resized = mask_resized.permute(0, 2, 3, 1)
            
            # 归一化稀疏光流场和掩码
            img_resized = img_resized / (mask_resized + 1e-8)
            mask_resized = (mask_resized > 0).float()
        else:
            # 正常调整大小
            img_nchw = img.permute(0, 3, 1, 2)
            img_resized = F.interpolate(img_nchw, size=(height, width), 
                                       mode='bilinear', align_corners=False)
            img_resized = img_resized.permute(0, 2, 3, 1)
        
        if is_flow:
            # 如果图像是光流图像，缩放光流值以与新图像大小一致
            scaling = torch.tensor([float(height) / orig_height, 
                                   float(width) / orig_width], 
                                  device=img.device, dtype=img.dtype)
            scaling = scaling.view(1, 1, 1, 2)
            img_resized *= scaling
        
        if mask is not None:
            return img_resized, mask_resized
        return img_resized
    
    # 在正确的形状下应用调整大小
    shape = list(img.shape)
    if len(shape) == 3:
        if mask is not None:
            img_resized, mask_resized = _resize(img.unsqueeze(0), mask.unsqueeze(0))
            return img_resized.squeeze(0), mask_resized.squeeze(0)
        else:
            return _resize(img.unsqueeze(0)).squeeze(0)
    elif len(shape) == 4:
        # 输入形状正确
        return _resize(img, mask)
    elif len(shape) > 4:
        # 将输入重塑为[b, h, w, c]，调整大小并重塑回来
        img_flattened = img.view(-1, *shape[-3:])
        if mask is not None:
            mask_flattened = mask.view(-1, *shape[-3:])
            img_resized, mask_resized = _resize(img_flattened, mask_flattened)
        else:
            img_resized = _resize(img_flattened)
        
        result_img = img_resized.view(*shape[:-3], *img_resized.shape[-3:])
        if mask is not None:
            result_mask = mask_resized.view(*shape[:-3], *mask_resized.shape[-3:])
            return result_img, result_mask
        return result_img
    else:
        raise ValueError('Cannot resize an image of shape', shape)


def random_subseq(sequence, subseq_len):
    """选择给定长度的随机子序列"""
    seq_len = sequence.shape[0]
    start_index = torch.randint(0, seq_len - subseq_len + 1, (1,)).item()
    subseq = sequence[start_index:start_index + subseq_len]
    return subseq


def normalize_for_feature_metric_loss(features):
    """为特征度量损失归一化特征"""
    normalized_features = dict()
    for key, feature_map in features.items():
        # 归一化特征通道以具有相同的绝对激活
        norm_feature_map = feature_map / (
            torch.sum(torch.abs(feature_map), dim=[0, 1, 2], keepdim=True) + 1e-16)
        # 归一化每个像素特征在所有通道上的平均值为1
        norm_feature_map /= (
            torch.sum(torch.abs(norm_feature_map), dim=-1, keepdim=True) + 1e-16)
        normalized_features[key] = norm_feature_map
    return normalized_features


def l1(x):
    """L1损失"""
    return torch.abs(x + 1e-6)


def robust_l1(x):
    """鲁棒L1度量"""
    return (x**2 + 0.001**2)**0.5


def abs_robust_loss(diff, eps=0.01, q=0.4):
    """DDFlow使用的所谓鲁棒损失"""
    return torch.pow(torch.abs(diff) + eps, q)


def image_grads(image_batch, stride=1):
    """计算图像梯度"""
    image_batch_gh = image_batch[:, stride:] - image_batch[:, :-stride]
    image_batch_gw = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
    return image_batch_gh, image_batch_gw


def image_averages(image_batch):
    """计算图像平均值"""
    image_batch_ah = (image_batch[:, 1:] + image_batch[:, :-1]) / 2.
    image_batch_aw = (image_batch[:, :, 1:] + image_batch[:, :, :-1]) / 2
    return image_batch_ah, image_batch_aw


def get_distance_metric_fns(distance_metrics):
    """返回距离度量字典"""
    output = {}
    for key, distance_metric in distance_metrics.items():
        if distance_metric == 'l1':
            output[key] = l1
        elif distance_metric == 'robust_l1':
            output[key] = robust_l1
        elif distance_metric == 'ddflow':
            output[key] = abs_robust_loss
        else:
            raise ValueError('Unknown loss function')
    return output


def compute_loss(weights, images, flows, warps, valid_warp_masks, not_occluded_masks,
                fb_sq_diff, fb_sum_sq, warped_images, only_forward=False,
                selfsup_transform_fns=None, fb_sigma_teacher=0.003, fb_sigma_student=0.03,
                plot_dir=None, distance_metrics=None, smoothness_edge_weighting='gaussian',
                stop_gradient_mask=True, selfsup_mask='gaussian', ground_truth_occlusions=None,
                smoothness_at_level=2):
    """计算UFlow损失"""
    if distance_metrics is None:
        distance_metrics = {
            'photo': 'robust_l1',
            'census': 'ddflow',
        }
    
    distance_metric_fns = get_distance_metric_fns(distance_metrics)
    losses = dict()
    for key in weights:
        if key not in ['edge_constant']:
            losses[key] = 0.0
    
    compute_loss_for_these_flows = ['augmented-student']
    # 计算非自监督对的数量，我们将对其应用损失
    num_pairs = sum([1.0 for (i, j, c) in warps if c in compute_loss_for_these_flows])
    
    # 遍历图像对
    for key in warps:
        i, j, c = key
        
        if c not in compute_loss_for_these_flows or (only_forward and i > j):
            continue
        
        if ground_truth_occlusions is None:
            if stop_gradient_mask:
                mask_level0 = (not_occluded_masks[key][0] * valid_warp_masks[key][0]).detach()
            else:
                mask_level0 = not_occluded_masks[key][0] * valid_warp_masks[key][0]
        else:
            # 使用真实掩码
            if i > j:
                continue
            ground_truth_occlusions = 1.0 - ground_truth_occlusions.float()
            mask_level0 = (ground_truth_occlusions * valid_warp_masks[key][0]).detach()
        
        if 'photo' in weights:
            error = distance_metric_fns['photo'](images[i] - warped_images[key])
            losses['photo'] += (
                weights['photo'] * torch.sum(mask_level0 * error) /
                (torch.sum(mask_level0) + 1e-16) / num_pairs)
        
        if 'smooth2' in weights or 'smooth1' in weights:
            edge_constant = 0.0
            if 'edge_constant' in weights:
                edge_constant = weights['edge_constant']
            
            abs_fn = None
            if smoothness_edge_weighting == 'gaussian':
                abs_fn = lambda x: x**2
            elif smoothness_edge_weighting == 'exponential':
                abs_fn = torch.abs
            
            # 计算图像梯度
            images_level0 = images[i]
            height, width = images_level0.shape[-3:-1]
            # 调整两次以获得更平滑的结果
            images_level1 = resize(images_level0, height // 2, width // 2, is_flow=False)
            images_level2 = resize(images_level1, height // 4, width // 4, is_flow=False)
            images_at_level = [images_level0, images_level1, images_level2]
            
            if 'smooth1' in weights:
                img_gx, img_gy = image_grads(images_at_level[smoothness_at_level])
                weights_x = torch.exp(-torch.mean(
                    abs_fn(edge_constant * img_gx), dim=-1, keepdim=True))
                weights_y = torch.exp(-torch.mean(
                    abs_fn(edge_constant * img_gy), dim=-1, keepdim=True))
                
                # 计算预测平滑度的二阶导数
                flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])
                
                # 计算加权平滑度
                losses['smooth1'] += (
                    weights['smooth1'] *
                    (torch.mean(weights_x * robust_l1(flow_gx)) +
                     torch.mean(weights_y * robust_l1(flow_gy))) / 2. / num_pairs)
            
            if 'smooth2' in weights:
                img_gx, img_gy = image_grads(images_at_level[smoothness_at_level], stride=2)
                weights_xx = torch.exp(-torch.mean(
                    abs_fn(edge_constant * img_gx), dim=-1, keepdim=True))
                weights_yy = torch.exp(-torch.mean(
                    abs_fn(edge_constant * img_gy), dim=-1, keepdim=True))
                
                # 计算预测平滑度的二阶导数
                flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])
                flow_gxx, _ = image_grads(flow_gx)
                _, flow_gyy = image_grads(flow_gy)
                
                # 计算加权平滑度
                losses['smooth2'] += (
                    weights['smooth2'] *
                    (torch.mean(weights_xx * robust_l1(flow_gxx)) +
                     torch.mean(weights_yy * robust_l1(flow_gyy))) / 2. / num_pairs)
        
        if 'ssim' in weights:
            ssim_error, avg_weight = weighted_ssim(
                warped_images[key], images[i], mask_level0.squeeze(-1))
            losses['ssim'] += weights['ssim'] * (
                torch.sum(ssim_error * avg_weight) /
                (torch.sum(avg_weight) + 1e-16) / num_pairs)
        
        if 'census' in weights:
            losses['census'] += weights['census'] * census_loss(
                images[i], warped_images[key], mask_level0,
                distance_metric_fn=distance_metric_fns['census']) / num_pairs
        
        if 'selfsup' in weights:
            assert selfsup_transform_fns is not None
            _, h, w, _ = flows[key][2].shape
            teacher_flow = flows[(i, j, 'original-teacher')][2]
            student_flow = flows[(i, j, 'transformed-student')][2]
            teacher_flow = selfsup_transform_fns[2](teacher_flow, i_or_ij=(i, j), is_flow=True)
            
            if selfsup_mask == 'gaussian':
                student_fb_consistency = torch.exp(
                    -fb_sq_diff[(i, j, 'transformed-student')][2] /
                    (fb_sigma_student**2 * (h**2 + w**2)))
                teacher_fb_consistency = torch.exp(
                    -fb_sq_diff[(i, j, 'original-teacher')][2] /
                    (fb_sigma_teacher**2 * (h**2 + w**2)))
            elif selfsup_mask == 'advection':
                student_fb_consistency = not_occluded_masks[(i, j, 'transformed-student')][2]
                teacher_fb_consistency = not_occluded_masks[(i, j, 'original-teacher')][2]
            elif selfsup_mask == 'ddflow':
                threshold_student = 0.01 * fb_sum_sq[(i, j, 'transformed-student')][2] + 0.5
                threshold_teacher = 0.01 * fb_sum_sq[(i, j, 'original-teacher')][2] + 0.5
                student_fb_consistency = (
                    fb_sq_diff[(i, j, 'transformed-student')][2] < threshold_student).float()
                teacher_fb_consistency = (
                    fb_sq_diff[(i, j, 'original-teacher')][2] < threshold_teacher).float()
            else:
                raise ValueError('Unknown selfsup_mask', selfsup_mask)
            
            student_mask = 1. - (
                student_fb_consistency * valid_warp_masks[(i, j, 'transformed-student')][2])
            teacher_mask = (
                teacher_fb_consistency * valid_warp_masks[(i, j, 'original-teacher')][2])
            teacher_mask = selfsup_transform_fns[2](teacher_mask, i_or_ij=(i, j), is_flow=False)
            error = robust_l1(teacher_flow.detach() - student_flow)
            mask = (teacher_mask * student_mask).detach()
            losses['selfsup'] += (
                weights['selfsup'] * torch.sum(mask * error) /
                (torch.sum(torch.ones_like(mask)) + 1e-16) / num_pairs)
    
    losses['total'] = sum(losses.values())
    return losses


def supervised_loss(weights, ground_truth_flow, ground_truth_valid, predicted_flows):
    """当提供真实光流时返回监督l1损失"""
    losses = {}
    # 真实光流从图像0到图像1
    predicted_flow = predicted_flows[(0, 1, 'augmented')][0]
    # 调整光流大小以匹配真实值
    _, height, width, _ = ground_truth_flow.shape
    predicted_flow = resize(predicted_flow, height, width, is_flow=True)
    # 计算误差/损失度量
    error = robust_l1(ground_truth_flow - predicted_flow)
    if ground_truth_valid is None:
        b, h, w, _ = ground_truth_flow.shape
        ground_truth_valid = torch.ones((b, h, w, 1), dtype=torch.float32, device=ground_truth_flow.device)
    losses['supervision'] = (
        weights['supervision'] *
        torch.sum(ground_truth_valid * error) /
        (torch.sum(ground_truth_valid) + 1e-16))
    losses['total'] = losses['supervision']
    return losses


def random_crop(batch, max_offset_height=32, max_offset_width=32):
    """随机裁剪一批图像
    
    Args:
        batch: 形状为 [batch_size, height, width, num_channels] 的4D张量
        max_offset_height: 裁剪结果左上角的最大垂直坐标
        max_offset_width: 裁剪结果左上角的最大水平坐标
    
    Returns:
        1) 裁剪图像的张量，形状为 [batch_size, height-max_offset, width-max_offset, num_channels]
        2) 形状为 [batch_size, 2] 的偏移张量，用于高度和宽度偏移
    """
    # 计算当前形状和裁剪的目标形状
    batch_size, height, width, num_channels = batch.shape
    target_height = height - max_offset_height
    target_width = width - max_offset_width
    
    # 随机采样偏移量
    offsets_height = torch.randint(0, max_offset_height + 1, (batch_size,))
    offsets_width = torch.randint(0, max_offset_width + 1, (batch_size,))
    offsets = torch.stack([offsets_height, offsets_width], dim=-1)
    
    # 循环遍历批次并执行裁剪
    cropped_images = []
    for image, offset_height, offset_width in zip(batch, offsets_height, offsets_width):
        cropped_images.append(
            image[offset_height:offset_height + target_height,
                  offset_width:offset_width + target_width, :])
    cropped_batch = torch.stack(cropped_images)
    
    return cropped_batch, offsets


def random_shift(batch, max_shift_height=32, max_shift_width=32):
    """随机移位一批图像（带环绕）
    
    Args:
        batch: 形状为 [batch_size, height, width, num_channels] 的4D张量
        max_shift_height: 沿高度维度任一方向的最大移位
        max_shift_width: 沿宽度维度任一方向的最大移位
    
    Returns:
        1) 移位图像的张量，形状为 [batch_size, height, width, num_channels]
        2) 形状为 [batch_size, 2] 的随机移位，正数表示图像向下/右移位，负数表示向上/左移位
    """
    # 随机采样图像移位量
    batch_size, _, _, _ = batch.shape
    shifts_height = torch.randint(-max_shift_height, max_shift_height + 1, (batch_size,))
    shifts_width = torch.randint(-max_shift_width, max_shift_width + 1, (batch_size,))
    shifts = torch.stack([shifts_height, shifts_width], dim=-1)
    
    # 循环遍历批次并移位图像
    shifted_images = []
    for image, shift_height, shift_width in zip(batch, shifts_height, shifts_width):
        shifted_images.append(
            torch.roll(image, shifts=(shift_height.item(), shift_width.item()), dims=(0, 1)))
    shifted_images = torch.stack(shifted_images)
    
    return shifted_images, shifts


def zero_mask_border(mask_bhw3, patch_size):
    """用于忽略census_transform的边界效应"""
    mask_padding = patch_size // 2
    mask = mask_bhw3[:, mask_padding:-mask_padding, mask_padding:-mask_padding, :]
    return F.pad(mask.permute(0, 3, 1, 2), 
                (mask_padding, mask_padding, mask_padding, mask_padding)).permute(0, 2, 3, 1)


def census_transform(image, patch_size):
    """DDFlow描述的census变换
    
    Args:
        image: 形状为 (b, h, w, c) 的张量
        patch_size: 整数
    
    Returns:
        应用census变换的图像
    """
    intensities = torch.mean(image, dim=-1, keepdim=True) * 255
    kernel = torch.eye(patch_size * patch_size, device=image.device).view(
        patch_size, patch_size, 1, patch_size * patch_size)
    
    # 转换为NCHW格式进行卷积
    intensities_nchw = intensities.permute(0, 3, 1, 2)
    kernel_nchw = kernel.permute(3, 2, 0, 1)
    
    neighbors = F.conv2d(intensities_nchw, kernel_nchw, padding='same')
    neighbors = neighbors.permute(0, 2, 3, 1)
    
    diff = neighbors - intensities
    # 采用DDFlow的系数
    diff_norm = diff / torch.sqrt(0.81 + torch.square(diff))
    return diff_norm


def soft_hamming(a_bhwk, b_bhwk, thresh=0.1):
    """张量a_bhwk和张量b_bhwk之间的软汉明距离
    
    Args:
        a_bhwk: 形状为 (batch, height, width, features) 的张量
        b_bhwk: 形状为 (batch, height, width, features) 的张量
        thresh: 浮点阈值
    
    Returns:
        在(h, w)位置上显著比thresh更不同的位置约为1，显著比thresh更相似的位置约为0的张量
    """
    sq_dist_bhwk = torch.square(a_bhwk - b_bhwk)
    soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
    return torch.sum(soft_thresh_dist_bhwk, dim=3, keepdim=True)


def census_loss(image_a_bhw3, image_b_bhw3, mask_bhw3, patch_size=7, distance_metric_fn=abs_robust_loss):
    """比较两个图像的census变换的相似性"""
    census_image_a_bhwk = census_transform(image_a_bhw3, patch_size)
    census_image_b_bhwk = census_transform(image_b_bhw3, patch_size)
    
    hamming_bhw1 = soft_hamming(census_image_a_bhwk, census_image_b_bhwk)
    
    # 将掩码边界设置为零以忽略边缘效应
    padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)
    diff = distance_metric_fn(hamming_bhw1)
    diff *= padded_mask_bhw3
    diff_sum = torch.sum(diff)
    loss_mean = diff_sum / (torch.sum(padded_mask_bhw3.detach()) + 1e-6)
    return loss_mean


def time_it(f, num_reps=1, execute_once_before=False):
    """在eager模式下计时pytorch函数
    
    Args:
        f: 应该计时的无参数函数
        num_reps: 计时的重复次数
        execute_once_before: 是否在计时前执行一次函数
    
    Returns:
        平均时间（毫秒）和函数输出的元组
    """
    assert num_reps >= 1
    # 在计时前执行一次f以允许编译
    if execute_once_before:
        x = f()
    
    # 确保GPU上没有任何东西仍在运行
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时f的多次重复
    start_in_s = time.time()
    for _ in range(num_reps):
        x = f()
        # 确保f已完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if isinstance(x, (tuple, list)):
            _ = [torch.sum(xi) for xi in x]
        else:
            _ = torch.sum(x)
    end_in_s = time.time()
    
    # 计算平均时间（毫秒）
    avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
    return avg_time, x


def _avg_pool3x3(x):
    """3x3平均池化"""
    return F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=0).permute(0, 2, 3, 1)


def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """计算加权结构图像相似性度量
    
    Args:
        x: 表示图像批次的张量，形状为 [B, H, W, C]
        y: 表示图像批次的张量，形状为 [B, H, W, C]
        weight: 形状为 [B, H, W] 的张量，表示计算矩（均值和相关性）时每个像素的权重
        c1: 浮点数，正则化均值的除零
        c2: 浮点数，正则化二阶矩的除零
        weight_epsilon: 浮点数，用于正则化权重的除法
    
    Returns:
        两个张量的元组。第一个，形状为 [B, H-2, W-2, C]，是每像素每通道的标量相似性损失，
        第二个，形状为 [B, H-2, W-2, 1]，是平均池化的权重
    """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is likely unintended.')
    
    weight = weight.unsqueeze(-1)
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)
    
    def weighted_avg_pool3x3(z):
        weighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return weighted_avg * inverse_average_pooled_weight
    
    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight