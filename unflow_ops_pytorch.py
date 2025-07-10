import torch
import torch.nn.functional as F
import numpy as np


def forward_warp_op(flow):
    """UnFlow Forward Warp Operation - PyTorch Implementation
    
    根据UnFlow项目的实现，forward_warp用于遮挡检测。
    它将源像素根据光流映射到目标位置，生成密度图。
    
    Args:
        flow: 光流张量，形状为 [batch, height, width, 2]
              类型: torch.Tensor (float32)
              flow[:,:,:,0] 是x方向位移，flow[:,:,:,1] 是y方向位移
    
    Returns:
        密度图，形状为 [batch, height, width, 1]
        类型: torch.Tensor (float32)
        值表示每个位置被映射到的密度/权重
    
    Note:
        这个实现基于UnFlow项目的forward_warp_op.cu.cc
        主要用于遮挡检测，通过统计每个位置的映射密度来识别遮挡区域
    """
    batch, height, width, _ = flow.shape
    device = flow.device
    
    # 创建源坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 扩展到batch维度
    x_coords = x_coords.unsqueeze(0).repeat(batch, 1, 1)  # [batch, height, width]
    y_coords = y_coords.unsqueeze(0).repeat(batch, 1, 1)  # [batch, height, width]
    
    # 根据光流计算目标位置 (UnFlow使用加法)
    flow_x = flow[:, :, :, 0]  # x方向位移
    flow_y = flow[:, :, :, 1]  # y方向位移
    
    target_x = x_coords + flow_x  # 目标x坐标
    target_y = y_coords + flow_y  # 目标y坐标
    
    # 初始化密度图
    density_map = torch.zeros(batch, height, width, 1, dtype=torch.float32, device=device)
    
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
            # 使用双线性插值分配密度
            # 计算四个邻近像素的坐标
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
            
            # 累加密度到对应位置
            density_map[b, y0, x0, 0] += w00
            density_map[b, y1, x0, 0] += w01
            density_map[b, y0, x1, 0] += w10
            density_map[b, y1, x1, 0] += w11
    
    return density_map


def backward_warp_op(image, flow):
    """UnFlow Backward Warp Operation - PyTorch Implementation
    
    根据UnFlow项目的实现，backward_warp用于图像变形。
    对于目标图像的每个像素，根据光流从源图像采样。
    
    Args:
        image: 输入图像，形状为 [batch, height, width, channels]
               类型: torch.Tensor (float32)
        flow: 光流张量，形状为 [batch, height, width, 2]
              类型: torch.Tensor (float32)
              flow[:,:,:,0] 是x方向位移，flow[:,:,:,1] 是y方向位移
    
    Returns:
        变形后的图像，形状为 [batch, height, width, channels]
        类型: torch.Tensor (float32)
    
    Note:
        这个实现基于UnFlow项目的backward_warp_op.cu.cc
        使用双线性插值进行后向采样，确保输出图像的每个像素都有值
    """
    batch, height, width, channels = image.shape
    device = image.device
    
    # 创建网格坐标
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 扩展到batch维度
    x_coords = x_coords.unsqueeze(0).repeat(batch, 1, 1)  # [batch, height, width]
    y_coords = y_coords.unsqueeze(0).repeat(batch, 1, 1)  # [batch, height, width]
    
    # 根据光流计算采样位置 (UnFlow backward warp使用减法)
    flow_x = flow[:, :, :, 0]  # x方向位移
    flow_y = flow[:, :, :, 1]  # y方向位移
    
    # 计算采样坐标 (backward warp: 从目标位置减去光流得到源位置)
    sample_x = x_coords - flow_x
    sample_y = y_coords - flow_y
    
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


def occlusion_detection(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    """基于前向和后向光流的遮挡检测
    
    使用UnFlow的方法检测遮挡区域：
    1. 使用forward_warp生成密度图
    2. 结合前向-后向一致性检查
    
    Args:
        flow_fw: 前向光流 [batch, height, width, 2]
        flow_bw: 后向光流 [batch, height, width, 2]
        alpha1: 密度阈值
        alpha2: 一致性阈值
    
    Returns:
        遮挡掩码 [batch, height, width, 1]，1表示非遮挡，0表示遮挡
    """
    # 1. 使用forward warp生成密度图
    density_fw = forward_warp_op(flow_fw)
    density_bw = forward_warp_op(flow_bw)
    
    # 2. 密度检测：密度低的区域可能是遮挡
    density_mask_fw = (density_fw > alpha1).float()
    density_mask_bw = (density_bw > alpha1).float()
    
    # 3. 前向-后向一致性检查
    # 使用backward warp将后向光流变形到前向
    flow_bw_warped = backward_warp_op(flow_bw, flow_fw)
    
    # 计算光流差异
    flow_diff = torch.norm(flow_fw + flow_bw_warped, dim=3, keepdim=True)
    flow_magnitude = torch.norm(flow_fw, dim=3, keepdim=True) + torch.norm(flow_bw_warped, dim=3, keepdim=True)
    
    # 一致性掩码
    consistency_mask = (flow_diff < alpha2 * (flow_magnitude + 1e-6)).float()
    
    # 综合遮挡掩码
    occlusion_mask = density_mask_fw * density_mask_bw * consistency_mask
    
    return occlusion_mask


# 测试函数
def test_unflow_ops():
    """测试UnFlow操作的实现"""
    batch, height, width = 2, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    image = torch.randn(batch, height, width, 3, device=device)
    flow = torch.randn(batch, height, width, 2, device=device) * 5  # 光流范围
    
    print("Testing UnFlow Operations...")
    
    # 测试forward warp
    print("\n1. Testing Forward Warp:")
    density = forward_warp_op(flow)
    print(f"Input flow shape: {flow.shape}")
    print(f"Output density shape: {density.shape}")
    print(f"Density range: [{density.min():.3f}, {density.max():.3f}]")
    
    # 测试backward warp
    print("\n2. Testing Backward Warp:")
    warped_image = backward_warp_op(image, flow)
    print(f"Input image shape: {image.shape}")
    print(f"Input flow shape: {flow.shape}")
    print(f"Output warped image shape: {warped_image.shape}")
    
    # 测试遮挡检测
    print("\n3. Testing Occlusion Detection:")
    flow_bw = torch.randn(batch, height, width, 2, device=device) * 5
    occlusion_mask = occlusion_detection(flow, flow_bw)
    print(f"Occlusion mask shape: {occlusion_mask.shape}")
    print(f"Non-occluded ratio: {occlusion_mask.mean():.3f}")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_unflow_ops()