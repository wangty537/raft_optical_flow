
import torch
import torch.nn as nn
import torch.nn.functional as F

backwarp_tenGrid = {}
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(tenFlow.device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    """
    反卷积层构建函数
    
    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        kernel_size (int): 卷积核大小，默认4
        stride (int): 步长，默认2
        padding (int): 填充，默认1
    
    Returns:
        nn.Sequential: 包含转置卷积和PReLU激活的序列模块
        
    Shape:
        Input: (B, in_planes, H, W)
        Output: (B, out_planes, H*stride, W*stride)
    """
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """
    卷积层构建函数
    
    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        kernel_size (int): 卷积核大小，默认3
        stride (int): 步长，默认1
        padding (int): 填充，默认1
        dilation (int): 膨胀率，默认1
    
    Returns:
        nn.Sequential: 包含卷积和PReLU激活的序列模块
        
    Shape:
        Input: (B, in_planes, H, W)
        Output: (B, out_planes, H//stride, W//stride)
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    """
    插值流估计块 (Interpolation Flow Block)
    
    该模块用于估计光流和遮罩，是RIFE网络的核心组件。
    采用多尺度处理，通过下采样-处理-上采样的方式估计光流。
    
    Args:
        in_planes (int): 输入通道数
        c (int): 中间特征通道数，默认64
    """
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        # 下采样层：将输入特征下采样4倍 (stride=2*2=4)
        # Shape: (B, in_planes, H, W) -> (B, c, H//4, W//4)
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),  # (B, in_planes, H, W) -> (B, c//2, H//2, W//2)
            conv(c//2, c, 3, 2, 1),          # (B, c//2, H//2, W//2) -> (B, c, H//4, W//4)
            )
        # 残差处理块：8个卷积层进行特征提取和细化
        # Shape: (B, c, H//4, W//4) -> (B, c, H//4, W//4)
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 输出层：转置卷积上采样并输出5通道(4通道光流+1通道遮罩)
        # Shape: (B, c, H//4, W//4) -> (B, 5, H//2, W//2)
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        """
        IFBlock前向传播
        
        Args:
            x (torch.Tensor): 输入特征张量
                - 第一次调用: (B, 7, H, W) - img0(3) + img1(3) + timestep(1)
                - 后续调用: (B, 18, H, W) - img0(3) + img1(3) + timestep(1) + warped_img0(3) + warped_img1(3) + mask(1) + flow(4)
            flow (torch.Tensor or None): 前一级的光流，形状为 (B, 4, H, W)
                - flow[:, :2]: img0到中间帧的光流
                - flow[:, 2:4]: img1到中间帧的光流
            scale (int): 当前处理的尺度 (4, 2, 1)
        
        Returns:
            tuple: (flow, mask)
                - flow (torch.Tensor): 光流张量，形状 (B, 4, H, W)
                - mask (torch.Tensor): 遮罩张量，形状 (B, 1, H, W)
        """
        # 多尺度处理：根据scale缩放输入
        if scale != 1:
            # 将输入下采样到对应尺度 (B, C, H, W) -> (B, C, H//scale, W//scale)
            # 使用bilinear模式替代area模式以兼容ONNX导出
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        
        # 如果存在前一级光流，将其与输入特征拼接
        if flow != None:
            # 将光流也下采样到对应尺度，并调整光流数值
            # (B, 4, H, W) -> (B, 4, H//scale, W//scale)
            # 使用bilinear模式替代area模式以兼容ONNX导出
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            # 拼接特征和光流 (B, C, H//scale, W//scale) + (B, 4, H//scale, W//scale) -> (B, C+4, H//scale, W//scale)
            x = torch.cat((x, flow), 1)
        
        # 下采样特征提取 (B, C, H//scale, W//scale) -> (B, c, H//scale//4, W//scale//4)
        x = self.conv0(x)
        # 残差连接的特征细化 (B, c, H//scale//4, W//scale//4) -> (B, c, H//scale//4, W//scale//4)
        x = self.convblock(x) + x
        # 上采样并输出光流和遮罩 (B, c, H//scale//4, W//scale//4) -> (B, 5, H//scale//2, W//scale//2)
        tmp = self.lastconv(x)
        # 将输出上采样回原始尺度 (B, 5, H//scale//2, W//scale//2) -> (B, 5, H, W)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        # 分离光流和遮罩，并调整光流数值
        flow = tmp[:, :4] * scale * 2  # (B, 4, H, W) - 4通道光流
        mask = tmp[:, 4:5]             # (B, 1, H, W) - 1通道遮罩
        return flow, mask
class IFNet_m_flow(nn.Module):
    """
    RIFE插值网络主模型 (引入timestep支持，默认=0.5)
    
    该网络采用多尺度金字塔结构，通过学生-教师蒸馏机制进行训练。
    支持任意时间步长的帧插值，并使用上下文网络和U-Net进行最终细化。
    
    网络结构:
    - 3个学生IFBlock (多尺度: 4x, 2x, 1x)
    - 1个教师IFBlock (仅训练时使用)
    - ContextNet: 提取上下文特征
    - UNet: 最终图像细化
    """
    def __init__(self):
        super(IFNet_m_flow, self).__init__()
        # 学生网络：3个不同尺度的IFBlock
        self.block0 = IFBlock(6+1, c=240)    # 输入: img0(3) + img1(3) + timestep(1) = 7通道
        self.block1 = IFBlock(13+4+1, c=150) # 输入: img0(3) + img1(3) + timestep(1) + warped(6) + mask(1) + flow(4) = 18通道
        self.block2 = IFBlock(13+4+1, c=90)  # 输入: 同block1
        # # 教师网络：用于知识蒸馏
        # self.block_tea = IFBlock(16+4+1, c=90) # 输入: 比学生网络多gt(3)通道 = 21通道
        # # 上下文网络和细化网络
        # self.contextnet = Contextnet()  # 提取多尺度上下文特征
        # self.unet = Unet()             # 最终图像细化

    def forward(self, x, scale=[4,2,1], timestep=0.5, returnflow=False):
        """
        IFNet_m前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状 (B, C, H, W)
                - 训练时: (B, 9, H, W) - img0(3) + img1(3) + gt(3)
                - 推理时: (B, 6, H, W) - img0(3) + img1(3)
            scale (list): 多尺度处理的尺度列表，默认[4,2,1]
            timestep (float): 时间步长，0.5表示中间帧，默认0.5
            returnflow (bool): 是否只返回光流，默认False
        
        Returns:
            训练时: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
            推理时: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
            returnflow=True时: flow
            
            - flow_list (list): 3个尺度的光流列表，每个形状 (B, 4, H, W)
            - mask (torch.Tensor): 最终遮罩，形状 (B, 1, H, W)
            - merged (list): 3个尺度的融合图像列表，每个形状 (B, 3, H, W)
            - flow_teacher (torch.Tensor): 教师网络光流，形状 (B, 4, H, W)
            - merged_teacher (torch.Tensor): 教师网络融合图像，形状 (B, 3, H, W)
            - loss_distill (torch.Tensor): 蒸馏损失，标量
        """
        # 创建时间步长张量，形状与输入batch匹配 (B, 1, H, W)
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
        
        # 分离输入图像
        img0 = x[:, :3]   # (B, 3, H, W) - 第一帧
        img1 = x[:, 3:6]  # (B, 3, H, W) - 第二帧
        # # 处理gt：训练时有9通道，验证时只有6通道
        # if x.shape[1] > 6:
        #     gt = x[:, 6:]     # (B, 3, H, W) - 真值中间帧，训练时
        # else:
        #     gt = torch.zeros_like(img0)  # (B, 3, H, W) - 验证时创建空的gt
        
        # 初始化存储列表和变量
        flow_list = []    # 存储每个尺度的光流
        warped_list = []       # 存储每个尺度的融合图像
        mask_list = []    # 存储每个尺度的遮罩
        warped_img0 = img0  # 初始化变形图像
        warped_img1 = img1
        flow = None       # 初始光流为空
        
        
        # 学生网络列表
        stu = [self.block0, self.block1, self.block2]
        # 多尺度学生网络处理循环
        for i in range(3):
            if flow != None:
                # 非第一次迭代：使用前一级的光流和变形图像
                # 输入: img0(3) + img1(3) + timestep(1) + warped_img0(3) + warped_img1(3) + mask(1) + flow(4) = 18通道
                flow_d, mask_d = stu[i](torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                # 累加光流和遮罩增量
                flow = flow + flow_d    # (B, 4, H, W)
                mask = mask + mask_d    # (B, 1, H, W)
            else:
                # 第一次迭代：只使用原始图像和时间步长
                # 输入: img0(3) + img1(3) + timestep(1) = 7通道
                flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scale[i])
            
            # 保存当前尺度的结果
            mask_list.append(torch.sigmoid(mask))  # 将遮罩转换为概率值 (B, 1, H, W)
            flow_list.append(flow)                 # 保存光流 (B, 4, H, W)
            
            # 使用当前光流变形图像
            warped_img0 = warp(img0, flow[:, :2])  # 用前半部分光流变形img0 (B, 3, H, W)
            warped_img1 = warp(img1, flow[:, 2:4]) # 用后半部分光流变形img1 (B, 3, H, W)
            warped_list.append([warped_img0, warped_img1])
            # print(i, flow_list[-1].shape, mask_list[-1].shape, warped_list[-1][0].shape)
        # # 使用上下文网络和U-Net进行最终细化
        # c0 = self.contextnet(img0, flow[:, :2])  # 提取img0的上下文特征，多尺度特征
        # c1 = self.contextnet(img1, flow[:, 2:4]) # 提取img1的上下文特征，多尺度特征
        # # U-Net细化网络
        # tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)  # (B, 4, H, W)
        # flow += tmp
        return flow_list, mask_list, warped_list
        #  flow, mask, ret,  flow_teacher, ret_teacher, loss

# def ifnet_loss_unsupervised(warped_list, img0, img1):
#     for warped_img0, warped_img1 in warped_list:
#         # loss = F.mse_loss(warped_img0, img0) + F.mse_loss(warped_img1, img1)


# class IFNet_m_simple_flow(nn.Module):
#     """
#     RIFE插值网络简化版本
    
#     相比完整版本，该简化版本移除了ContextNet和UNet，
#     只保留多尺度IFBlock进行光流估计和图像融合。
#     适用于对计算资源有限制或需要更快推理速度的场景。
    
#     网络结构:
#     - 3个学生IFBlock (多尺度: 4x, 2x, 1x)
#     - 1个教师IFBlock (仅训练时使用)
#     - 无ContextNet和UNet细化
#     """
#     def __init__(self):
#         super(IFNet_m_simple, self).__init__()
#         # 学生网络：3个不同尺度的IFBlock
#         self.block0 = IFBlock(6+1, c=240)    # 输入: img0(3) + img1(3) + timestep(1) = 7通道
#         self.block1 = IFBlock(13+4+1, c=150) # 输入: img0(3) + img1(3) + timestep(1) + warped(6) + mask(1) + flow(4) = 18通道
#         self.block2 = IFBlock(13+4+1, c=90)  # 输入: 同block1
#         # 教师网络：用于知识蒸馏
#         self.block_tea = IFBlock(16+4+1, c=90) # 输入: 比学生网络多gt(3)通道 = 21通道
#         # 注释掉的上下文网络和细化网络（简化版本不使用）
#         # self.contextnet = Contextnet()
#         # self.unet = Unet()

#     def forward(self, x, scale=[4,2,1], timestep=0.5, returnflow=False):
#         """
#         IFNet_m_simple前向传播（简化版本）
        
#         与完整版本相比，该方法移除了ContextNet和UNet的细化步骤，
#         直接返回多尺度IFBlock的融合结果。
        
#         Args:
#             x (torch.Tensor): 输入张量，形状 (B, C, H, W)
#                 - 训练时: (B, 9, H, W) - img0(3) + img1(3) + gt(3)
#                 - 推理时: (B, 6, H, W) - img0(3) + img1(3)
#             scale (list): 多尺度处理的尺度列表，默认[4,2,1]
#             timestep (float): 时间步长，0.5表示中间帧，默认0.5
#             returnflow (bool): 是否只返回光流，默认False
        
#         Returns:
#             与完整版本相同的返回格式，但merged[2]未经过UNet细化
#         """
#         # 创建时间步长张量，形状与输入batch匹配 (B, 1, H, W)
#         timestep = (x[:, :1].clone() * 0 + 1) * timestep
        
#         # 分离输入图像
#         img0 = x[:, :3]   # (B, 3, H, W) - 第一帧
#         img1 = x[:, 3:6]  # (B, 3, H, W) - 第二帧
#         # # 处理gt：训练时有9通道，验证时只有6通道
#         # if x.shape[1] > 6:
#         #     gt = x[:, 6:]     # (B, 3, H, W) - 真值中间帧，训练时
#         # else:
#         #     gt = torch.zeros_like(img0)  # (B, 3, H, W) - 验证时创建空的gt
        
#         # 初始化存储列表和变量
#         flow_list = []    # 存储每个尺度的光流
#         merged = []       # 存储每个尺度的融合图像
#         mask_list = []    # 存储每个尺度的遮罩
#         warped_img0 = img0  # 初始化变形图像
#         warped_img1 = img1
#         flow = None       # 初始光流为空
#         loss_distill = 0  # 蒸馏损失初始化
        
#         # 学生网络列表
#         stu = [self.block0, self.block1, self.block2]
#         for i in range(3):
#             if flow != None:
#                 flow_d, mask_d = stu[i](torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
#                 flow = flow + flow_d
#                 mask = mask + mask_d
#             else:
#                 flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scale[i]) # 第一次进入
#             mask_list.append(torch.sigmoid(mask))
#             flow_list.append(flow)
#             warped_img0 = warp(img0, flow[:, :2])
#             warped_img1 = warp(img1, flow[:, 2:4])
#             #print('origin', i, warped_img0.mean(), warped_img1.mean(), flow.mean(), mask.mean())
#             merged_student = (warped_img0, warped_img1)
#             merged.append(merged_student)
#         if gt.shape[1] == 3:
#             flow_d, mask_d = self.block_tea(torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
#             flow_teacher = flow + flow_d
#             warped_img0_teacher = warp(img0, flow_teacher[:, :2])
#             warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
#             mask_teacher = torch.sigmoid(mask + mask_d)
#             merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
#         else:
#             flow_teacher = None
#             merged_teacher = None
#         for i in range(3):
#             merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
#             if gt.shape[1] == 3:
#                 loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
#                 loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
#         # 如果只需要光流，直接返回
#         if returnflow:
#             return flow  # (B, 4, H, W)
        
#         # 简化版本：直接返回多尺度融合结果，无ContextNet和UNet细化
#         # 注释掉的代码是完整版本的细化步骤：
#         # else:
#         #     # 使用上下文网络提取多尺度特征
#         #     c0 = self.contextnet(img0, flow[:, :2]) # 提取img0的上下文特征
#         #     c1 = self.contextnet(img1, flow[:, 2:4]) # 提取img1的上下文特征
#         #     # U-Net细化网络进行最终图像质量提升
#         #     tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
#         #     res = tmp[:, :3] * 2 - 1  # 细化残差
#         #     merged[2] = torch.clamp(merged[2] + res, 0, 1)  # 应用细化结果
        
#         return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


def compare_models():
    """
    比较IFNet_m和IFNet_m_simple的参数量、计算量、内存占用和运行时间
    """
    import time
    import torch.profiler
    from thop import profile, clever_format
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型实例
    model_full = IFNet_m_flow().to(device)
    model_simple = IFNet_m_flow().to(device)
    
    # 设置为评估模式
    model_full.eval()
    model_simple.eval()
    
    # 创建测试输入 (batch_size=1, channels=6, height=256, width=256)
    # 前3个通道是img0，后3个通道是img1
    test_input = torch.randn(1, 6, 256, 256).to(device)
    
    print("=" * 60)
    print("模型比较分析报告")
    print("=" * 60)
    
    # 1. 参数量比较
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_full = count_parameters(model_full)
    params_simple = count_parameters(model_simple)
    
    print(f"\n1. 参数量比较:")
    print(f"   IFNet_m:        {params_full:,} 参数")
    print(f"   IFNet_m_simple: {params_simple:,} 参数")
    print(f"   差异:           {params_full - params_simple:,} 参数 ({(params_full - params_simple) / params_full * 100:.2f}% 减少)")
    
    # 2. 计算量比较 (FLOPs)
    try:
        # 尝试使用thop计算FLOPs
        result_full = profile(model_full, inputs=(test_input,), verbose=False)
        result_simple = profile(model_simple, inputs=(test_input,), verbose=False)
        
        # 处理不同版本的thop返回值格式
        if isinstance(result_full, tuple):
            if len(result_full) == 2:
                flops_full, params_full_thop = result_full
                flops_simple, params_simple_thop = result_simple
            else:
                flops_full = result_full[0]
                flops_simple = result_simple[0]
        else:
            flops_full = result_full
            flops_simple = result_simple
        
        flops_full_str, _ = clever_format([flops_full], "%.3f")
        flops_simple_str, _ = clever_format([flops_simple], "%.3f")
        
        print(f"\n2. 计算量比较 (FLOPs):")
        print(f"   IFNet_m:        {flops_full_str}")
        print(f"   IFNet_m_simple: {flops_simple_str}")
        print(f"   差异:           {(flops_full - flops_simple) / flops_full * 100:.2f}% 减少")
    except Exception as e:
        print(f"\n2. 计算量比较 (FLOPs): 无法计算 - {e}")
        print("   提示: 请安装thop库: pip install thop")
        print("   或者thop库版本不兼容，可以尝试: pip install thop==0.1.1-2209072238")
    
    # 3. 内存占用比较
    def get_model_memory(model):
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    memory_full = get_model_memory(model_full)
    memory_simple = get_model_memory(model_simple)
    
    print(f"\n3. 模型内存占用比较:")
    print(f"   IFNet_m:        {memory_full / 1024 / 1024:.2f} MB")
    print(f"   IFNet_m_simple: {memory_simple / 1024 / 1024:.2f} MB")
    print(f"   差异:           {(memory_full - memory_simple) / 1024 / 1024:.2f} MB ({(memory_full - memory_simple) / memory_full * 100:.2f}% 减少)")
    
    # 4. 运行时间比较
    def benchmark_model(model, input_tensor, num_runs=100):
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        return (end_time - start_time) / num_runs
    
    print(f"\n4. 运行时间比较 (平均100次推理):")
    
    time_full = benchmark_model(model_full, test_input)
    time_simple = benchmark_model(model_simple, test_input)
    
    print(f"   IFNet_m:        {time_full * 1000:.2f} ms")
    print(f"   IFNet_m_simple: {time_simple * 1000:.2f} ms")
    print(f"   加速比:         {time_full / time_simple:.2f}x ({(time_full - time_simple) / time_full * 100:.2f}% 时间减少)")
    
    # 5. GPU内存使用比较（如果可用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model_full(test_input)
        memory_full_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model_simple(test_input)
        memory_simple_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"\n5. GPU内存使用比较:")
        print(f"   IFNet_m:        {memory_full_gpu:.2f} MB")
        print(f"   IFNet_m_simple: {memory_simple_gpu:.2f} MB")
        print(f"   差异:           {memory_full_gpu - memory_simple_gpu:.2f} MB ({(memory_full_gpu - memory_simple_gpu) / memory_full_gpu * 100:.2f}% 减少)")
    
    # 6. 模型结构差异分析
    print(f"\n6. 模型结构差异分析:")
    print(f"   主要差异: IFNet_m_simple移除了contextnet和unet模块")
    print(f"   - contextnet: 用于提取上下文特征")
    print(f"   - unet: 用于最终的图像细化")
    print(f"   影响: 简化版本在推理速度上更快，但可能在图像质量上有所损失")
    
    print("\n" + "=" * 60)
    print("比较完成")
    print("=" * 60)

if __name__ == "__main__":
    compare_models()











