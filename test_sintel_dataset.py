
from torch.utils.data import Dataset, DataLoader
from core import datasets
import argparse
import cv2
import numpy as np
import torch
import os

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help='batch size used during training')
    args = parser.parse_args()
    
    # 创建数据集
    aug_params = {'crop_size': [384, 512], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    #things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
    sintel_clean = datasets.MpiSintel(aug_params, split='training', dstype='clean')
    sintel_final = datasets.MpiSintel(aug_params, split='training', dstype='final')        

    print(len(sintel_clean), len(sintel_final))
    train_dataset = 100*sintel_clean + 100*sintel_final 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    val_dataset_sintel_clean = datasets.MpiSintel_val(split='training', dstype='clean')
    val_dataset_sintel_final = datasets.MpiSintel_val(split='training', dstype='final')
    val_loader_sintel_clean = DataLoader(val_dataset_sintel_clean, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    val_loader_sintel_final = DataLoader(val_dataset_sintel_final, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset_sintel_clean), len(val_dataset_sintel_final)}')
    


    img1, img2, flow, valid = next(iter(train_loader))
    print(img1.shape, img2.shape, flow.shape, valid.shape)

    img1, img2, flow, valid = next(iter(val_loader_sintel_final))
    print(img1.shape, img2.shape, flow.shape, valid.shape)

    # 测试数据集
    img1, img2, flow, valid = sintel_final[0]
    print(img1.shape, img2.shape, flow.shape, valid.shape)

    # 利用opencv 保存img1, img2, flow, valid为图片格式
    
    def save_tensor_as_image(tensor, save_path, is_flow=False):
        """
        将 PyTorch tensor 保存为图像
        
        Args:
            tensor: 输入张量 (C, H, W) 或 (H, W)
            save_path: 保存路径
            is_flow: 是否为光流数据
        """
        if tensor.dim() == 3:
            # (C, H, W) -> (H, W, C)
            img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # (H, W)
            img = tensor.cpu().numpy()
        
        if is_flow:
            # 光流可视化：将光流转换为HSV颜色空间
            if img.shape[2] == 2:  # 光流有两个通道 (u, v)
                h, w = img.shape[:2]
                hsv = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 计算光流的幅度和角度
                magnitude, angle = cv2.cartToPolar(img[..., 0], img[..., 1])
                
                # 设置HSV值
                hsv[..., 0] = angle * 180 / np.pi / 2  # 色调
                hsv[..., 1] = 255  # 饱和度
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度
                
                # 转换为BGR
                img = flow_to_rgb
            else:
                # 单通道光流，转换为灰度图
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # 普通图像处理
            if img.dtype != np.uint8:
                # 如果是浮点数，假设范围是[0,1]或[-1,1]
                if img.max() <= 1.0 and img.min() >= 0.0:
                    img = (img * 255).astype(np.uint8)
                elif img.max() <= 1.0 and img.min() >= -1.0:
                    img = ((img + 1) * 127.5).astype(np.uint8)
                else:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 如果是RGB，转换为BGR（OpenCV格式）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 保存图像
        cv2.imwrite(save_path, img)
        print(f"图像已保存到: {save_path}")
    
    def save_valid_mask(valid_tensor, save_path):
        """
        保存有效性掩码
        
        Args:
            valid_tensor: 有效性掩码张量
            save_path: 保存路径
        """
        valid_np = valid_tensor.cpu().numpy()
        if valid_np.dtype == bool:
            valid_np = valid_np.astype(np.uint8) * 255
        elif valid_np.dtype == np.float32:
            valid_np = (valid_np * 255).astype(np.uint8)
        
        cv2.imwrite(save_path, valid_np)
        print(f"有效性掩码已保存到: {save_path}")
    
    # 创建保存目录
    save_dir = "dataset_samples"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存第一个样本的数据
    print("\n正在保存数据集样本...")
    
    # 保存图像1
    save_tensor_as_image(img1, os.path.join(save_dir, "img1.png"))
    
    # 保存图像2
    save_tensor_as_image(img2, os.path.join(save_dir, "img2.png"))
    
    # 保存光流（彩色可视化）
    save_tensor_as_image(flow, os.path.join(save_dir, "flow_color.png"), is_flow=True)
    
    # 保存光流幅度图
    flow_np = flow.permute(1, 2, 0).cpu().numpy()
    magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "flow_magnitude.png"), magnitude_normalized)
    print(f"光流幅度图已保存到: {os.path.join(save_dir, 'flow_magnitude.png')}")
    
    # 保存有效性掩码
    save_valid_mask(valid, os.path.join(save_dir, "valid_mask.png"))
    
    # 保存光流的原始数据（.npy格式，便于后续分析）
    flow_np_raw = flow.cpu().numpy()
    np.save(os.path.join(save_dir, "flow_raw.npy"), flow_np_raw)
    print(f"光流原始数据已保存到: {os.path.join(save_dir, 'flow_raw.npy')}")
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"图像1 - 形状: {img1.shape}, 数据类型: {img1.dtype}, 范围: [{img1.min():.3f}, {img1.max():.3f}]")
    print(f"图像2 - 形状: {img2.shape}, 数据类型: {img2.dtype}, 范围: [{img2.min():.3f}, {img2.max():.3f}]")
    print(f"光流 - 形状: {flow.shape}, 数据类型: {flow.dtype}, 范围: [{flow.min():.3f}, {flow.max():.3f}]")
    print(f"有效掩码 - 形状: {valid.shape}, 数据类型: {valid.dtype}, 有效像素比例: {valid.float().mean():.3f}")
    
    # 光流统计
    flow_magnitude_mean = magnitude.mean()
    flow_magnitude_max = magnitude.max()
    print(f"光流幅度 - 平均值: {flow_magnitude_mean:.3f}, 最大值: {flow_magnitude_max:.3f}")
    
    print(f"\n所有文件已保存到目录: {save_dir}/")

