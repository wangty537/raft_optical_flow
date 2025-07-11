import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)  # 设置OpenCV线程数为0，避免多线程冲突
cv2.ocl.setUseOpenCL(False)  # 禁用OpenCL加速，确保稳定性

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    """
    光流数据增强器
    
    原理：通过对输入的图像对和光流进行随机变换来增强训练数据的多样性，
    包括空间变换（缩放、裁剪、翻转）、光度变换（颜色抖动）和遮挡变换。
    这些增强技术能够提高模型的泛化能力和鲁棒性。
    """
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # 空间增强参数
        self.crop_size = crop_size  # 裁剪尺寸 (H, W)
        self.min_scale = min_scale  # 最小缩放比例（对数尺度）
        self.max_scale = max_scale  # 最大缩放比例（对数尺度）
        self.spatial_aug_prob = 0.8  # 空间增强概率
        self.stretch_prob = 0.8  # 非等比缩放概率
        self.max_stretch = 0.2  # 最大拉伸程度

        # 翻转增强参数
        self.do_flip = do_flip  # 是否启用翻转
        self.h_flip_prob = 0.5  # 水平翻转概率
        self.v_flip_prob = 0.1  # 垂直翻转概率

        # 光度增强参数
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)  # 颜色抖动变换
        self.asymmetric_color_aug_prob = 0.2  # 非对称颜色增强概率
        self.eraser_aug_prob = 0.5  # 遮挡增强概率

    def color_transform(self, img1, img2):
        """ 
        光度增强变换
        
        原理：通过随机调整图像的亮度、对比度、饱和度和色调来模拟不同的光照条件。
        支持对称和非对称两种模式：
        - 对称模式：对两幅图像应用相同的颜色变换
        - 非对称模式：对两幅图像应用不同的颜色变换
        """

        # 非对称颜色增强：对两幅图像分别应用不同的颜色变换
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # 对称颜色增强：对两幅图像应用相同的颜色变换
        else:
            image_stack = np.concatenate([img1, img2], axis=0)  # 将两幅图像垂直拼接
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)  # 重新分离两幅图像

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ 
        遮挡增强变换
        
        原理：在第二幅图像上随机添加矩形遮挡区域，用图像的平均颜色填充。
        这模拟了现实场景中的遮挡情况，提高模型对遮挡的鲁棒性。
        """

        ht, wd = img1.shape[:2]  # 获取图像高度和宽度
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)  # 计算图像平均颜色
            # 随机生成1-2个遮挡区域
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)  # 遮挡区域左上角x坐标
                y0 = np.random.randint(0, ht)  # 遮挡区域左上角y坐标
                dx = np.random.randint(bounds[0], bounds[1])  # 遮挡区域宽度
                dy = np.random.randint(bounds[0], bounds[1])  # 遮挡区域高度
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color  # 用平均颜色填充遮挡区域

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        """
        空间变换增强
        
        原理：通过缩放、翻转和裁剪操作来增强数据的空间多样性。
        包括：
        1. 随机缩放：模拟不同距离的拍摄
        2. 非等比缩放：模拟镜头畸变
        3. 翻转：增加数据的对称性
        4. 随机裁剪：关注图像的不同区域
        """
        # 随机采样缩放比例
        ht, wd = img1.shape[:2]  # 获取原始图像尺寸
        # 计算最小缩放比例，确保缩放后图像尺寸不小于裁剪尺寸
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        # 在对数空间中随机采样缩放比例
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale  # x方向缩放比例
        scale_y = scale  # y方向缩放比例
        
        # 随机应用非等比缩放（拉伸变换）
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        # 限制缩放比例不小于最小值
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # 随机应用空间变换
        if np.random.rand() < self.spatial_aug_prob:
            # 对图像和光流进行缩放
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]  # 光流向量也需要相应缩放

        # 随机翻转变换
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # 水平翻转
                img1 = img1[:, ::-1]  # 图像水平翻转
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]  # 光流x分量取反，y分量不变

            if np.random.rand() < self.v_flip_prob:  # 垂直翻转
                img1 = img1[::-1, :]  # 图像垂直翻转
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]  # 光流y分量取反，x分量不变

        # 随机裁剪到目标尺寸
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])  # 随机选择裁剪起始y坐标
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])  # 随机选择裁剪起始x坐标
        
        # 执行裁剪操作
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        """
        执行完整的数据增强流程
        
        原理：按顺序应用光度增强、遮挡增强和空间增强，
        最后确保数组内存连续以提高后续处理效率。
        """
        img1, img2 = self.color_transform(img1, img2)  # 应用光度增强
        img1, img2 = self.eraser_transform(img1, img2)  # 应用遮挡增强
        img1, img2, flow = self.spatial_transform(img1, img2, flow)  # 应用空间增强

        # 确保数组内存连续，提高后续处理效率
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    """
    稀疏光流数据增强器
    
    原理：专门用于稀疏光流数据的增强，与密集光流增强器类似，
    但针对稀疏数据的特点进行了优化。主要用于处理关键点匹配
    或稀疏光流估计任务。
    
    参数：
        crop_size: 裁剪后的图像尺寸
        min_scale: 最小缩放比例（对数空间）
        max_scale: 最大缩放比例（对数空间）
        do_flip: 是否启用翻转增强
    """
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # 空间增强参数
        self.crop_size = crop_size  # 裁剪尺寸
        self.min_scale = min_scale  # 最小缩放比例
        self.max_scale = max_scale  # 最大缩放比例
        self.spatial_aug_prob = 0.8  # 空间增强概率
        self.stretch_prob = 0.8  # 拉伸变换概率
        self.max_stretch = 0.2  # 最大拉伸幅度

        # 翻转增强参数
        self.do_flip = do_flip  # 是否启用翻转
        self.h_flip_prob = 0.5  # 水平翻转概率
        self.v_flip_prob = 0.1  # 垂直翻转概率

        # 光度增强参数
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2  # 非对称颜色增强概率
        self.eraser_aug_prob = 0.5  # 遮挡增强概率
        
    def color_transform(self, img1, img2):
        """
        光度变换增强（稀疏版本）
        
        原理：对两帧图像应用相同的光度变换，保持稀疏匹配点的一致性。
        与密集版本不同，这里总是对两帧图像应用相同的变换。
        """
        # 将两帧图像堆叠，确保应用相同的光度变换
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)  # 分离两帧图像
        return img1, img2

    def eraser_transform(self, img1, img2):
        """
        遮挡变换增强（稀疏版本）
        
        原理：在第二帧图像上随机添加矩形遮挡区域，模拟现实中的遮挡情况。
        对于稀疏光流，这有助于提高算法对遮挡的鲁棒性。
        """
        ht, wd = img1.shape[:2]  # 获取图像尺寸
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)  # 计算图像平均颜色
            # 随机添加1-2个遮挡区域
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)  # 遮挡区域左上角x坐标
                y0 = np.random.randint(0, ht)  # 遮挡区域左上角y坐标
                dx = np.random.randint(50, 100)  # 遮挡区域宽度
                dy = np.random.randint(50, 100)  # 遮挡区域高度
                # 用平均颜色填充遮挡区域
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        """
        调整稀疏光流图的尺寸
        
        原理：对稀疏光流进行缩放，需要同时处理坐标变换和光流向量缩放。
        与密集光流不同，稀疏光流只在有效像素位置有值，需要特殊处理。
        
        参数：
            flow: 稀疏光流图 [H, W, 2]
            valid: 有效性掩码 [H, W]
            fx, fy: x和y方向的缩放因子
        
        返回：
            缩放后的光流图和有效性掩码
        """
        ht, wd = flow.shape[:2]  # 原始尺寸
        # 创建坐标网格
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        # 展平为一维数组便于处理
        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        # 提取有效的坐标和光流
        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        # 计算缩放后的尺寸
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        # 缩放坐标和光流向量
        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        # 将坐标四舍五入到整数像素位置
        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        # 过滤出在新图像范围内的点
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        # 创建新的稀疏光流图
        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        # 填充有效位置的光流值
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        """
        稀疏光流的空间变换增强
        
        原理：与密集光流类似，但需要额外处理有效性掩码，
        并使用专门的稀疏光流缩放函数。裁剪时允许一定的边界容差。
        """
        # 随机采样缩放比例
        ht, wd = img1.shape[:2]  # 获取原始图像尺寸
        # 计算最小缩放比例，确保缩放后图像尺寸不小于裁剪尺寸
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        # 在对数空间中随机采样缩放比例
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)  # x方向缩放比例
        scale_y = np.clip(scale, min_scale, None)  # y方向缩放比例

        # 随机应用空间变换
        if np.random.rand() < self.spatial_aug_prob:
            # 对图像进行缩放
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # 对稀疏光流进行特殊的缩放处理
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        # 随机翻转变换
        if self.do_flip:
            if np.random.rand() < 0.5:  # 水平翻转
                img1 = img1[:, ::-1]  # 图像水平翻转
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]  # 光流x分量取反，y分量不变
                valid = valid[:, ::-1]  # 有效性掩码也需要翻转

        # 裁剪时的边界容差，允许稍微超出边界
        margin_y = 20
        margin_x = 50

        # 随机选择裁剪位置，允许负坐标（边界容差）
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        # 将裁剪坐标限制在有效范围内
        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        # 执行裁剪操作
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        """
        执行完整的稀疏光流数据增强流程
        
        原理：按顺序应用光度增强、遮挡增强和空间增强，
        同时处理稀疏光流的有效性掩码，最后确保数组内存连续。
        
        参数：
            img1, img2: 输入的两帧图像
            flow: 稀疏光流图
            valid: 有效性掩码
        
        返回：
            增强后的图像、光流和有效性掩码
        """
        img1, img2 = self.color_transform(img1, img2)  # 应用光度增强
        img1, img2 = self.eraser_transform(img1, img2)  # 应用遮挡增强
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)  # 应用空间增强

        # 确保数组内存连续，提高后续处理效率
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
