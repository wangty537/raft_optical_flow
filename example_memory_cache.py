#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存缓存功能使用示例

这个脚本演示了如何使用 FlowDataset 的内存缓存功能来提高数据加载速度。
内存缓存会在初始化时将所有图像和光流数据加载到内存中，避免重复的磁盘I/O操作。
"""

import sys
import os
sys.path.append('core')

import time
import torch
from torch.utils.data import DataLoader
from datasets import MpiSintel, FlyingChairs, KITTI

def benchmark_dataset(dataset_class, dataset_kwargs, num_samples=10):
    """
    对比启用和不启用缓存的数据加载速度
    """
    print(f"\n=== {dataset_class.__name__} 基准测试 ===")
    
    # 不启用缓存
    print("\n1. 不启用缓存:")
    dataset_no_cache = dataset_class(**dataset_kwargs, preload_data=False)
    
    start_time = time.time()
    for i in range(min(num_samples, len(dataset_no_cache))):
        img1, img2, flow = dataset_no_cache[i]
    no_cache_time = time.time() - start_time
    print(f"   加载 {min(num_samples, len(dataset_no_cache))} 个样本耗时: {no_cache_time:.3f} 秒")
    
    # 启用缓存
    print("\n2. 启用缓存:")
    dataset_with_cache = dataset_class(**dataset_kwargs, preload_data=True)
    
    # 第一次访问（包含预加载时间）
    start_time = time.time()
    for i in range(min(num_samples, len(dataset_with_cache))):
        img1, img2, flow = dataset_with_cache[i]
    first_access_time = time.time() - start_time
    print(f"   首次加载 {min(num_samples, len(dataset_with_cache))} 个样本耗时: {first_access_time:.3f} 秒（包含预加载）")
    
    # 第二次访问（纯缓存访问）
    start_time = time.time()
    for i in range(min(num_samples, len(dataset_with_cache))):
        img1, img2, flow = dataset_with_cache[i]
    cache_access_time = time.time() - start_time
    print(f"   缓存访问 {min(num_samples, len(dataset_with_cache))} 个样本耗时: {cache_access_time:.3f} 秒")
    
    # 显示缓存信息
    cache_info = dataset_with_cache.get_cache_info()
    print(f"   缓存状态: {cache_info}")
    
    # 计算加速比
    if cache_access_time > 0:
        speedup = no_cache_time / cache_access_time
        print(f"   加速比: {speedup:.2f}x")
    
    return dataset_with_cache

def test_dataloader_with_cache():
    """
    测试在 DataLoader 中使用缓存
    """
    print("\n=== DataLoader 缓存测试 ===")
    
    # 创建一个小的测试数据集
    try:
        dataset = FlyingChairs(preload_data=True)
        
        # 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        print(f"数据集大小: {len(dataset)}")
        print(f"批次大小: {dataloader.batch_size}")
        
        # 测试几个批次
        start_time = time.time()
        for i, (img1, img2, flow) in enumerate(dataloader):
            if i >= 3:  # 只测试前3个批次
                break
            print(f"批次 {i+1}: img1 shape={img1.shape}, img2 shape={img2.shape}, flow shape={flow.shape}")
        
        elapsed_time = time.time() - start_time
        print(f"加载3个批次耗时: {elapsed_time:.3f} 秒")
        
    except Exception as e:
        print(f"DataLoader 测试失败: {e}")

def main():
    """
    主函数：演示内存缓存功能
    """
    print("FlowDataset 内存缓存功能演示")
    print("=" * 50)
    
    # 测试不同的数据集（根据实际可用的数据集调整）
    datasets_to_test = [
        # (MpiSintel, {'split': 'training', 'dstype': 'clean'}),
        # (FlyingChairs, {'split': 'training'}),
        # (KITTI, {'split': 'training'}),
    ]
    
    # 如果没有可用的数据集，创建一个模拟示例
    if not datasets_to_test:
        print("\n注意: 没有找到可用的数据集路径")
        print("请根据您的数据集路径修改 example_memory_cache.py 中的数据集配置")
        print("\n示例用法:")
        print("```python")
        print("# 启用内存缓存")
        print("dataset = MpiSintel(preload_data=True)")
        print("")
        print("# 获取缓存信息")
        print("cache_info = dataset.get_cache_info()")
        print("print(f'缓存状态: {cache_info}')")
        print("")
        print("# 清理缓存")
        print("dataset.clear_cache()")
        print("```")
        return
    
    cached_datasets = []
    
    # 对每个数据集进行基准测试
    for dataset_class, kwargs in datasets_to_test:
        try:
            cached_dataset = benchmark_dataset(dataset_class, kwargs, num_samples=5)
            cached_datasets.append(cached_dataset)
        except Exception as e:
            print(f"测试 {dataset_class.__name__} 时出错: {e}")
    
    # 测试 DataLoader
    test_dataloader_with_cache()
    
    # 清理缓存
    print("\n=== 清理缓存 ===")
    for dataset in cached_datasets:
        dataset.clear_cache()
        print(f"已清理 {dataset.__class__.__name__} 的缓存")
    
    print("\n演示完成！")
    print("\n使用建议:")
    print("1. 对于小型数据集，启用缓存可以显著提高训练速度")
    print("2. 对于大型数据集，请确保有足够的内存")
    print("3. 在多进程 DataLoader 中使用时，每个进程都会有自己的缓存副本")
    print("4. 可以使用 get_cache_info() 监控内存使用情况")
    print("5. 训练结束后使用 clear_cache() 释放内存")

if __name__ == '__main__':
    main()