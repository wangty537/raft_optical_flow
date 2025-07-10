# LiteFlowNet3 ONNX模型导出与可视化指南

本指南介绍如何导出LiteFlowNet3模型为ONNX格式并进行可视化。

## 🚀 快速开始

### 方法1: 使用独立导出脚本（推荐）

```bash
# 基本用法
python export_onnx.py

# 指定模型类型
python export_onnx.py --model liteflownet3s

# 自定义输入尺寸
python export_onnx.py --model liteflownet3 --input_size 512 768

# 指定输出目录
python export_onnx.py --model liteflownet3 --output_dir ./my_models
```

### 方法2: 直接运行模型文件

```bash
python liteflownet3_simple.py
```

## 📋 支持的模型

| 模型名称 | 描述 |
|---------|------|
| `liteflownet3` | 标准LiteFlowNet3模型 |
| `liteflownet3_pseudoreg` | 带伪正则化的LiteFlowNet3 |
| `liteflownet3s` | 轻量版LiteFlowNet3 |
| `liteflownet3s_pseudoreg` | 带伪正则化的轻量版LiteFlowNet3 |

## 🎨 模型可视化

### 在线可视化（推荐）

1. 访问 [Netron在线版](https://netron.app)
2. 拖拽或上传生成的`.onnx`文件
3. 即可查看模型结构

### 本地可视化

```bash
# 安装Netron
pip install netron

# 可视化模型
netron liteflownet3_384x512.onnx

# 或者使用命令行模式
python -m netron liteflownet3_384x512.onnx
```

## 📁 输出文件

导出成功后，你将得到：

```
onnx_models/
└── liteflownet3_384x512.onnx  # ONNX模型文件
```

文件命名格式：`{模型名称}_{高度}x{宽度}.onnx`

## 🔧 依赖要求

```bash
# 必需依赖
pip install torch onnx

# 可视化依赖（可选）
pip install netron
```

## 📊 模型信息

### 输入格式
- **名称**: `images`
- **形状**: `[batch_size, 2, 3, height, width]`
- **类型**: `float32`
- **描述**: 连续的两帧RGB图像

### 输出格式
- **flows**: `[batch_size, num_levels, 2, height, width]` - 光流场
- **confs**: `[batch_size, num_levels, 1, height, width]` - 置信度图

### 动态轴
- `batch_size`: 支持不同的批次大小
- 输入图像的高度和宽度在导出时固定

## 🎯 使用示例

### 导出不同尺寸的模型

```bash
# 小尺寸模型（适合移动端）
python export_onnx.py --model liteflownet3s --input_size 256 384

# 标准尺寸模型
python export_onnx.py --model liteflownet3 --input_size 384 512

# 高分辨率模型
python export_onnx.py --model liteflownet3 --input_size 512 768
```

### 批量导出所有模型

```bash
# 导出所有模型变体
for model in liteflownet3 liteflownet3_pseudoreg liteflownet3s liteflownet3s_pseudoreg; do
    python export_onnx.py --model $model --input_size 384 512
done
```

## 🐛 故障排除

### 常见问题

1. **导出失败**
   ```
   解决方案：
   - 检查PyTorch和ONNX版本兼容性
   - 确保有足够的内存和磁盘空间
   - 检查模型文件是否完整
   ```

2. **可视化失败**
   ```
   解决方案：
   - 确保ONNX文件完整且未损坏
   - 更新Netron到最新版本
   - 尝试在线版本的Netron
   ```

3. **内存不足**
   ```
   解决方案：
   - 使用较小的输入尺寸
   - 选择轻量版模型（liteflownet3s）
   - 关闭其他占用内存的程序
   ```

### 版本兼容性

- **PyTorch**: >= 1.7.0
- **ONNX**: >= 1.8.0
- **Python**: >= 3.7

## 📈 性能对比

| 模型 | 参数量 | ONNX文件大小 | 推荐用途 |
|------|--------|-------------|----------|
| liteflownet3s | ~1.3M | ~5MB | 移动端/边缘设备 |
| liteflownet3 | ~5.4M | ~22MB | 桌面端/服务器 |

## 🔗 相关链接

- [Netron官网](https://netron.app)
- [ONNX官方文档](https://onnx.ai/)
- [PyTorch ONNX导出指南](https://pytorch.org/docs/stable/onnx.html)

---

**提示**: 建议先使用较小的模型和输入尺寸进行测试，确认导出流程正常后再导出完整模型。