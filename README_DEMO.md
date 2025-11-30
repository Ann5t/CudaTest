# RealSense + CUDA + OpenCV 实时图像处理演示

## 概述

这个演示展示了一个标准的高性能图像处理流水线：

```
RealSense相机 → 原始数据 → GPU内存 → CUDA处理 → CPU内存 → OpenCV显示
```

**核心特点：**
- ✅ 不转换为cv::Mat，直接使用原始指针传输数据
- ✅ GPU加速图像处理
- ✅ 最小化CPU-GPU数据传输
- ✅ 实时显示处理结果

## 文件结构

```
CudaTest/
├── ImageProcessingDemo.cuh  # CUDA图像处理核函数声明
├── ImageProcessingDemo.cu   # CUDA图像处理核函数实现
├── demo_main.cu             # 演示程序主入口
└── README_DEMO.md           # 本说明文档
```

## 数据流程详解

### 1. 数据采集（RealSense → CPU）

```cpp
// 获取原始帧指针（无拷贝）
const uint8_t* raw_color = nullptr;
const uint16_t* raw_depth = nullptr;
cam.get_raw_frames(raw_color, raw_depth);
```

RealSense SDK 返回的是指向内部缓冲区的原始指针：
- `raw_color`: RGB888格式，每像素3字节
- `raw_depth`: Z16格式，每像素2字节

### 2. 数据上传（CPU → GPU）

```cpp
// 直接拷贝原始数据到GPU（不经过cv::Mat）
cudaMemcpy(d_raw_color, raw_color, 
    INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_raw_depth, raw_depth, 
    INPUT_PIXEL_COUNT * sizeof(uint16_t), cudaMemcpyHostToDevice);
```

### 3. GPU图像处理

演示实现了以下CUDA核函数：

| 核函数 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `cuda_rgb_to_gray` | RGB转灰度 | RGB888 | Gray8 |
| `cuda_gaussian_blur` | 5x5高斯模糊 | Gray8 | Gray8 |
| `cuda_sobel_edge` | Sobel边缘检测 | Gray8 | Gray8 |
| `cuda_depth_colorize` | 深度伪彩色 | Z16 | RGB888 |
| `cuda_rgb_to_bgr` | RGB→BGR转换 | RGB888 | BGR888 |
| `cuda_threshold` | 二值化阈值 | Gray8 | Gray8 |

### 4. 数据下载（GPU → CPU）

```cpp
// 拷贝处理结果回CPU
cudaMemcpy(h_color_bgr.data(), d_bgr_output, 
    INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
```

### 5. OpenCV显示

```cpp
// 直接包装CPU buffer为Mat（零拷贝）
cv::Mat color_mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, h_color_bgr.data());
cv::imshow("原始彩色", color_mat);
```

## 内存布局

### RGB图像 (RGB888)
```
地址: [pixel_0_R, pixel_0_G, pixel_0_B, pixel_1_R, pixel_1_G, pixel_1_B, ...]
大小: WIDTH * HEIGHT * 3 bytes
```

### 深度图像 (Z16)
```
地址: [pixel_0_depth_low, pixel_0_depth_high, pixel_1_depth_low, ...]
大小: WIDTH * HEIGHT * 2 bytes
单位: 毫米
```

## 性能优化要点

### 1. 避免不必要的数据转换
```cpp
// ❌ 不推荐：先转Mat再上传
cv::Mat mat(height, width, CV_8UC3, raw_color);
cudaMemcpy(d_data, mat.data, ...);

// ✅ 推荐：直接上传原始指针
cudaMemcpy(d_data, raw_color, ...);
```

### 2. 使用常量内存存储卷积核
```cpp
// 常量内存访问有缓存，适合频繁读取的小数据
__constant__ float c_gaussian_kernel[25];
```

### 3. 使用固定内存（Pinned Memory）加速传输
```cpp
// 对于频繁传输的数据，可以使用固定内存
uint8_t* h_pinned;
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);
```

### 4. 使用CUDA流实现异步处理
```cpp
// 可以重叠数据传输和计算
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
```

## 编译配置

在Visual Studio项目中添加以下文件：

**CUDA编译文件：**
- `ImageProcessingDemo.cu`
- `demo_main.cu`（如果作为单独程序）

**头文件：**
- `ImageProcessingDemo.cuh`

## 使用方法

1. 编译项目
2. 运行程序
3. 观察三个窗口：
   - 原始彩色图像（BGR格式）
   - 边缘检测结果
   - 深度伪彩色图像
4. 按 `q` 或 `ESC` 退出

## 扩展建议

### 添加更多图像处理算法

```cpp
// 在 ImageProcessingDemo.cuh 中声明
__global__ void cuda_bilateral_filter(...);
__global__ void cuda_harris_corner(...);
__global__ void cuda_optical_flow(...);

// 在 ImageProcessingDemo.cu 中实现
```

### 使用纹理内存加速2D访问
```cpp
// 对于需要频繁2D采样的操作，纹理内存更高效
texture<uint8_t, 2, cudaReadModeElementType> tex_input;
```

### 集成cuDNN深度学习推理
```cpp
// 可以将预处理后的数据直接传给cuDNN
// 无需额外的数据转换
```

## 常见问题

**Q: 为什么不直接在RealSense回调中上传GPU？**

A: RealSense回调运行在SDK内部线程，与CUDA context可能不在同一线程。
   需要正确管理CUDA context或使用主线程处理。

**Q: 如何减少延迟？**

A: 
1. 使用CUDA流实现流水线并行
2. 使用Pinned Memory
3. 减少同步点
4. 考虑使用零拷贝内存（对于集成GPU）

**Q: 深度图单位是什么？**

A: RealSense D435返回的Z16深度值单位是毫米(mm)。
   可以通过`depth_sensor.get_depth_scale()`获取精确比例。
