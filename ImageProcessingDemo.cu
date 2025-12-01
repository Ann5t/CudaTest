#include "ImageProcessingDemo.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

/**
 * @file ImageProcessingDemo.cu
 * @brief CUDA图像处理演示实现
 * 
 * 这个文件展示了如何在CUDA中高效处理图像数据，
 * 直接操作GPU内存中的原始像素数据，避免CPU-GPU数据传输开销。
 */

// ============================================================================
// CUDA错误检查宏
// ============================================================================
#define CHECK_CUDA_IMG(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error in ImageProcessor: ") \
                + cudaGetErrorString(err) + " (" #expr ")"); \
        } \
    } while (0)

// ============================================================================
// 常量内存中的高斯核
// ============================================================================
__constant__ float c_gaussian_kernel[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
};

// Sobel X方向算子
__constant__ int c_sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Sobel Y方向算子
__constant__ int c_sobel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// ============================================================================
// CUDA核函数实现
// ============================================================================

__global__ void cuda_rgb_to_gray(
    const uint8_t* __restrict__ d_rgb_input,
    uint8_t* __restrict__ d_gray_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int rgb_idx = idx * 3;
    
    // RGB to Gray: Gray = 0.299*R + 0.587*G + 0.114*B
    // 使用整数运算加速：Gray = (77*R + 150*G + 29*B) >> 8
    uint8_t r = d_rgb_input[rgb_idx + 0];
    uint8_t g = d_rgb_input[rgb_idx + 1];
    uint8_t b = d_rgb_input[rgb_idx + 2];
    
    d_gray_output[idx] = (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
}

__global__ void cuda_gaussian_blur(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int kernel_idx = 0;
    
    // 5x5高斯卷积
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int nx = min(max(x + kx, 0), width - 1);  // 边界处理：钳位
            int ny = min(max(y + ky, 0), height - 1);
            
            sum += d_input[ny * width + nx] * c_gaussian_kernel[kernel_idx];
            kernel_idx++;
        }
    }
    
    d_output[y * width + x] = (uint8_t)fminf(fmaxf(sum, 0.0f), 255.0f);
}

__global__ void cuda_sobel_edge(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // 边界像素置0
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        d_output[y * width + x] = 0;
        return;
    }
    
    int gx = 0, gy = 0;
    int kernel_idx = 0;
    
    // 3x3 Sobel卷积
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pixel = d_input[(y + ky) * width + (x + kx)];
            gx += pixel * c_sobel_x[kernel_idx];
            gy += pixel * c_sobel_y[kernel_idx];
            kernel_idx++;
        }
    }
    
    // 计算梯度幅值
    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    d_output[y * width + x] = (uint8_t)fminf(magnitude, 255.0f);
}

__global__ void cuda_depth_colorize(
    const uint16_t* __restrict__ d_depth_input,
    uint8_t* __restrict__ d_rgb_output,
    int width,
    int height,
    float min_depth,
    float max_depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uint16_t depth_raw = d_depth_input[idx];
    
    int rgb_idx = idx * 3;
    
    // 无效深度显示为黑色
    if (depth_raw == 0) {
        d_rgb_output[rgb_idx + 0] = 0;
        d_rgb_output[rgb_idx + 1] = 0;
        d_rgb_output[rgb_idx + 2] = 0;
        return;
    }
    
    // 归一化深度到[0, 1]
    float depth = (float)depth_raw;
    float normalized = (depth - min_depth) / (max_depth - min_depth);
    normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
    
    // 热度图着色：蓝(近) -> 青 -> 绿 -> 黄 -> 红(远)
    // 使用HSV颜色空间思想，H从240°(蓝)到0°(红)
    float hue = (1.0f - normalized) * 240.0f;  // 240°到0°
    float h = hue / 60.0f;
    int i = (int)h;
    float f = h - i;
    
    uint8_t r, g, b;
    float v = 255.0f;
    float p = 0.0f;
    float q = v * (1.0f - f);
    float t = v * f;
    
    switch (i) {
        case 0: r = (uint8_t)v; g = (uint8_t)t; b = (uint8_t)p; break;  // 红到黄
        case 1: r = (uint8_t)q; g = (uint8_t)v; b = (uint8_t)p; break;  // 黄到绿
        case 2: r = (uint8_t)p; g = (uint8_t)v; b = (uint8_t)t; break;  // 绿到青
        case 3: r = (uint8_t)p; g = (uint8_t)q; b = (uint8_t)v; break;  // 青到蓝
        case 4: r = (uint8_t)t; g = (uint8_t)p; b = (uint8_t)v; break;  // 蓝到品红
        default: r = (uint8_t)v; g = (uint8_t)p; b = (uint8_t)q; break; // 品红到红
    }
    
    d_rgb_output[rgb_idx + 0] = r;
    d_rgb_output[rgb_idx + 1] = g;
    d_rgb_output[rgb_idx + 2] = b;
}

__global__ void cuda_threshold(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height,
    uint8_t threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    d_output[idx] = (d_input[idx] > threshold) ? 255 : 0;
}

__global__ void cuda_rgb_to_bgr(
    const uint8_t* __restrict__ d_rgb_input,
    uint8_t* __restrict__ d_bgr_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // RGB -> BGR 交换R和B通道
    d_bgr_output[idx + 0] = d_rgb_input[idx + 2];  // B <- R
    d_bgr_output[idx + 1] = d_rgb_input[idx + 1];  // G <- G
    d_bgr_output[idx + 2] = d_rgb_input[idx + 0];  // R <- B
}

// ============================================================================
// CudaImageProcessor 类实现
// ============================================================================

CudaImageProcessor::CudaImageProcessor(int width, int height)
    : width_(width)
    , height_(height)
    , pixel_count_(width * height)
    , d_gray_buffer_(nullptr)
    , d_blur_buffer_(nullptr)
    , block_(16, 16)
    , grid_((width + 15) / 16, (height + 15) / 16)
{
    // 分配GPU中间缓冲区
    CHECK_CUDA_IMG(cudaMalloc(&d_gray_buffer_, pixel_count_ * sizeof(uint8_t)));
    CHECK_CUDA_IMG(cudaMalloc(&d_blur_buffer_, pixel_count_ * sizeof(uint8_t)));
}

CudaImageProcessor::~CudaImageProcessor() {
    if (d_gray_buffer_) cudaFree(d_gray_buffer_);
    if (d_blur_buffer_) cudaFree(d_blur_buffer_);
}

void CudaImageProcessor::process_edge_detection(
    const uint8_t* d_rgb_input,
    uint8_t* d_edge_output
) {
    // 步骤1: RGB转灰度
    cuda_rgb_to_gray<<<grid_, block_>>>(
        d_rgb_input, d_gray_buffer_, width_, height_
    );
    
    // 步骤2: 高斯模糊（降噪）
    cuda_gaussian_blur<<<grid_, block_>>>(
        d_gray_buffer_, d_blur_buffer_, width_, height_
    );
    
    // 步骤3: Sobel边缘检测
    cuda_sobel_edge<<<grid_, block_>>>(
        d_blur_buffer_, d_edge_output, width_, height_
    );
    
    // 注意：在生产环境中，可以移除此同步调用并在调用者端统一同步以提高性能
    CHECK_CUDA_IMG(cudaDeviceSynchronize());
}

void CudaImageProcessor::process_depth_visualization(
    const uint16_t* d_depth_input,
    uint8_t* d_rgb_output,
    float min_depth,
    float max_depth
) {
    cuda_depth_colorize<<<grid_, block_>>>(
        d_depth_input, d_rgb_output, width_, height_, min_depth, max_depth
    );
    
    // 注意：在生产环境中，可以移除此同步调用并在调用者端统一同步以提高性能
    CHECK_CUDA_IMG(cudaDeviceSynchronize());
}
