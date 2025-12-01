#pragma once

#include "Constant.hpp"
#include <cuda_runtime.h>
#include <cstdint>

/**
 * @file ImageProcessingDemo.cuh
 * @brief CUDA图像处理演示头文件
 * 
 * 这个文件展示了如何在CUDA中直接处理RealSense原始图像数据，
 * 不需要转换为OpenCV Mat格式，从而获得最高的处理效率。
 */

// ============================================================================
// CUDA图像处理核函数声明
// ============================================================================

/**
 * @brief RGB转灰度图
 * 
 * 使用加权平均法将RGB图像转换为灰度图
 * Gray = 0.299*R + 0.587*G + 0.114*B
 * 
 * @param d_rgb_input  输入RGB图像指针（设备内存，RGB排列）
 * @param d_gray_output 输出灰度图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void cuda_rgb_to_gray(
    const uint8_t* __restrict__ d_rgb_input,
    uint8_t* __restrict__ d_gray_output,
    int width,
    int height
);

/**
 * @brief 高斯模糊
 * 
 * 使用5x5高斯核进行图像模糊处理
 * 
 * @param d_input 输入灰度图像指针（设备内存）
 * @param d_output 输出模糊图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void cuda_gaussian_blur(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
);

/**
 * @brief Sobel边缘检测
 * 
 * 使用Sobel算子检测图像边缘
 * 
 * @param d_input 输入灰度图像指针（设备内存）
 * @param d_output 输出边缘图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void cuda_sobel_edge(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
);

/**
 * @brief 深度图伪彩色化
 * 
 * 将16位深度图转换为彩色可视化图像
 * 使用热度图（从蓝到红）表示深度
 * 
 * @param d_depth_input 输入深度图像指针（设备内存，16位）
 * @param d_rgb_output 输出RGB图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 * @param min_depth 最小深度值（毫米）
 * @param max_depth 最大深度值（毫米）
 */
__global__ void cuda_depth_colorize(
    const uint16_t* __restrict__ d_depth_input,
    uint8_t* __restrict__ d_rgb_output,
    int width,
    int height,
    float min_depth,
    float max_depth
);

/**
 * @brief 二值化阈值处理
 * 
 * 将灰度图像转换为二值图像
 * 
 * @param d_input 输入灰度图像指针（设备内存）
 * @param d_output 输出二值图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 * @param threshold 阈值（0-255）
 */
__global__ void cuda_threshold(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height,
    uint8_t threshold
);

/**
 * @brief RGB色彩空间转换
 * 
 * 将RGB图像转换为BGR格式（OpenCV默认格式）
 * 
 * @param d_rgb_input 输入RGB图像指针（设备内存）
 * @param d_bgr_output 输出BGR图像指针（设备内存）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void cuda_rgb_to_bgr(
    const uint8_t* __restrict__ d_rgb_input,
    uint8_t* __restrict__ d_bgr_output,
    int width,
    int height
);

// ============================================================================
// 主机端封装函数声明
// ============================================================================

/**
 * @brief 图像处理流水线封装类
 * 
 * 提供方便的主机端API来调用CUDA图像处理核函数
 */
class CudaImageProcessor {
public:
    CudaImageProcessor(int width, int height);
    ~CudaImageProcessor();

    // 禁用拷贝
    CudaImageProcessor(const CudaImageProcessor&) = delete;
    CudaImageProcessor& operator=(const CudaImageProcessor&) = delete;

    /**
     * @brief 执行完整的图像处理流水线
     * 
     * 包括：灰度转换 -> 高斯模糊 -> 边缘检测
     * 
     * @param d_rgb_input 输入RGB图像（设备内存）
     * @param d_edge_output 输出边缘图像（设备内存）
     */
    void process_edge_detection(
        const uint8_t* d_rgb_input,
        uint8_t* d_edge_output
    );

    /**
     * @brief 深度图可视化处理
     * 
     * @param d_depth_input 输入深度图像（设备内存）
     * @param d_rgb_output 输出伪彩色图像（设备内存）
     * @param min_depth 最小深度（毫米）
     * @param max_depth 最大深度（毫米）
     */
    void process_depth_visualization(
        const uint16_t* d_depth_input,
        uint8_t* d_rgb_output,
        float min_depth = 300.0f,
        float max_depth = 3000.0f
    );

    // 获取处理后的中间结果（灰度图）
    uint8_t* get_gray_buffer() const { return d_gray_buffer_; }
    
    // 获取处理后的中间结果（模糊图）
    uint8_t* get_blur_buffer() const { return d_blur_buffer_; }

private:
    int width_;
    int height_;
    int pixel_count_;
    
    // CUDA中间处理缓冲区
    uint8_t* d_gray_buffer_;   // 灰度图缓冲区
    uint8_t* d_blur_buffer_;   // 模糊图缓冲区
    
    // CUDA执行配置
    dim3 block_;
    dim3 grid_;
};
