#pragma once

#include "Constant.hpp"
#include <cuda_runtime.h>

// 仅声明 - 定义在 PointCloud.cpp 中
extern __constant__ float d_COLOR_INTR[4];    // 4个颜色内参 (fx, fy, cx, cy)
extern __constant__ float d_DEPTH_INTR[4];    // 4个深度内参
extern __constant__ float d_EXTR_ROT[9];      // 3x3旋转矩阵
extern __constant__ float d_EXTR_TRANS[3];    // 3个平移量
extern __constant__ float d_DEPTH_SCALE;      // 深度缩放因子（单个float）
extern __constant__ float d_DEPTH_MIN;        // 最小有效深度（单个float）
extern __constant__ float d_DEPTH_MAX;        // 最大有效深度（单个float）

// ���ƣ�ÿ֡���ô�СΪ INPUT_PIXEL_COUNT �Ļ���
struct PointCloud {
    float x[INPUT_PIXEL_COUNT];
    float y[INPUT_PIXEL_COUNT];
    float z[INPUT_PIXEL_COUNT];
    uint8_t r[INPUT_PIXEL_COUNT];
    uint8_t g[INPUT_PIXEL_COUNT];
    uint8_t b[INPUT_PIXEL_COUNT];
    uint8_t valid[INPUT_PIXEL_COUNT];
};

__global__ void generate_pointcloud(
    const uint8_t* __restrict__ color,
    const uint16_t* __restrict__ depth,
    PointCloud* cloud
);