#pragma once

#include "Constant.hpp"
#include <cuda_runtime.h>

__constant__ float d_COLOR_INTR[4];    // 4个彩色内参 (fx, fy, cx, cy)
__constant__ float d_DEPTH_INTR[4];    // 4个深度内参
__constant__ float d_EXTR_ROT[9];      // 3x3旋转矩阵
__constant__ float d_EXTR_TRANS[3];    // 3个平移量
__constant__ float d_DEPTH_SCALE;      // 深度缩放因子（单个float）
__constant__ float d_DEPTH_MIN;        // 最小有效深度（单个float）
__constant__ float d_DEPTH_MAX;        // 最大有效深度（单个float）

// 点云，每帧复用大小为 INPUT_PIXEL_COUNT 的缓冲
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