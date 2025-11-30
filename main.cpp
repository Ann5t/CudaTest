#include "RealsenseCamera.hpp"
#include "PointCloud.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "Visualizer.hpp"

// 定义错误检查宏
#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            std::string msg = std::string(cudaGetErrorString(err)) + \
                              " (操作: " #expr ")"; \
            throw std::runtime_error(msg); \
        } \
    } while (0)

void validate_gpu_constants() {
    float h_color_intr[4], h_depth_intr[4];
    float h_extr_rot[9], h_extr_trans[3];
    float h_depth_scale, h_depth_min, h_depth_max;

    // 从 GPU 常量内存拷贝回主机
    CHECK_CUDA(cudaMemcpyFromSymbol(h_color_intr, d_COLOR_INTR, 4 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(h_depth_intr, d_DEPTH_INTR, 4 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(h_extr_rot, d_EXTR_ROT, 9 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(h_extr_trans, d_EXTR_TRANS, 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_depth_scale, d_DEPTH_SCALE, sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_depth_min, d_DEPTH_MIN, sizeof(float)));
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_depth_max, d_DEPTH_MAX, sizeof(float)));

    // 打印验证
    printf("=== GPU Constant Memory Verification ===\n");
    printf("Color Intrinsics (fx, fy, cx, cy): %.6f, %.6f, %.6f, %.6f\n",
        h_color_intr[0], h_color_intr[1], h_color_intr[2], h_color_intr[3]);
    printf("Depth Intrinsics (fx, fy, cx, cy): %.6f, %.6f, %.6f, %.6f\n",
        h_depth_intr[0], h_depth_intr[1], h_depth_intr[2], h_depth_intr[3]);
    printf("Extrinsics Rotation (first 3): %.6f, %.6f, %.6f\n",
        h_extr_rot[0], h_extr_rot[1], h_extr_rot[2]);
    printf("Extrinsics Translation: %.6f, %.6f, %.6f\n",
        h_extr_trans[0], h_extr_trans[1], h_extr_trans[2]);
    printf("Depth Scale: %.10f\n", h_depth_scale);
    printf("Depth Min/Max: %.3f / %.3f\n", h_depth_min, h_depth_max);

    // 关键检查：depth scale 是否为 nan 或 0？
    if (std::isnan(h_depth_scale) || h_depth_scale == 0.0f) {
        fprintf(stderr, "❌ CRITICAL: Depth scale is invalid! (nan or zero)\n");
    }
}

int main() {
    RealsenseCamera cam;
    cam.warmup();

    // 把相机内参/外参等常量拷贝到 GPU 常量内存，验证无问题
    CHECK_CUDA(cudaMemcpyToSymbol(d_COLOR_INTR, cam.get_color_intrinsics(), 4 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_DEPTH_INTR, cam.get_depth_intrinsics(), 4 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_EXTR_ROT, cam.get_extrinsics_rotation(), 9 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_EXTR_TRANS, cam.get_extrinsics_translation(), 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_DEPTH_SCALE, cam.get_depth_scale(), sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_DEPTH_MIN, &DEPTH_MIN, sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_DEPTH_MAX, &DEPTH_MAX, sizeof(float)));

    // 设备端输入帧缓冲（预分配）
    uint8_t* d_raw_color = nullptr;
    uint16_t* d_raw_depth = nullptr;
    CHECK_CUDA(cudaMalloc(&d_raw_color, sizeof(uint8_t) * INPUT_PIXEL_COUNT * 3));
    CHECK_CUDA(cudaMalloc(&d_raw_depth, sizeof(uint16_t) * INPUT_PIXEL_COUNT));

    // 设备端点云 SOA 缓冲（每帧复用）
    PointCloud* d_pcd;
    CHECK_CUDA(cudaMalloc(&d_pcd, sizeof(PointCloud)));
    
    dim3 block(16, 16);
    dim3 grid((INPUT_WIDTH + block.x - 1) / block.x, (INPUT_HEIGHT + block.y - 1) / block.y);

    while (true) {
        // 获取原始帧指针到cpu
        const uint8_t* raw_color = nullptr;
        const uint16_t* raw_depth = nullptr;
        cam.get_raw_frames(raw_color, raw_depth);

        // 拷贝帧到gpu
        CHECK_CUDA(cudaMemcpy(d_raw_color, raw_color, sizeof(uint8_t) * INPUT_PIXEL_COUNT * 3, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_raw_depth, raw_depth, sizeof(uint16_t) * INPUT_PIXEL_COUNT, cudaMemcpyHostToDevice));

		// 计算点云
        generate_pointcloud<<<grid, block>>>(
            d_raw_color,
            d_raw_depth,
            d_pcd
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 可视化检查
        visualize(d_raw_color, d_raw_depth);
        savePointCloudToPLY(d_pcd, INPUT_PIXEL_COUNT, "output.ply");
    }

    // 释放（不可达）
    cudaFree(d_raw_color);
    cudaFree(d_raw_depth);
    cudaFree(d_pcd);
    return 0;
}