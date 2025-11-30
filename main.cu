#include "RealsenseCamera.hpp"
#include "PointCloud.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "Visualizer.cuh"
#include <cstdio>

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

// Helper: copy device __constant__ symbols to host and print
static void print_device_constants_and_ptrs(uint8_t* d_raw_color, uint16_t* d_raw_depth) {
    float h_color_intr[4] = {0};
    float h_depth_intr[4] = {0};
    float h_extr_rot[9] = {0};
    float h_extr_trans[3] = {0};
    float h_depth_scale = 0.0f;
    float h_depth_min = 0.0f;
    float h_depth_max = 0.0f;

    // 从常量内存复制回主机
    cudaMemcpyFromSymbol(h_color_intr, d_COLOR_INTR, sizeof(h_color_intr));
    cudaMemcpyFromSymbol(h_depth_intr, d_DEPTH_INTR, sizeof(h_depth_intr));
    cudaMemcpyFromSymbol(h_extr_rot, d_EXTR_ROT, sizeof(h_extr_rot));
    cudaMemcpyFromSymbol(h_extr_trans, d_EXTR_TRANS, sizeof(h_extr_trans));
    cudaMemcpyFromSymbol(&h_depth_scale, d_DEPTH_SCALE, sizeof(h_depth_scale));
    cudaMemcpyFromSymbol(&h_depth_min, d_DEPTH_MIN, sizeof(h_depth_min));
    cudaMemcpyFromSymbol(&h_depth_max, d_DEPTH_MAX, sizeof(h_depth_max));

    // 打印
    printf("--- Device __constant__ values ---\n");
    printf("d_COLOR_INTR: [%.6f, %.6f, %.6f, %.6f]\n", h_color_intr[0], h_color_intr[1], h_color_intr[2], h_color_intr[3]);
    printf("d_DEPTH_INTR: [%.6f, %.6f, %.6f, %.6f]\n", h_depth_intr[0], h_depth_intr[1], h_depth_intr[2], h_depth_intr[3]);
    printf("d_EXTR_ROT (first 6): [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f] ...\n", h_extr_rot[0], h_extr_rot[1], h_extr_rot[2], h_extr_rot[3], h_extr_rot[4], h_extr_rot[5]);
    printf("d_EXTR_TRANS: [%.6f, %.6f, %.6f]\n", h_extr_trans[0], h_extr_trans[1], h_extr_trans[2]);
    printf("d_DEPTH_SCALE: %.10f\n", h_depth_scale);
    printf("d_DEPTH_MIN/MAX: %.6f / %.6f\n", h_depth_min, h_depth_max);
    printf("Device pointers: d_raw_color=%p, d_raw_depth=%p\n", (void*)d_raw_color, (void*)d_raw_depth);
    printf("---------------------------------\n");
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

    // 1: print after constants + allocation
    print_device_constants_and_ptrs(d_raw_color, d_raw_depth);

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

        // 2: print after copying frame data to device and before kernel launch
        print_device_constants_and_ptrs(d_raw_color, d_raw_depth);

        // 计算点云
        generate_pointcloud<<<grid, block>>>(
            d_raw_color,
            d_raw_depth,
            d_pcd
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 3: print after kernel finished
        print_device_constants_and_ptrs(d_raw_color, d_raw_depth);

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