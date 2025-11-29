#pragma once

#include <cuda_runtime.h>
#include "Line3D_Batch.cuh"

class Cube_Batch {
private:
    float* x0_; // 所有立方体的 x0（最小x）
    float* y0_; // 所有立方体的 y0
    float* z0_; // 所有立方体的 z0
    float* x1_; // 所有立方体的 x1（最大x）
    float* y1_; // 所有立方体的 y1
    float* z1_; // 所有立方体的 z1
    size_t size_; // 立方体总数

public:
    Cube_Batch(size_t n);
    ~Cube_Batch();

    Cube_Batch(const Cube_Batch&) = delete;
    Cube_Batch& operator=(const Cube_Batch&) = delete;

    __host__ __device__ float& x0(size_t idx) { return x0_[idx]; }
    __host__ __device__ float& y0(size_t idx) { return y0_[idx]; }
    __host__ __device__ float& z0(size_t idx) { return z0_[idx]; }
    __host__ __device__ float& x1(size_t idx) { return x1_[idx]; }
    __host__ __device__ float& y1(size_t idx) { return y1_[idx]; }
    __host__ __device__ float& z1(size_t idx) { return z1_[idx]; }

    __host__ __device__ size_t size() const { return size_; }
    
    __host__ __device__ float* get_x0_ptr() const { return x0_; }
    __host__ __device__ float* get_y0_ptr() const { return y0_; }
    __host__ __device__ float* get_z0_ptr() const { return z0_; }
    __host__ __device__ float* get_x1_ptr() const { return x1_; }
    __host__ __device__ float* get_y1_ptr() const { return y1_; }
    __host__ __device__ float* get_z1_ptr() const { return z1_; }
};

// 1. 批量计算：所有线 × 所有立方体 的相交关系
// 结果存储：intersect_pairs[2*i] = 线idx，intersect_pairs[2*i+1] = 立方体idx（记录所有相交对）
// 返回：相交对的总数（通过 d_intersect_count 返回）
__global__ void lines_intersect_cubes(
    const Line3D_Batch lines,
    const Cube_Batch cubes,
    size_t* d_intersect_pairs,
    size_t* d_intersect_count
);

// 2. 整理结果：给每个立方体分配“穿过它的线索引列表”
// 输入：intersect_pairs（相交对）、cube_line_offsets（每个立方体的线列表起始位置）
// 输出：cube_lines（每个立方体的线索引列表）
__global__ void build_cube_line_lists(
    const size_t* d_intersect_pairs,
    size_t intersect_count,
    const size_t* d_cube_line_offsets,
    size_t* d_cube_lines
);