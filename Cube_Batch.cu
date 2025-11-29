#include "Cube_Batch.cuh"
#include <math.h>

// --------------------------
// Cube_Batch 类实现
// --------------------------
Cube_Batch::Cube_Batch(size_t n) : size_(n) {
    // 分配统一内存，存储所有立方体的参数
    cudaMallocManaged(&x0_, n * sizeof(float));
    cudaMallocManaged(&y0_, n * sizeof(float));
    cudaMallocManaged(&z0_, n * sizeof(float));
    cudaMallocManaged(&x1_, n * sizeof(float));
    cudaMallocManaged(&y1_, n * sizeof(float));
    cudaMallocManaged(&z1_, n * sizeof(float));
}

Cube_Batch::~Cube_Batch() {
    cudaFree(x0_);
    cudaFree(y0_);
    cudaFree(z0_);
    cudaFree(x1_);
    cudaFree(y1_);
    cudaFree(z1_);
}

// --------------------------
// 核心核函数1：计算所有线-立方体的相交对
// --------------------------
__global__ void lines_intersect_cubes(const Line3D_Batch lines, const Cube_Batch cubes,
    size_t* d_intersect_pairs, size_t* d_intersect_count) {
    // 每个线程处理 1条线 × 1个立方体（二维索引：line_idx = 线程块y，cube_idx = 线程块x）
    const size_t line_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t cube_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：线或立方体索引超出范围，直接返回
    if (line_idx >= lines.size() || cube_idx >= cubes.size()) return;

    // 取当前线的起点和终点
    const float sx = lines.get_sx_ptr()[line_idx];
    const float sy = lines.get_sy_ptr()[line_idx];
    const float sz = lines.get_sz_ptr()[line_idx];
    const float ex = lines.get_ex_ptr()[line_idx];
    const float ey = lines.get_ey_ptr()[line_idx];
    const float ez = lines.get_ez_ptr()[line_idx];

    // 取当前立方体的范围
    const float x0 = cubes.get_x0_ptr()[cube_idx];
    const float y0 = cubes.get_y0_ptr()[cube_idx];
    const float z0 = cubes.get_z0_ptr()[cube_idx];
    const float x1 = cubes.get_x1_ptr()[cube_idx];
    const float y1 = cubes.get_y1_ptr()[cube_idx];
    const float z1 = cubes.get_z1_ptr()[cube_idx];

    // 优化版 Liang-Barsky 算法：判断当前线是否穿过当前立方体
    const float dx = ex - sx;
    const float dy = ey - sy;
    const float dz = ez - sz;

    float t_min = 0.0f;
    float t_max = 1.0f;
    bool intersect = true;

    // 处理x轴
    if (fabsf(dx) < 1e-8f) {
        if (sx < x0 || sx > x1) intersect = false;
    }
    else {
        const float tx1 = (x0 - sx) / dx;
        const float tx2 = (x1 - sx) / dx;
        t_min = max(t_min, min(tx1, tx2));
        t_max = min(t_max, max(tx1, tx2));
        if (t_min > t_max) intersect = false;
    }

    // 处理y轴
    if (intersect && fabsf(dy) < 1e-8f) {
        if (sy < y0 || sy > y1) intersect = false;
    }
    else if (intersect) {
        const float ty1 = (y0 - sy) / dy;
        const float ty2 = (y1 - sy) / dy;
        t_min = max(t_min, min(ty1, ty2));
        t_max = min(t_max, max(ty1, ty2));
        if (t_min > t_max) intersect = false;
    }

    // 处理z轴
    if (intersect && fabsf(dz) < 1e-8f) {
        if (sz < z0 || sz > z1) intersect = false;
    }
    else if (intersect) {
        const float tz1 = (z0 - sz) / dz;
        const float tz2 = (z1 - sz) / dz;
        t_min = max(t_min, min(tz1, tz2));
        t_max = min(t_max, max(tz1, tz2));
        if (t_min > t_max) intersect = false;
    }

    // 最终判断：是否相交
    if (intersect && t_min <= 1.0f && t_max >= 0.0f) {
        // 原子操作：获取当前相交对的索引（避免线程冲突）
        size_t pair_idx = atomicAdd(d_intersect_count, 1);
        // 存储相交对：(线idx, 立方体idx)
        d_intersect_pairs[2 * pair_idx] = line_idx;
        d_intersect_pairs[2 * pair_idx + 1] = cube_idx;
    }
}

// --------------------------
// 核心核函数2：整理每个立方体的线列表
// --------------------------
__global__ void build_cube_line_lists(const size_t* d_intersect_pairs, size_t intersect_count,
    const size_t* d_cube_line_offsets, size_t* d_cube_lines) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intersect_count) return;

    // 取当前相交对：线idx + 立方体idx
    size_t line_idx = d_intersect_pairs[2 * idx];
    size_t cube_idx = d_intersect_pairs[2 * idx + 1];

    // 找到当前立方体的线列表起始位置，存入线idx
    size_t offset = d_cube_line_offsets[cube_idx];
    d_cube_lines[offset + idx - d_cube_line_offsets[0]] = line_idx;
}