#pragma once

#include <cuda_runtime.h>

// 三维直线批量数据结构（起点+终点表示法）
class Line3D_Batch {
private:
    float* sx_; // 起点x
    float* sy_; // 起点y
    float* sz_; // 起点z
    float* ex_; // 终点x
    float* ey_; // 终点y
    float* ez_; // 终点z
    size_t size_;

public:
    Line3D_Batch(size_t n);
    ~Line3D_Batch();

    Line3D_Batch(const Line3D_Batch&) = delete;
    Line3D_Batch& operator=(const Line3D_Batch&) = delete;

    __host__ __device__ float& sx(size_t idx) { return sx_[idx]; }
    __host__ __device__ float& sy(size_t idx) { return sy_[idx]; }
    __host__ __device__ float& sz(size_t idx) { return sz_[idx]; }
    __host__ __device__ float& ex(size_t idx) { return ex_[idx]; }
    __host__ __device__ float& ey(size_t idx) { return ey_[idx]; }
    __host__ __device__ float& ez(size_t idx) { return ez_[idx]; }

    __host__ __device__  size_t size() const { return size_; }
    
    __host__ __device__ float* get_sx_ptr() const { return sx_; }
    __host__ __device__ float* get_sy_ptr() const { return sy_; }
    __host__ __device__ float* get_sz_ptr() const { return sz_; }
    __host__ __device__ float* get_ex_ptr() const { return ex_; }
    __host__ __device__ float* get_ey_ptr() const { return ey_; }
    __host__ __device__ float* get_ez_ptr() const { return ez_; }
};