#pragma once

#include <cuda_runtime.h>

class Point3D_Batch {
private:
	float* x_;
	float* y_;
	float* z_;
	size_t size_;

public:
    Point3D_Batch(size_t n);
    ~Point3D_Batch();

    Point3D_Batch(const Point3D_Batch&) = delete;
    Point3D_Batch& operator=(const Point3D_Batch&) = delete;

    __host__ __device__ float& x(size_t idx) { return x_[idx]; }
    __host__ __device__ float& y(size_t idx) { return y_[idx]; }
    __host__ __device__ float& z(size_t idx) { return z_[idx]; }

    __host__ __device__ size_t size() const { return size_; }

    __host__ __device__ float* get_x_ptr() const { return x_; }
    __host__ __device__ float* get_y_ptr() const { return y_; }
    __host__ __device__ float* get_z_ptr() const { return z_; }
};