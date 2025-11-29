#include "Point3D_Batch.cuh"

Point3D_Batch::Point3D_Batch(size_t n) : size_(n) {
    cudaMallocManaged(&x_, n * sizeof(float));
    cudaMallocManaged(&y_, n * sizeof(float));
    cudaMallocManaged(&z_, n * sizeof(float));
}

Point3D_Batch::~Point3D_Batch() {
    cudaFree(x_);
    cudaFree(y_);
    cudaFree(z_);
}