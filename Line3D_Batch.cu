#include "Line3D_Batch.cuh"

Line3D_Batch::Line3D_Batch(size_t n) : size_(n) {
    cudaMallocManaged(&sx_, n * sizeof(float));
    cudaMallocManaged(&sy_, n * sizeof(float));
    cudaMallocManaged(&sz_, n * sizeof(float));
    cudaMallocManaged(&ex_, n * sizeof(float));
    cudaMallocManaged(&ey_, n * sizeof(float));
    cudaMallocManaged(&ez_, n * sizeof(float));
}

Line3D_Batch::~Line3D_Batch() {
    cudaFree(sx_);
    cudaFree(sy_);
    cudaFree(sz_);
    cudaFree(ex_);
    cudaFree(ey_);
    cudaFree(ez_);
}