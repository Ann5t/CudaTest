#include "PointCloud.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string(cudaGetErrorString(err)) + " (����: " #expr ")"); \
        } \
    } while (0)

__global__ void generate_pointcloud(
	const uint8_t* color,
	const uint16_t* depth,
	PointCloud* pcd
) {
	// ������ͼ��Χ�ڵ����ص㣬�����Ч�ԣ���v�е�u�У��Լ���gpu�����������idx
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (u >= INPUT_WIDTH || v >= INPUT_HEIGHT) return;
	int idx = v * INPUT_WIDTH + u;

	// Ĭ����Ϊ��Ч����������ж�Ϊ��Ч�Ḳ�ǣ�
	pcd->valid[idx] = 0;

	// ��ȡ���ֵ�������Ч��
	float z = depth[idx] * d_DEPTH_SCALE;
	//printf("Valid depth at (%d, %d, %d): %.3f mm, %.3f m, %f\n", u, v, idx, depth[idx], z, d_DEPTH_SCALE);
	if (z == 0 || z < d_DEPTH_MIN || z > d_DEPTH_MAX) return;
	
	// ���������ӳ�䵽����������ϵ��
	float x_d = (u - d_DEPTH_INTR[2]) * z / d_DEPTH_INTR[0];
	float y_d = (v - d_DEPTH_INTR[3]) * z / d_DEPTH_INTR[1];
	float z_d = z;

	// ��������ݴ�����������ϵת������ɫ�������ϵ������������󷽵ĵ�
	float px = d_EXTR_ROT[0] * x_d + d_EXTR_ROT[3] * y_d + d_EXTR_ROT[6] * z_d + d_EXTR_TRANS[0];
	float py = d_EXTR_ROT[1] * x_d + d_EXTR_ROT[4] * y_d + d_EXTR_ROT[7] * z_d + d_EXTR_TRANS[1];
	float pz = d_EXTR_ROT[2] * x_d + d_EXTR_ROT[5] * y_d + d_EXTR_ROT[8] * z_d + d_EXTR_TRANS[2];
	if (pz <= 0) return;

	// ����ͶӰ����ɫͼ��ƽ�棬��ȡ��Ӧ�Ĳ�ɫͼ���������꣬����ͶӰ��ͼ����ĵ�
	int u_c = __float2int_rn(d_COLOR_INTR[0] * px / pz + d_COLOR_INTR[2]);
	int v_c = __float2int_rn(d_COLOR_INTR[1] * py / pz + d_COLOR_INTR[3]);
	if (u_c < 0 || u_c >= INPUT_WIDTH || v_c < 0 || v_c >= INPUT_HEIGHT) return;

	// ���µľ�����Ч�㣬д���������
	int cidx = v_c * INPUT_WIDTH + u_c;
	pcd->x[idx] = px;
	pcd->y[idx] = py;
	pcd->z[idx] = pz;
	pcd->r[idx] = color[cidx * 3 + 0];
	pcd->g[idx] = color[cidx * 3 + 1];
	pcd->b[idx] = color[cidx * 3 + 2];
	pcd->valid[idx] = 1;
}