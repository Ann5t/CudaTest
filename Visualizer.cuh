#pragma once
#include <cstdint>
#include "PointCloud.cuh"

// 注意：该函数会阻塞一小段时间（等待显示刷新）
void visualize(
    const uint8_t* d_raw_color,
    const uint16_t* d_raw_depth
);

void savePointCloudToPLY(
    const PointCloud* d_cloud,
    int num_points,
    const char* filename = "pointcloud.ply"
);