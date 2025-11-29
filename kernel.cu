#include "Point3D_Batch.cuh"
#include "Line3D_Batch.cuh"
#include "Cube_Batch.cuh"
#include <iostream>
#include <vector>

// 辅助函数：CPU 侧整理结果，方便查询（每个立方体对应的线列表）
std::vector<std::vector<size_t>> get_cube_line_lists(size_t cube_count,
    const size_t* d_cube_line_offsets,
    const size_t* d_cube_lines) {
    std::vector<std::vector<size_t>> result(cube_count);
    std::vector<size_t> h_offsets(cube_count);
    std::vector<size_t> h_lines(d_cube_line_offsets[cube_count]); // 总线数 = 最后一个偏移量

    // 从GPU拷贝结果到CPU
    cudaMemcpy(h_offsets.data(), d_cube_line_offsets, cube_count * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lines.data(), d_cube_lines, h_offsets[cube_count] * sizeof(size_t), cudaMemcpyDeviceToHost);

    // 给每个立方体分配线列表
    for (size_t cube_idx = 0; cube_idx < cube_count; cube_idx++) {
        size_t start = h_offsets[cube_idx];
        size_t end = h_offsets[cube_idx + 1];
        result[cube_idx].assign(h_lines.begin() + start, h_lines.begin() + end);
    }

    return result;
}

int main() {
    // --------------------------
    // 1. 准备数据：10条线 + 3个立方体（赋值部分完全不变！）
    // --------------------------
    size_t line_count = 10;  // 10条线
    size_t cube_count = 3;   // 3个立方体

    // 1.1 创建并初始化线（Line3D_Batch）
    Line3D_Batch lines(line_count);
    for (size_t i = 0; i < line_count; i++) {
        lines.sx(i) = 0.0f; lines.sy(i) = 0.0f; lines.sz(i) = 0.0f; // 起点(0,0,0)
        lines.ex(i) = static_cast<float>(i); lines.ey(i) = static_cast<float>(i); lines.ez(i) = static_cast<float>(i); // 终点(i,i,i)
    }

    // 1.2 创建并初始化立方体（Cube_Batch）
    Cube_Batch cubes(cube_count);
    // 立方体0：(0,0,0) ~ (2,2,2)
    cubes.x0(0) = 0.0f; cubes.y0(0) = 0.0f; cubes.z0(0) = 0.0f;
    cubes.x1(0) = 2.0f; cubes.y1(0) = 2.0f; cubes.z1(0) = 2.0f;
    // 立方体1：(2,2,2) ~ (4,4,4)
    cubes.x0(1) = 2.0f; cubes.y0(1) = 2.0f; cubes.z0(1) = 2.0f;
    cubes.x1(1) = 4.0f; cubes.y1(1) = 4.0f; cubes.z1(1) = 4.0f;
    // 立方体2：(4,4,4) ~ (6,6,6)
    cubes.x0(2) = 4.0f; cubes.y0(2) = 4.0f; cubes.z0(2) = 4.0f;
    cubes.x1(2) = 6.0f; cubes.y1(2) = 6.0f; cubes.z1(2) = 6.0f;

    // --------------------------
    // 2. GPU 批量计算相交关系（只改核函数调用：传 &lines 和 &cubes 指针）
    // --------------------------
    // 2.1 分配内存：存储相交对（最大可能：线数×立方体数，避免溢出）
    size_t max_pairs = line_count * cube_count;
    size_t* d_intersect_pairs = nullptr;
    cudaMallocManaged(&d_intersect_pairs, 2 * max_pairs * sizeof(size_t));

    // 2.2 分配内存：存储相交对总数
    size_t* d_intersect_count = nullptr;
    cudaMallocManaged(&d_intersect_count, sizeof(size_t));
    *d_intersect_count = 0; // 初始化为0

    // 2.3 启动核函数1：计算所有线-立方体相交对（传 &lines 和 &cubes，取地址）
    dim3 block(16, 16); // 二维线程块：16×16=256线程，适配GPU
    dim3 grid((cube_count + block.x - 1) / block.x,
        (line_count + block.y - 1) / block.y); // 二维网格：x=立方体，y=线
    lines_intersect_cubes << <grid, block >> > (&lines, &cubes, d_intersect_pairs, d_intersect_count); // 这里改！
    cudaDeviceSynchronize(); // 等待GPU算完

    size_t intersect_count = *d_intersect_count;
    std::cout << "总相交对数：" << intersect_count << "\n\n";

    // --------------------------
    // 3. 整理结果：每个立方体对应的线列表（完全不变！）
    // --------------------------
    // 3.1 分配内存：每个立方体的线列表偏移量（cube_count+1 个，最后一个是总长度）
    size_t* d_cube_line_offsets = nullptr;
    cudaMallocManaged(&d_cube_line_offsets, (cube_count + 1) * sizeof(size_t));
    // 初始化偏移量（先统计每个立方体的相交线数）
    cudaMemset(d_cube_line_offsets, 0, (cube_count + 1) * sizeof(size_t));
    for (size_t i = 0; i < intersect_count; i++) {
        size_t cube_idx = d_intersect_pairs[2 * i + 1];
        d_cube_line_offsets[cube_idx + 1]++;
    }
    // 计算前缀和（得到每个立方体的线列表起始位置）
    for (size_t i = 1; i <= cube_count; i++) {
        d_cube_line_offsets[i] += d_cube_line_offsets[i - 1];
    }

    // 3.2 分配内存：存储所有立方体的线列表
    size_t total_line_count = d_cube_line_offsets[cube_count];
    size_t* d_cube_lines = nullptr;
    cudaMallocManaged(&d_cube_lines, total_line_count * sizeof(size_t));

    // 3.3 启动核函数2：整理线列表（完全不变！）
    dim3 build_block(256);
    dim3 build_grid((intersect_count + build_block.x - 1) / build_block.x);
    build_cube_line_lists << <build_grid, build_block >> > (d_intersect_pairs, intersect_count,
        d_cube_line_offsets, d_cube_lines);
    cudaDeviceSynchronize();

    // --------------------------
    // 4. 查询结果（CPU侧，打印部分完全不变！）
    // --------------------------
    std::vector<std::vector<size_t>> cube_line_lists = get_cube_line_lists(cube_count, d_cube_line_offsets, d_cube_lines);

    // 打印每个立方体穿过的线
    for (size_t cube_idx = 0; cube_idx < cube_count; cube_idx++) {
        std::cout << "立方体" << cube_idx << "（范围：("
            << cubes.x0(cube_idx) << "," << cubes.y0(cube_idx) << "," << cubes.z0(cube_idx) << ") ~ ("
            << cubes.x1(cube_idx) << "," << cubes.y1(cube_idx) << "," << cubes.z1(cube_idx) << ")）\n";
        std::cout << "穿过的线索引：";
        for (size_t line_idx : cube_line_lists[cube_idx]) {
            std::cout << line_idx << " ";
        }
        std::cout << "\n\n";
    }

    // --------------------------
    // 5. 释放内存（完全不变！）
    // --------------------------
    cudaFree(d_intersect_pairs);
    cudaFree(d_intersect_count);
    cudaFree(d_cube_line_offsets);
    cudaFree(d_cube_lines);

    return 0;
}