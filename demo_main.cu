/**
 * @file demo_main.cu
 * @brief RealSense + CUDA图像处理 + OpenCV实时可视化演示程序
 * 
 * 这个文件展示了一个标准的RealSense数据采集流水线：
 * 1. RealSense捕获原始RGB和深度数据
 * 2. 数据直接传输到GPU（不转换为cv::Mat）
 * 3. 在CUDA中进行图像处理
 * 4. 将处理结果传回CPU并用OpenCV显示
 * 
 * 核心优势：
 * - 避免CPU端的数据格式转换开销
 * - 利用GPU并行计算加速图像处理
 * - 最小化CPU-GPU数据传输
 */

#include "RealsenseCamera.hpp"
#include "ImageProcessingDemo.cuh"
#include "Constant.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// CUDA错误检查宏
#define CHECK_CUDA_DEMO(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " (" #expr ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief 演示程序入口
 * 
 * 流程：
 * 1. 初始化RealSense相机
 * 2. 分配GPU内存缓冲区
 * 3. 主循环：
 *    a. 获取原始帧
 *    b. 上传到GPU
 *    c. CUDA处理
 *    d. 下载到CPU
 *    e. OpenCV显示
 */
int main() {
    std::cout << "=== RealSense + CUDA + OpenCV 演示程序 ===" << std::endl;
    std::cout << "分辨率: " << INPUT_WIDTH << "x" << INPUT_HEIGHT << std::endl;
    std::cout << "帧率: " << INPUT_FPS << " FPS" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 1. 初始化RealSense相机
    // ========================================================================
    std::cout << "[1/4] 初始化RealSense相机..." << std::endl;
    RealsenseCamera cam;
    cam.warmup(30);  // 丢弃前30帧，确保数据稳定
    std::cout << "      相机初始化完成！" << std::endl;

    // ========================================================================
    // 2. 分配GPU内存
    // ========================================================================
    std::cout << "[2/4] 分配GPU内存..." << std::endl;
    
    // 输入缓冲区
    uint8_t* d_raw_color = nullptr;    // RGB图像 (W*H*3)
    uint16_t* d_raw_depth = nullptr;   // 深度图像 (W*H*2)
    
    // 输出缓冲区
    uint8_t* d_edge_output = nullptr;       // 边缘检测结果 (W*H)
    uint8_t* d_depth_colorized = nullptr;   // 深度伪彩色 (W*H*3)
    uint8_t* d_bgr_output = nullptr;        // BGR转换结果 (W*H*3)
    
    CHECK_CUDA_DEMO(cudaMalloc(&d_raw_color, INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t)));
    CHECK_CUDA_DEMO(cudaMalloc(&d_raw_depth, INPUT_PIXEL_COUNT * sizeof(uint16_t)));
    CHECK_CUDA_DEMO(cudaMalloc(&d_edge_output, INPUT_PIXEL_COUNT * sizeof(uint8_t)));
    CHECK_CUDA_DEMO(cudaMalloc(&d_depth_colorized, INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t)));
    CHECK_CUDA_DEMO(cudaMalloc(&d_bgr_output, INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t)));
    
    std::cout << "      GPU内存分配完成！" << std::endl;
    std::cout << "      - 输入RGB: " << (INPUT_PIXEL_COUNT * 3) / 1024 << " KB" << std::endl;
    std::cout << "      - 输入深度: " << (INPUT_PIXEL_COUNT * 2) / 1024 << " KB" << std::endl;

    // ========================================================================
    // 3. 初始化图像处理器和CPU缓冲区
    // ========================================================================
    std::cout << "[3/4] 初始化图像处理器..." << std::endl;
    
    CudaImageProcessor processor(INPUT_WIDTH, INPUT_HEIGHT);
    
    // CPU端显示缓冲区（用于OpenCV imshow）
    std::vector<uint8_t> h_color_bgr(INPUT_PIXEL_COUNT * 3);
    std::vector<uint8_t> h_edge(INPUT_PIXEL_COUNT);
    std::vector<uint8_t> h_depth_color(INPUT_PIXEL_COUNT * 3);
    
    std::cout << "      处理器初始化完成！" << std::endl;

    // ========================================================================
    // 4. 创建OpenCV窗口
    // ========================================================================
    std::cout << "[4/4] 创建显示窗口..." << std::endl;
    
    cv::namedWindow("原始彩色 (BGR)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("边缘检测 (CUDA)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("深度伪彩色 (CUDA)", cv::WINDOW_AUTOSIZE);
    
    std::cout << "      窗口创建完成！" << std::endl;
    std::cout << std::endl;
    std::cout << "=== 开始实时处理 ===" << std::endl;
    std::cout << "按 'q' 或 ESC 退出" << std::endl;
    std::cout << std::endl;

    // CUDA执行配置
    dim3 block(16, 16);
    dim3 grid((INPUT_WIDTH + 15) / 16, (INPUT_HEIGHT + 15) / 16);

    // 性能统计
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    float total_upload_time = 0.0f;
    float total_process_time = 0.0f;
    float total_download_time = 0.0f;

    // ========================================================================
    // 主处理循环
    // ========================================================================
    while (true) {
        auto frame_start = std::chrono::steady_clock::now();
        
        // --------------------------------------------------------------------
        // 步骤A: 获取原始帧数据
        // --------------------------------------------------------------------
        const uint8_t* raw_color = nullptr;
        const uint16_t* raw_depth = nullptr;
        cam.get_raw_frames(raw_color, raw_depth);
        
        // --------------------------------------------------------------------
        // 步骤B: 上传数据到GPU（Host -> Device）
        // --------------------------------------------------------------------
        auto upload_start = std::chrono::steady_clock::now();
        
        CHECK_CUDA_DEMO(cudaMemcpy(d_raw_color, raw_color, 
            INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_DEMO(cudaMemcpy(d_raw_depth, raw_depth, 
            INPUT_PIXEL_COUNT * sizeof(uint16_t), cudaMemcpyHostToDevice));
        
        auto upload_end = std::chrono::steady_clock::now();
        
        // --------------------------------------------------------------------
        // 步骤C: CUDA图像处理
        // --------------------------------------------------------------------
        auto process_start = std::chrono::steady_clock::now();
        
        // C.1: RGB转BGR（用于OpenCV显示）
        cuda_rgb_to_bgr<<<grid, block>>>(d_raw_color, d_bgr_output, 
            INPUT_WIDTH, INPUT_HEIGHT);
        
        // C.2: 边缘检测流水线（灰度 -> 模糊 -> Sobel）
        processor.process_edge_detection(d_raw_color, d_edge_output);
        
        // C.3: 深度图伪彩色化
        processor.process_depth_visualization(d_raw_depth, d_depth_colorized, 
            DEPTH_MIN * 1000.0f, DEPTH_MAX * 1000.0f);  // 转换为毫米
        
        CHECK_CUDA_DEMO(cudaDeviceSynchronize());
        
        auto process_end = std::chrono::steady_clock::now();
        
        // --------------------------------------------------------------------
        // 步骤D: 下载结果到CPU（Device -> Host）
        // --------------------------------------------------------------------
        auto download_start = std::chrono::steady_clock::now();
        
        CHECK_CUDA_DEMO(cudaMemcpy(h_color_bgr.data(), d_bgr_output, 
            INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_DEMO(cudaMemcpy(h_edge.data(), d_edge_output, 
            INPUT_PIXEL_COUNT * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_DEMO(cudaMemcpy(h_depth_color.data(), d_depth_colorized, 
            INPUT_PIXEL_COUNT * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        
        auto download_end = std::chrono::steady_clock::now();
        
        // --------------------------------------------------------------------
        // 步骤E: OpenCV显示（直接包装CPU buffer，无拷贝）
        // --------------------------------------------------------------------
        cv::Mat color_mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, h_color_bgr.data());
        cv::Mat edge_mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1, h_edge.data());
        cv::Mat depth_mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, h_depth_color.data());
        
        cv::imshow("原始彩色 (BGR)", color_mat);
        cv::imshow("边缘检测 (CUDA)", edge_mat);
        cv::imshow("深度伪彩色 (CUDA)", depth_mat);
        
        // --------------------------------------------------------------------
        // 统计性能
        // --------------------------------------------------------------------
        frame_count++;
        
        float upload_ms = std::chrono::duration<float, std::milli>(
            upload_end - upload_start).count();
        float process_ms = std::chrono::duration<float, std::milli>(
            process_end - process_start).count();
        float download_ms = std::chrono::duration<float, std::milli>(
            download_end - download_start).count();
        
        total_upload_time += upload_ms;
        total_process_time += process_ms;
        total_download_time += download_ms;
        
        // 每60帧打印一次统计
        if (frame_count % 60 == 0) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - start_time).count();
            float fps = frame_count / elapsed;
            
            std::cout << "帧数: " << frame_count 
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | 上传: " << std::setprecision(2) << total_upload_time / frame_count << "ms"
                      << " | 处理: " << total_process_time / frame_count << "ms"
                      << " | 下载: " << total_download_time / frame_count << "ms"
                      << std::endl;
        }
        
        // --------------------------------------------------------------------
        // 检测退出按键
        // --------------------------------------------------------------------
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' 或 ESC
            std::cout << std::endl;
            std::cout << "=== 程序退出 ===" << std::endl;
            break;
        }
    }

    // ========================================================================
    // 清理资源
    // ========================================================================
    std::cout << "清理GPU内存..." << std::endl;
    
    cudaFree(d_raw_color);
    cudaFree(d_raw_depth);
    cudaFree(d_edge_output);
    cudaFree(d_depth_colorized);
    cudaFree(d_bgr_output);
    
    cv::destroyAllWindows();
    
    // 打印最终统计
    std::cout << std::endl;
    std::cout << "=== 最终统计 ===" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "平均上传时间: " << total_upload_time / frame_count << " ms" << std::endl;
    std::cout << "平均处理时间: " << total_process_time / frame_count << " ms" << std::endl;
    std::cout << "平均下载时间: " << total_download_time / frame_count << " ms" << std::endl;
    
    return 0;
}
