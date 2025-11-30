#include "Constant.hpp"
#include "RealsenseCamera.hpp"
#include <iostream>

RealsenseCamera::RealsenseCamera() {
    // 启用流并启动管道
    CFG_.enable_stream(RS2_STREAM_COLOR, INPUT_WIDTH, INPUT_HEIGHT, RS2_FORMAT_RGB8, INPUT_FPS);
    CFG_.enable_stream(RS2_STREAM_DEPTH, INPUT_WIDTH, INPUT_HEIGHT, RS2_FORMAT_Z16, INPUT_FPS);
    PROFILE_ = PIPE_.start(CFG_);

    // 获取内参
    COLOR_INTRINICS_ = PROFILE_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    DEPTH_INTRINICS_ = PROFILE_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

    // 填充 COLOR_INTR_[4] = {fx, fy, cx, cy}
    COLOR_INTR_[0] = COLOR_INTRINICS_.fx;  // fx
    COLOR_INTR_[1] = COLOR_INTRINICS_.fy;  // fy
    COLOR_INTR_[2] = COLOR_INTRINICS_.ppx; // cx
    COLOR_INTR_[3] = COLOR_INTRINICS_.ppy; // cy

    // 填充 DEPTH_INTR_[4] = {fx, fy, cx, cy}
    DEPTH_INTR_[0] = DEPTH_INTRINICS_.fx;  // fx
    DEPTH_INTR_[1] = DEPTH_INTRINICS_.fy;  // fy
    DEPTH_INTR_[2] = DEPTH_INTRINICS_.ppx; // cx
    DEPTH_INTR_[3] = DEPTH_INTRINICS_.ppy; // cy

    // 获取外参：depth → color
    DEPTH_TO_COLOR_EXTRINSICS_ =
        PROFILE_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>()
        .get_extrinsics_to(
            PROFILE_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>()
        );

    // 提取外参标量
    for (int i = 0; i < 9; ++i) {
        EXTR_ROT_[i] = DEPTH_TO_COLOR_EXTRINSICS_.rotation[i];
    }
    for (int i = 0; i < 3; ++i) {
        EXTR_TRANS_[i] = DEPTH_TO_COLOR_EXTRINSICS_.translation[i];
    }

    // 获取深度比例
    DEPTH_SCALE_ = PROFILE_.get_device().first<rs2::depth_sensor>().get_depth_scale();
}

RealsenseCamera::~RealsenseCamera() {
    PIPE_.stop();
}

void RealsenseCamera::warmup(int frames_to_discard) {
    for (int i = 0; i < frames_to_discard; ++i) {
        PIPE_.wait_for_frames();
    }
}

void RealsenseCamera::get_raw_frames(const uint8_t*& color, const uint16_t*& depth) {
    // 获取当前帧集
    frames_ = PIPE_.wait_for_frames();
    // 提取深度和彩色帧指针
    color = static_cast<const uint8_t*>(frames_.get_color_frame().get_data());
    depth = static_cast<const uint16_t*>(frames_.get_depth_frame().get_data());
}

float* RealsenseCamera::get_color_intrinsics() {
    return COLOR_INTR_;
}

float* RealsenseCamera::get_depth_intrinsics() {
    return DEPTH_INTR_;
}

float* RealsenseCamera::get_extrinsics_rotation() {
    return EXTR_ROT_;
}

float* RealsenseCamera::get_extrinsics_translation() {
    return EXTR_TRANS_;
}

float* RealsenseCamera::get_depth_scale() {
    return &DEPTH_SCALE_;
}