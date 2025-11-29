#include "RealsenseCamera.cuh"

#include "Constant.hpp"
#include <cuda_runtime.h>

RealsenseCamera::RealsenseCamera() {
    // 启用流并启动管道
    CFG_.enable_stream(RS2_STREAM_COLOR, INPUT_WIDTH, INPUT_HEIGHT, RS2_FORMAT_RGB8, INPUT_FPS);
    CFG_.enable_stream(RS2_STREAM_DEPTH, INPUT_WIDTH, INPUT_HEIGHT, RS2_FORMAT_Z16, INPUT_FPS);
    PROFILE_ = PIPE_.start(CFG_);

    // 获取内外参
    COLOR_INTRINICS_ = PROFILE_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    DEPTH_INTRINICS_ = PROFILE_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    DEPTH_TO_COLOR_EXTRINSICS_ = PROFILE_
        .get_stream(RS2_STREAM_DEPTH)
        .as<rs2::video_stream_profile>()
        .get_extrinsics_to(
            PROFILE_
            .get_stream(RS2_STREAM_COLOR)
            .as<rs2::video_stream_profile>()
        );

    // 提取彩色内参标量
    COLOR_FX_ = COLOR_INTRINICS_.fx;
    COLOR_FY_ = COLOR_INTRINICS_.fy;
    COLOR_FX_INV_ = 1.0f / COLOR_FX_;
    COLOR_FY_INV_ = 1.0f / COLOR_FY_;
    COLOR_CX_ = COLOR_INTRINICS_.ppx;
    COLOR_CY_ = COLOR_INTRINICS_.ppy;

    // 提取深度内参标量
    DEPTH_FX_ = DEPTH_INTRINICS_.fx;
    DEPTH_FY_ = DEPTH_INTRINICS_.fy;
    DEPTH_FX_INV_ = 1.0f / DEPTH_FX_;
    DEPTH_FY_INV_ = 1.0f / DEPTH_FY_;
    DEPTH_CX_ = DEPTH_INTRINICS_.ppx;
    DEPTH_CY_ = DEPTH_INTRINICS_.ppy;

	// 提取外参标量，列主序
    for (int i = 0; i < 9; ++i) {
        EXTR_ROT_[i] = DEPTH_TO_COLOR_EXTRINSICS_.rotation[i];
    }
    for (int i = 0; i < 3; ++i) {
        EXTR_TRANS_[i] = DEPTH_TO_COLOR_EXTRINSICS_.translation[i];
    }

	// 提取深度转换比例
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
