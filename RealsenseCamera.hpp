#pragma once

#include <librealsense2/rs.hpp>

class RealsenseCamera {
public:
	// 获取相机实例、内外参到gpu内存
	RealsenseCamera();
	~RealsenseCamera();

	// 丢弃前frames_to_discard帧数据，避免初始帧不稳定
	void warmup(int frames_to_discard = 20);

	// 获取原始帧指针
	void get_raw_frames(const uint8_t*& color, const uint16_t*& depth);
	// 获取相机颜色内参指针
	float* get_color_intrinsics();
	// 获取相机深度内参指针
	float* get_depth_intrinsics();
	// 获取相机深度转颜色的旋转外参指针
	float* get_extrinsics_rotation();
	// 获取相机深度转颜色的平移外参指针
	float* get_extrinsics_translation();
	// 获取深度转换比例
	float* get_depth_scale();

private:
	// RealSense管道
	rs2::pipeline PIPE_;
	// RealSense配置
	rs2::config CFG_;
	// RealSense管道配置文件
	rs2::pipeline_profile PROFILE_;

	// 相机外内参
	rs2_intrinsics COLOR_INTRINICS_;
	rs2_intrinsics DEPTH_INTRINICS_;
	rs2_extrinsics DEPTH_TO_COLOR_EXTRINSICS_;

	// 彩色源内参矩阵（fx, fy, cx, cy）
	float COLOR_INTR_[4];
	// 深度源内参矩阵（fx, fy, cx, cy）
	float DEPTH_INTR_[4];
	
	// 旋转矩阵（列主序，与 RealSense rs2_extrinsics 一致）
	float EXTR_ROT_[9];  // [r00, r10, r20, r01, r11, r21, r02, r12, r22] （列优先）
	// 平移向量（单位：米）
	float EXTR_TRANS_[3]; // [tx, ty, tz]

	// 深度转换比例
	float DEPTH_SCALE_;

	// 当前帧集
	rs2::frameset frames_;
	// 当前深度帧数据指针
	const uint16_t* depth_frame_ptr = nullptr;
	// 当前彩色帧数据指针
	const uint8_t* color_frame_ptr = nullptr;
};