#pragma once

#include "Constant.hpp"
#include <cuda_runtime.h>
#include <librealsense2/rs.hpp>

class RealsenseCamera {
public:
	RealsenseCamera();
	~RealsenseCamera();

	// 丢弃前frames_to_discard帧数据，避免初始帧不稳定
	void warmup(int frames_to_discard = 20);

	// 获取一帧数据
	
	// 对齐深度到彩色


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

	// 彩色源输入水平焦距（单位：像素）
	float COLOR_FX_;
	// 彩色源输入垂直焦距（单位：像素）
	float COLOR_FY_;
	// 彩色源水平焦距倒数（避免除法运算，加速计算）
	float COLOR_FX_INV_;
	// 彩色源垂直焦距倒数（避免除法运算，加速计算）
	float COLOR_FY_INV_;
	// 彩色源输入主点X坐标（单位：像素）
	float COLOR_CX_;
	// 彩色源输入主点Y坐标（单位：像素）
	float COLOR_CY_;
	// 深度源输入水平焦距（单位：像素）
	float DEPTH_FX_;
	// 深度源输入垂直焦距（单位：像素）
	float DEPTH_FY_;
	// 深度源水平焦距倒数（避免除法运算，加速计算）
	float DEPTH_FX_INV_;
	// 深度源垂直焦距倒数（避免除法运算，加速计算）
	float DEPTH_FY_INV_;
	// 深度源输入主点X坐标（单位：像素）
	float DEPTH_CX_;
	// 深度源输入主点Y坐标（单位：像素）
	float DEPTH_CY_;

	// 深度转换比例
	float DEPTH_SCALE_;
	
	// 旋转矩阵（列主序，与 RealSense rs2_extrinsics 一致）
	float EXTR_ROT_[9];  // [r00, r10, r20, r01, r11, r21, r02, r12, r22] （列优先）
	// 平移向量（单位：米）
	float EXTR_TRANS_[3]; // [tx, ty, tz]

	// 当前帧集
	rs2::frameset frames_;
	// 当前深度帧数据指针
	const uint16_t* depth_frame_ptr = nullptr;
	// 当前彩色帧数据指针
	const uint8_t* color_frame_ptr = nullptr;
};