#pragma once

/*
* 输入分辨率和帧率
* 目前只采用深度相机RealSense D435的颜色与深度通道分辨率相同的配置，以免去对齐的时间
* 经测试，彩色通道支持最高1920*1080@30FPS与960*540@60FPS，
* 深度通道支持最高1280*720@30FPS、848*480@60FPS
* 用INPUT_*命名以区别于COLOR_*与DEPTH_*，便于后续扩展其他分辨率
*/

// 目标输入水平分辨率
constexpr int INPUT_WIDTH = 848;
// 目标输入垂直分辨率
constexpr int INPUT_HEIGHT = 480;
// 目标输入帧率
constexpr int INPUT_FPS = 60;
// 目标输入总像素数
constexpr int INPUT_PIXEL_COUNT = INPUT_WIDTH * INPUT_HEIGHT;

// 相机最小有效深度值，过滤过近无效数据（单位：米）
constexpr float DEPTH_MIN = 0.3f;
// 相机最大有效深度值，过滤过远无效数据（单位：米）
constexpr float DEPTH_MAX = 3.0f;