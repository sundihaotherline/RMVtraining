#pragma once

#include "ArmorDetector.hpp"
#include "AutoAimTypes.hpp"

#include <vector>

#include <opencv2/core.hpp>

// 任务 1：管理相机 SDK，返回一张非空的 CV_8UC3 BGR 图像。
// 超时或断开时返回 false；不要在此函数中终止进程。
bool get_pic(cv::Mat& pic);

// 关闭相机并释放 SDK 资源。操作线程结束前调用，避免 SDK 在进程退出阶段析构崩溃。
void close_camera();

// 离线视频源状态查询（配合 RMV_TEST_VIDEO 环境变量使用）：
//  - is_video_source：当前是否从视频文件取流；
//  - video_source_ended：视频是否已播完；
//  - get_pic_timestamp_sec：当前帧在视频时间轴上的时间戳（秒，逐帧单调递增）；
//  - get_video_fps：视频标称帧率。
// 这些接口让主循环在离线视频模式下用真实视频时间戳跑 EKF/显示 ω，而非壁钟时间。
bool is_video_source();
bool video_source_ended();
double get_pic_timestamp_sec();
double get_video_fps();

// 任务 2：检测敌方装甲板并识别其编号。每个结果必须包含四个按顺时针排列的
// 图像角点、装甲板尺寸、目标编号和置信度。
std::vector<ArmorDetection> armor_detect(const cv::Mat& pic, TeamColor enemy_color);

// 调试：返回检测到的灯条（不配对），供实时可视化反馈灯条级中间结果。
std::vector<ArmorDetector::LightBar> armor_detect_debug(const cv::Mat& pic,
                                                        TeamColor enemy_color);

// 调试：返回上一帧配对诊断（最宽灯条对的几何量与是否被拒），定位「框丢失」断点。
ArmorDetector::PairDiagnostic armor_pair_diagnostic();

// 任务 3：使用已标定的相机内参进行位姿解算，再将观测从相机坐标系变换到云台坐标系。
// 输出位置单位为米。
std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                  const CameraParameters& camera,
                                  const GimbalState& gimbal);

// 调试：对单块检测框复算 PnP，逐项记录每一道拒绝门槛的通过情况与数值，
// 用于定位“PnP 位姿解算失败 / 读数不显示”的确切断点。
struct PnPDiagnostic {
    bool camera_valid = false;
    bool corners_inside = false;
    bool solved = false;
    double depth_m = 0.0;
    bool depth_in_range = false;
    double reprojection_error_px = 0.0;
    bool reprojection_ok = false;
};
PnPDiagnostic armor_solve_debug(const ArmorDetection& detection,
                                const CameraParameters& camera);

// 任务 4：完成观测关联，更新旋转目标 EKF，并预测云台指令。
// timestamp_seconds 单调递增；prediction_bias_s 来自控制器。
// 多车支持：返回每辆车一个 PredictionResult（按车辆出现顺序），单辆车时长度 1。
std::vector<PredictionResult> ekf_predict(const std::vector<ArmorPose>& observations,
                                          const GimbalState& gimbal,
                                          double timestamp_seconds);
