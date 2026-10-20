#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

enum class TeamColor : std::uint8_t {
    Red = 0,
    Blue = 1,
};

enum class ArmorSize : std::uint8_t {
    Small = 0,
    Large = 1,
};

enum class TrackingState : std::uint16_t {
    // 数值直接对应历史通信字段 crc。
    NoTarget = 0,
    Stable = 1,
    Unstable = 2,
};

struct GimbalState {
    // 控制器快照：角度单位为弧度，prediction_bias_s 单位为秒。
    float yaw{0.0F};
    float pitch{0.0F};
    float roll{0.0F};
    float prediction_bias_s{0.0F};
};

struct CameraParameters {
    // 所有矩阵均为单通道 CV_64F。刚体变换为
    // p_gimbal = R_camera_to_gimbal * p_camera + t_camera_to_gimbal。
    bool calibrated{false};
    int image_width{0};
    int image_height{0};
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Mat rotation_camera_to_gimbal;
    cv::Mat translation_camera_to_gimbal;  // 3x1，单位为米，表示在云台坐标系中的平移。
    double mean_reprojection_error_px{-1.0};

    bool valid() const;
};

struct ArmorDetection {
    // 图像角点按顺时针排列：左上、右上、右下、左下。
    std::array<cv::Point2f, 4> corners{};
    ArmorSize size{ArmorSize::Small};
    int target_id{0};       // 0 保留给未知目标/无目标。
    float confidence{0.0F}; // 分类器置信度，范围为 [0, 1]。
    // 时间保持帧：本帧未检测到，但上一帧刚检测到，沿用上一帧框的拷贝，
    // 用于抹平「单帧漏检导致框闪灭」的抖动。held=true 表示这是保持帧而非新检测。
    bool held{false};
};

struct ArmorPose {
    // 平移向量单位为米。armor_yaw / armor_pitch 是云台坐标系中的偏航/俯仰角
    // （弧度，x 向右、y 向下、z 向前），reprojection_error 是单个观测的重投影误差（像素）。
    ArmorDetection detection;
    cv::Vec3d position_camera_m{0.0, 0.0, 0.0};
    cv::Vec3d position_gimbal_m{0.0, 0.0, 0.0};
    double armor_yaw{0.0};
    double armor_pitch{0.0};
    // 板表面法向在云台水平面内的朝向（弧度），供 EKF 判断目标是否自转。
    double armor_orientation_yaw{0.0};
    double reprojection_error{0.0};
};

struct PredictionResult {
    // EKF 在约定跟踪坐标系中的车体状态：位置/半径单位为米，
    // 速度单位为米每秒，角度单位为弧度，角速度单位为弧度每秒。
    TrackingState state{TrackingState::NoTarget};
    // EKF 是否仍持有有效车体估计（初始化后、未因长期无观测而重置）。它与 state 独立：
    // 快速旋转的哨兵板会周期性转到背面被遮挡（边缘 ~0.3s，检测跟不上），此时 state
    // 因「及时删编号」会落回 NoTarget，但 EKF 仍在纯预测、中心/yaw/半径/编号依旧有效，
    // 俯视图应继续用预测结果绘制、而不是整个消失。true = 可以画俯视图/继续预测。
    bool tracking{false};
    std::uint8_t target_id{0};
    std::uint8_t tracked_slot{0};  // 当前跟踪/瞄准的装甲板槽位（0..3，四板均布）
    // 出现顺序编号：槽位(0..3) -> 编号(1..4)，首次锁到该槽位时按先后顺序分配，
    // 转一圈后重新出现的板沿用旧编号（槽位是板的固定身份）。0 = 尚未观测到。
    std::array<std::uint8_t, 4> slot_ids{0, 0, 0, 0};
    // 槽位(0..3) -> 识别到的贴纸数字（板上印刷的 1..9）；0 = 尚未识别。与 slot_ids
    // 一样是板的固定身份，转一圈后沿用。
    std::array<std::uint8_t, 4> slot_digits{0, 0, 0, 0};
    std::uint8_t tracked_id{0};  // 当前跟踪槽位的出现顺序编号（= slot_ids[tracked_slot]）
    float center_x_m{0.0F};
    float center_y_m{0.0F};
    float center_velocity_x_m_s{0.0F};
    float center_velocity_y_m_s{0.0F};
    float body_yaw{0.0F};
    float body_yaw_velocity{0.0F};
    float radius_1_m{0.0F};
    float radius_2_m{0.0F};
    // 最终云台指令：距离单位为米，角度单位为弧度，角速度单位为弧度每秒。
    float distance_m{0.0F};
    float target_yaw{0.0F};
    float target_pitch{0.0F};
    float target_yaw_velocity{0.0F};
    float target_pitch_velocity{0.0F};
    bool fire_allowed{false};
};

// 读取 OpenCV YAML，将矩阵统一为 CV_64F，并拒绝占位数据或几何上无效的标定结果。
// 失败时会在 error 中写入错误信息。
bool loadCameraParameters(const std::string& path, CameraParameters& parameters,
                          std::string& error);
