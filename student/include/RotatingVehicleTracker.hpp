#pragma once

#include "AutoAimTypes.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

// 旋转车体 EKF 跟踪器（任务 4）。
//
// 跟踪坐标系：直接使用 gimbal 系（x 向右、y 向下、z 向前），地面平面 = x-z，
// 竖直方向 = y。EKF 主观测 = 板位置 position_gimbal_m（米，h 非线性）；板朝向
// armor_orientation_yaw（弧度）作为 yaw 的直接观测（独立量测），使静止时
// （单板位置对 yaw/w 不可观测）yaw 依然有锚、不会随机游走。w 不由 yaw 差分
// 注入，而是让 EKF 借位置序列 + 过程模型自行估计，避免派生量测的相关性污染。
// 整条链不再需要相机->云台的旋转外参（见文档「为什么旋转向量不是必须」）。
//
// 状态向量（13 维，与 armor/src/Planner.cpp 的 trackLine 同构，单位改为米）：
//   kCx,kVx   -> 车体旋转中心 x 及其速度（右）
//   kCz,kVz   -> 车体旋转中心 z 及其速度（前）
//   kY0..kY3  -> 四块装甲板的高度（下）
//   kR        -> 半径（四块板到旋转中心的距离，同一辆车只有一个半径）
//   kYaw,kYawRate -> 车体 yaw 与 yaw 角速度
//   剩余 kUnused/kUnused2 为保留位（前哨站模型扩展用）。
class RotatingVehicleTracker {
public:
    RotatingVehicleTracker();
    ~RotatingVehicleTracker();

    // 每帧调用一次：观测关联 -> 预测/更新 -> 预测到命中时刻 -> 返回协议结果。
    // timestamp_seconds 单调递增（main 传入取帧时刻，单位秒）。
    PredictionResult step(const std::vector<ArmorPose>& observations,
                          const GimbalState& gimbal, double timestamp_seconds);

    // 多车管理：把本跟踪器的出现顺序编号源从「本地 1..4」切到共享的全局计数器，
    // 使多辆车之间板编号不冲突、按全局出现先后统一递增。传 nullptr 则退回本地计数
    // （单车主循环或独立单测时仍自洽）。槽位(0..3) -> 全局编号写入 slot_ids 输出。
    void setGlobalIdAllocator(std::uint8_t* next_global_id) { next_global_id_ = next_global_id; }

private:
    static constexpr int kStateDim = 13;
    enum : int {
        kCx = 0, kVx = 1, kCz = 2, kVz = 3,
        kY0 = 4, kY1 = 5, kUnused = 6, kR = 7, kUnused2 = 8,
        kYaw = 9, kYawRate = 10, kY2 = 11, kY3 = 12,
    };

    // 模型（在 gimbal 系内定义）。
    Eigen::VectorXd processModel(const Eigen::VectorXd& x, double dt) const;
    Eigen::MatrixXd processJacobian(const Eigen::VectorXd& x, double dt) const;
    Eigen::VectorXd observationModel(const Eigen::VectorXd& x, int slot) const;
    Eigen::MatrixXd observationJacobian(const Eigen::VectorXd& x, int slot) const;
    Eigen::MatrixXd processNoise(double dt) const;
    Eigen::MatrixXd measurementNoise() const;
    // 位置观测噪声（3x3，gimbal 系）：按重投影误差与相机深度自适应缩放（见 .cpp）。
    Eigen::MatrixXd positionMeasurementNoise(double reproj_px, double depth_m) const;

    // 数字(1~9) -> 槽位(0..3)；未知返回 -1，走全局最近邻关联。
    int numberToSlot(int target_id) const;

    void initializeFromObservation(const ArmorPose& obs);
    double wrapToPi(double angle) const;

    static double slotPhase(int slot);
    static double slotRadius(const Eigen::VectorXd& x, int slot);
    static int slotHeightIndex(int slot);

    std::unique_ptr<class ExtendedKalmanFilter> ekf_;
    int active_slot_ = 0;   // 当前 update() 使用的槽位（h/jacobian_h 的 lambda 读它）
    double active_reproj_error_{0.0};  // 当前 position update 观测的重投影误差（像素），供自适应 R 读
    double active_depth_m_{0.0};       // 当前 position update 观测的相机深度（米），供自适应 R 读
    double dt_ = 0.0;       // 当前帧间隔，predict()/Q 的 lambda 读它

    bool initialized_ = false;
    double last_time_ = 0.0;
    int frames_since_observation_ = 0;
    int updates_ = 0;

    std::uint8_t tracked_target_id_ = 0;
    int tracked_slot_ = 0;

    // 槽位(0..3) -> 出现顺序编号：首次锁到该槽位时按先后顺序分配，转一圈后
    // 重新出现的板沿用旧编号（槽位是板的固定身份，与单帧丢失/遮挡无关）。0 = 尚未观测到。
    std::array<std::uint8_t, 4> slot_to_id_{{0, 0, 0, 0}};
    // 槽位(0..3) -> 识别到的贴纸数字（板上印刷的 1..9）；0 = 尚未识别。每次锁到该
    // 槽位且识别到非零数字都会更新，不永久绑定——转回来时重新识别比冻结首次结果可靠。
    std::array<std::uint8_t, 4> slot_to_digit_{{0, 0, 0, 0}};
    std::uint8_t next_appearance_id_{1};       // 本地计数器（next_global_id_ 为空时使用）
    std::uint8_t* next_global_id_{nullptr};    // 多车共享的全局计数器（非空时优先）

    // 实测 ω 的跨帧状态：对板朝向取负做相邻帧差分（作车体 yaw 代理，槽位相位 φ
    // 是常数、差分中抵消），±π 翻转/换板跳变用门限丢弃，轻 EMA 平滑。用于给 ω
    // 一个直接观测（见 step() 的 (e) 段）。
    double meas_omega_rad_s_ = 0.0;
    double prev_meas_yaw_ = 0.0;
    double prev_meas_time_ = 0.0;
    bool has_prev_meas_yaw_ = false;

    // 半径软观测的最近板位置窗（(x, z)，m）：单板位置对半径弱可观测，EKF 会把半径
    // 拉塌（0.2 -> 0.05），进而产生假速度/中心漂移/关联被拒。用最近 ~1s 板位置窗的
    // RMS 离质心距离直接观测半径（整圈时 RMS≈r），把半径锚回真值（见 step() 的 (f) 段）。
    std::deque<std::array<double, 3>> recent_pos_;   // (x, z, t)，按时间裁剪

    // 位置扫角 ω 软观测的最近「前板」位置窗（(x, z, t)，m）：板只在正对相机时被检测到，
    // 用前板位置绕旋转中心的方位角 θ_p = atan2(z-cz, x-cx) 的 wrap 增量窗口累计估计 ω
    // （切向速度 = r·ω，见 step() 的 (g) 段）。圆心由本窗做「圆拟合」还原（EKF 半径塌缩
    // 会把中心钉到板附近，错中心让扫速放大好几倍，故不直接用 EKF 中心）。与
    // recent_pos_ 分开：这里只收 observations.front() 的位置，且带时间戳。
    std::deque<std::array<double, 3>> recent_front_pos_;

    // 平移/静止模式：|ω| 低于阈值时 yaw 从单板位置几何不可观测（中心自由时任意
    // yaw 都能配出零残差），朝向观测又因板正对相机而病态，会驱动 yaw 漂移、进而
    // 让同一块板在槽位间跳变（id 更换）。此模式冻结 yaw/ω 的过程噪声与观测注入。
    bool translation_mode_ = false;
};
