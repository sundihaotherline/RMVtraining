#pragma once

#include "AutoAimTypes.hpp"
#include "RotatingVehicleTracker.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

// 多车管理：维护多个 RotatingVehicleTracker（一车一个），按「观测 -> 最近车中心」
// 的距离门限把当帧装甲板观测分配到各车；不与任何已有车关联的观测按相互距离聚类后
// 新开一辆车。所有车的装甲板出现顺序编号共用同一个全局计数器，使板编号全局唯一、
// 按出现先后统一递增（v1 单车时退化为 1..4，与旧行为一致）。
//
// 关联键用「最近 1s 板位置滑动窗的外接盒中心 + 对角线自适应门限」，而非 EKF 的旋转
// 中心：旋转车单板可见时中心/半径/yaw 三重退化，EKF 中心会沿半径方向漂移 ~0.2 m
// （已实测），用它做门限会把远侧板误拆成新车。而外接盒中心只依赖「板扫过的范围」，
// 对可见性不均（板在某侧停留更久）不敏感；门限随盒对角线缩放，静止车盒小、门限紧
// （只收自己那块几乎不动的板），旋转车盒大、门限松（收全一圈板），天然分开两者。
class MultiVehicleTracker {
public:
    // 一帧调用一次：观测分配 -> 各车 step -> 返回每辆车一个 PredictionResult。
    // 结果顺序 = 车辆出现顺序（先出现者在前），主循环据此绘制多个俯视图。
    std::vector<PredictionResult> step(const std::vector<ArmorPose>& observations,
                                       const GimbalState& gimbal,
                                       double timestamp_seconds);

private:
    struct Vehicle {
        std::unique_ptr<RotatingVehicleTracker> tracker;
        // 最近板位置滑动窗（(x, z)，单位 m）。用于算外接盒中心与对角线门限。
        std::deque<std::array<double, 2>> recent_;
        double last_seen_t{0.0};  // 最近一次本车分到板的时间戳（s），用于回收
    };

    std::vector<Vehicle> vehicles_;
    std::uint8_t next_global_id_{1};  // 所有车共享的板出现顺序编号
};
