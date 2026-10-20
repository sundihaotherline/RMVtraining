#include "MultiVehicleTracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// 最小接受门限（m）：新开或只见过一个点的车，外接盒对角线≈0，门限退化为该值。
// 静止车板抖动范围 ~0.05 m，取 0.20 给足余量；旋转车第一块板之后连续扫过的板
// 逐帧只移动 ~0.01 m，也不会被 0.20 误拒。
constexpr double kMinGateM = 0.20;

// 门限 = max(kMinGateM, kBoxScale * 外接盒对角线)。盒中心到板距离最多 ≈ 对角线/2，
// 取 1.0 倍对角线（> 对角线/2）保证一圈板全收；静止车盒小，门限仍被 kMinGateM 托底。
constexpr double kBoxScale = 1.0;

// 重新捕获半径（m）：被所有车门限拒绝的板，若离某已有车中心 < 此值，就近并入该车，
// 而不是开新车。用于「旋转/平移车换板 + 稀疏检测」下的板位置跳变（v1 实测换板跳
// ~0.5 m、跨检测间隙再跳 ~1 m）；本素材 v4 第二辆车首次入场离静止车 ~1.3 m > 0.9，
// 仍会被正确拆成新车。
constexpr double kReacquireM = 0.90;

// 滑动窗长度（帧）：166 帧 ≈ 1s @166fps。旋转车 ~2 圈/s，1s 窗覆盖整圈外接盒，
// 中心即旋转中心；静止车 1s 窗几乎全在同一块板，盒极小。也决定了对移动中心的滞后。
constexpr std::size_t kRecentWindow = 166;

// 未关联观测之间的聚类门限（m）。同车两块板最多相距 2r≈0.5 m，两辆不同车的板
// 相距 >0.8 m，取中间值 0.60 m：同一帧出现的同一新车的多块板被合并成一车，
// 两辆同时入场的新车仍被拆成两车。
constexpr double kNewVehicleMergeM = 0.60;

// 长期无有效结果才回收该车（秒）。本素材里静止车单板被另一车遮挡最长 ~2 s，
// 取 10 s 保证遮挡期间不误回收、板编号不丢（回收后重新出现会被当新车重新编号）。
constexpr double kVehicleForgetSeconds = 10.0;

double dist2d(double dx, double dz) {
    return std::sqrt(dx * dx + dz * dz);
}

// 由滑动窗算出外接盒中心 (cx,cz) 与对角线 diag。空窗返回 diag=0。
void boxStats(const std::deque<std::array<double, 2>>& recent,
              double& cx, double& cz, double& diag) {
    if (recent.empty()) {
        cx = cz = diag = 0.0;
        return;
    }
    double minx = std::numeric_limits<double>::infinity();
    double maxx = -std::numeric_limits<double>::infinity();
    double minz = std::numeric_limits<double>::infinity();
    double maxz = -std::numeric_limits<double>::infinity();
    for (const auto& p : recent) {
        minx = std::min(minx, p[0]);
        maxx = std::max(maxx, p[0]);
        minz = std::min(minz, p[1]);
        maxz = std::max(maxz, p[1]);
    }
    cx = 0.5 * (minx + maxx);
    cz = 0.5 * (minz + maxz);
    diag = std::sqrt((maxx - minx) * (maxx - minx) + (maxz - minz) * (maxz - minz));
}

}  // namespace

std::vector<PredictionResult> MultiVehicleTracker::step(
    const std::vector<ArmorPose>& observations, const GimbalState& gimbal,
    double timestamp_seconds) {
    const int n = static_cast<int>(observations.size());

    // 0. 先回收长期无板的车（回收很罕见，本素材不会触发；放在开头使本帧关联/结果与
    //    回收后的 vehicles_ 严格对齐）。
    vehicles_.erase(
        std::remove_if(vehicles_.begin(), vehicles_.end(),
                       [&](const Vehicle& v) {
                           return !v.recent_.empty() &&
                                  timestamp_seconds - v.last_seen_t >
                                      kVehicleForgetSeconds;
                       }),
        vehicles_.end());

    // 1. 每块观测 -> 最近车盒中心（距离 < 该车自适应门限才接受），否则 -1（待新开）。
    std::vector<int> vehicle_of(n, -1);
    for (int i = 0; i < n; ++i) {
        const cv::Vec3d& p = observations[i].position_gimbal_m;
        if (!std::isfinite(p[0]) || !std::isfinite(p[2])) {
            continue;
        }
        double best_d = std::numeric_limits<double>::infinity();
        int best_v = -1;
        for (std::size_t v = 0; v < vehicles_.size(); ++v) {
            if (vehicles_[v].recent_.empty()) {
                continue;  // 空窗车：尚未有板，不该抢观测（新车由后面的聚类步骤创建）
            }
            double cx, cz, diag;
            boxStats(vehicles_[v].recent_, cx, cz, diag);
            const double gate = std::max(kMinGateM, kBoxScale * diag);
            const double d = dist2d(p[0] - cx, p[2] - cz);
            if (d < gate && d < best_d) {
                best_d = d;
                best_v = static_cast<int>(v);
            }
        }
        vehicle_of[i] = best_v;
    }

    // 1.5 重新捕获：被所有车门限拒绝的板，若离某已有车中心在 kReacquireM 内，就近并入
    //     该车（旋转/平移车换板或稀疏检测下的位置跳变），而不是开新车——否则会碎片化。
    for (int i = 0; i < n; ++i) {
        if (vehicle_of[i] >= 0) {
            continue;
        }
        const cv::Vec3d& p = observations[i].position_gimbal_m;
        if (!std::isfinite(p[0]) || !std::isfinite(p[2])) {
            continue;
        }
        double best_d = kReacquireM;
        int best_v = -1;
        for (std::size_t v = 0; v < vehicles_.size(); ++v) {
            if (vehicles_[v].recent_.empty()) {
                continue;
            }
            double cx, cz, diag;
            boxStats(vehicles_[v].recent_, cx, cz, diag);
            const double d = dist2d(p[0] - cx, p[2] - cz);
            if (d < best_d) {
                best_d = d;
                best_v = static_cast<int>(v);
            }
        }
        vehicle_of[i] = best_v;
    }

    // 2. 仍未关联的观测按相互距离聚类成新车（贪心：够近并入已有群，否则开新群）。
    std::vector<int> new_group_of(n, -1);
    std::vector<std::vector<int>> new_groups;
    for (int i = 0; i < n; ++i) {
        if (vehicle_of[i] >= 0) {
            continue;
        }
        const cv::Vec3d& p = observations[i].position_gimbal_m;
        if (!std::isfinite(p[0]) || !std::isfinite(p[2])) {
            continue;
        }
        int g = -1;
        for (std::size_t gi = 0; gi < new_groups.size() && g < 0; ++gi) {
            for (int j : new_groups[gi]) {
                const cv::Vec3d& q = observations[j].position_gimbal_m;
                if (dist2d(p[0] - q[0], p[2] - q[2]) <= kNewVehicleMergeM) {
                    g = static_cast<int>(gi);
                    break;
                }
            }
        }
        if (g < 0) {
            g = static_cast<int>(new_groups.size());
            new_groups.emplace_back();
        }
        new_groups[g].push_back(i);
        new_group_of[i] = g;
    }

    // 3. 为每个新车群创建 tracker（编号源指向全局计数器）。
    const std::size_t old_n = vehicles_.size();
    for (std::size_t gi = 0; gi < new_groups.size(); ++gi) {
        vehicles_.emplace_back();
        vehicles_.back().tracker = std::make_unique<RotatingVehicleTracker>();
        vehicles_.back().tracker->setGlobalIdAllocator(&next_global_id_);
    }

    // 4. 逐车收集本帧观测并 step（每车都 step，即使本帧没有观测，也要推进其纯预测/
    //    超时重置状态机）。
    std::vector<PredictionResult> results;
    results.reserve(vehicles_.size());
    for (std::size_t v = 0; v < vehicles_.size(); ++v) {
        std::vector<ArmorPose> mine;
        for (int i = 0; i < n; ++i) {
            const bool belongs =
                v < old_n
                    ? (vehicle_of[i] == static_cast<int>(v))
                    : (new_group_of[i] == static_cast<int>(v - old_n));
            if (!belongs) {
                continue;
            }
            mine.push_back(observations[i]);
            const cv::Vec3d& p = observations[i].position_gimbal_m;
            if (std::isfinite(p[0]) && std::isfinite(p[2])) {
                vehicles_[v].recent_.push_back({p[0], p[2]});
                if (vehicles_[v].recent_.size() > kRecentWindow) {
                    vehicles_[v].recent_.pop_front();
                }
                vehicles_[v].last_seen_t = timestamp_seconds;
            }
        }

        results.push_back(
            vehicles_[v].tracker->step(mine, gimbal, timestamp_seconds));
    }

    return results;
}
