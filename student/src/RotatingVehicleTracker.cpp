#include "RotatingVehicleTracker.hpp"

#include "BulletModel.hpp"
#include "KalmanFilter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// 调参区：这是本骨架最需要你按现场数据调整的量。
// ---------------------------------------------------------------------------
constexpr double kRadiusPriorM = 0.20;      // 半径先验（标准步兵约 0.2 m）
constexpr double kMinRadiusM = 0.03;        // 半径下界：半径物理上必须为正
constexpr double kMaxRadiusM = 0.60;        // 半径上界：防御发散（放宽，不硬锚定先验）
constexpr double kBulletSpeedMps = 23.5;    // 子弹初速（m/s），报告写清来源
constexpr double kMaxDtS = 0.2;             // 帧间隔上限，防止停顿后一步外推过远
constexpr int    kStableMinUpdates = 10;    // 连续成功更新多少次才判 Stable
constexpr double kLongLossSeconds = 1.5;    // 连续无观测超过此值 -> 重置 NoTarget
constexpr double kHoldLostSeconds = 0.2;    // 连续无观测超过此值 -> 输出先清空板编号（NoTarget），EKF 仍预测到 1.5s 才重置
// 旋转车体的装甲板会周期性转到背面被遮挡，两板之间的观测盲区可达 1s 量级；
// 0.5s 会在盲区里反复重置、把 ω 重新归零，导致角速度永远拟合不出来，故放宽到
// 1.5s，让 EKF 在盲区里做纯预测、保留已收敛的 ω。
constexpr double kOutlierGate = 9.0;        // 3 自由度卡方 95% 分位约 7.8，放宽到 9
constexpr double kMaxMeasuredLatencyS = 1.0; // 处理延迟上限：超过即认为时间基准不一致（离线视频），按 0 处理
constexpr double kFireYawErrorRad = 0.10;   // 开火 yaw 误差门限（约 5.7°）
constexpr double kFireMinDistanceM = 0.5;
constexpr double kFireMaxDistanceM = 6.0;

// 过程噪声 Q / 观测噪声 R 的对角量级（初值可跑，再按收敛速度调）。
constexpr double kQPosition = 1e-3;   // (m²)     位置
constexpr double kQVelocity = 1e-2;   // (m/s)²   速度
constexpr double kQHeight  = 1e-4;   // (m²)     高度
constexpr double kQRadius  = 1e-5;   // (m²)     半径（认为基本不变）
constexpr double kQYaw     = 1e-4;   // (rad²/s) yaw 过程噪声：θ 有位置+yaw 双观测锚定，噪声给小
constexpr double kQYawRate = 1e-2;   // (rad/s)²/s yaw 角速度过程噪声：现在有 ω 直接观测（朝向差分）锚定，可放大让 ω 快速跟上转速变化
constexpr double kRPosition = 1e-3;   // (m²)     观测位置噪声（约 3 cm 量级）
// 自适应位置观测噪声（见 positionMeasurementNoise()）：R 不再固定 kRPosition，而是按
// 观测质量缩放。重投影误差直接反映 4 角点拟合质量（角点→PnP 噪声的完整代理）；深度
// 平方放大像素噪声（σ∝z²）。坏角点/远目标的观测被放大噪声降权，不再带飞 EKF；
// 好观测噪声收紧、收敛更快。噪声作用在 gimbal 系三轴（旋转保持长度，量级不随外参变）。
constexpr double kReprojNoiseGain = 0.15;   // R ∝ (1 + gain·reproj_px²)：6px 时 ≈6.4x
constexpr double kRefDepthM = 2.0;          // 参考深度（m）：位置 R 按 (z/z_ref)² 缩放
constexpr double kDepthNoiseFloor = 0.25;   // 深度缩放下限（近处不再缩，避免过度自信）
constexpr double kDepthNoiseCeil = 16.0;    // 深度缩放上限（远处不再放大，避免彻底无视观测）
constexpr double kRYawObs = 2e-2;     // (rad²)   yaw 直接观测噪声基准（板朝向，侧对时约 8°）
constexpr double kYawSlantGain = 100.0;  // 板正对相机时 PnP 绕法向旋转病态，yaw/ω 观测噪声按 |cos(法向-视线)|² 放大
constexpr double kRWObs = 4.0;        // (rad/s)² ω 直接观测噪声基准（侧对、30fps 时朝向差分+EMA 后约 σ≈2 rad/s）
constexpr double kNominalDtS = 0.006;    // 基准帧间隔（~30fps），ω 观测噪声随帧率按 (dt0/dt)² 反比放大
constexpr double kMinMeasDtS = 0.005;   // ω 差分最小帧间隔：低于此说明时间戳抖动，d/dt 爆炸成假 ω，跳过
constexpr double kMaxMeasWInst = 20.0;  // 瞬时 ω 限幅（rad/s，约 1146°/s，~3 圈/秒），挡差分爆炸尖峰
constexpr double kMaxRotationDeltaRad = 1.0;  // yaw 观测投影到先验 branch 后残差仍超过此值视为翻转/离群，丢弃（1.0 rad 仍远小于 π，翻转仍被拒）

// 中心速度阻尼：哨兵/旋转车体原地旋转，中心本应固定。单板位置对「中心平移 vs 旋转」
// 弱可观测时，EKF 会把板的切向运动误判为中心速度（实测可飙到 ~5-20 m/s），中心随
// 之漂移数米、关联彻底失效。让速度按 exp(-kVelDamp · max(0, ω²-ω₀²) · dt) 衰减：ω
// 超过阈值 ω₀=2.5 rad/s 才阻尼（哨兵 ω≈4-6 rad/s 在阈值之上、~1s 拉到 0），平移车
// 的慢速转向（v1 ω≈1.7 rad/s）在阈值之下、完全不受影响（不误伤平移速度估计）。
constexpr double kVelDamp = 0.1;               // (1/(rad²·s)) 速度阻尼强度
constexpr double kVelDampOmega2Thresh = 6.25;  // ω² 阈值 = 2.5² rad/s

// 半径软观测参数（见 step() 的 (f) 段）。整圈板位置 RMS 离质心距离≈半径，用它对半径
// 做直接观测，防止 EKF 借单板弱可观测把半径拉塌。仅旋转车（|ω| 大）适用：平移车
// 位置窗是平移轨迹、RMS 与半径无关，不注入。
constexpr std::size_t kRadiusHintWindow = 166;      // 半径软观测滑动窗（帧）≈1s，覆盖整圈
constexpr std::size_t kRadiusHintMinSamples = 20;   // 至少这么多点才注入（避免初始弧段 RMS 低估）
constexpr double kRadiusHintOmegaRadS = 1.5;        // |ω| 超过此值才注入（~86°/s，区分旋转/平移）
constexpr double kRadiusHintMaxCenterVel = 0.5;     // 中心速度低于此值才视为「原地旋转」（哨兵），
                                                     // 高于则是平移（边转边平移，RMS 是平移轨迹、与半径无关）
constexpr double kRadiusHintNoiseM = 0.02;          // 半径软观测噪声（m），比位置观测紧，压过塌缩方向
constexpr double kRadiusPriorNoiseM = 0.06;         // 平移/静止车半径弱先验噪声（m）：先验 0.2 只是猜测，放宽不硬锁定

// 朝向窗 ω 软观测（见 step() 的 (g) 段）：板只在正对相机时被检测到（slant≈1），被检测
// 板的位置几乎恒在正前方、不绕中心扫角（实测位置角 std≈10°），朝向差分的 ω 观测又几乎
// 恒被 slant 加权降到 0，ω 只能靠位置→θ 间接耦合估计、噪声大且随转速周期性振荡。改用
// 「前板朝向 -armor_orientation_yaw 的 wrap 增量」窗口累计（逐帧 wrapToPi 增量，
// |增量|>门限判换板丢弃，总扫角/总时长 = ω）。仅旋转车注入。
constexpr std::size_t kOmegaWindow = 166;          // 前板朝向窗（帧）≈1s，覆盖整圈
constexpr std::size_t kOmegaMinSamples = 40;       // 至少这么多有效增量才注入
constexpr double kOmegaIncGateRad = 0.15;          // 相邻帧扫角增量门限（>8.6°判换板/离群/正对乱跳，丢弃）
                                                    // 0.5→0.15：0.5 会放过正对相机时十几度的小幅乱跳，
                                                    // 让 ω 被错误方向的增量带负；真实每帧增量仅 ~0.015 rad。
constexpr double kOmegaObsNoiseRadS = 1.0;         // 朝向窗 ω 观测噪声（rad/s），整圈累计后 σ≈1~2

// 平移/静止判定门限：|ω| 低于此值（约 23°/s）视为纯平移/静止。此时单板位置对
// yaw 几何不可观测（中心自由时任意 yaw 都能配出零残差），朝向观测又因板正对相机
// 病态（PnP 绕法向旋转病态，armor_orientation_yaw 在 ±π 附近剧烈抖动/翻转），继续
// 注入 yaw/ω 观测会让 yaw 漂移。yaw 一漂，同一块板就被判到相邻槽位、id 更换。
// 故平移时冻结 yaw/ω（过程噪声置 0 + 不注入朝向/ω 观测），让 yaw 保持初值恒定。
constexpr double kTranslationOmegaThresh = 0.4;

// 极小规模的全局最小代价匹配：为每个观测挑一个互不重复的槽位，使总成本最小。
// cost[i][s] 为观测 i 到槽位 s 的成本，不可用项填 +inf。返回每个观测选中的
// 槽位下标（-1 表示该观测无可用槽位、作为离群点丢弃）。规模 ≤ 4×4，直接穷举。
std::vector<int> optimalAssign(const std::vector<std::vector<double>>& cost) {
    const int n = static_cast<int>(cost.size());
    const int m = n > 0 ? static_cast<int>(cost[0].size()) : 0;
    std::vector<int> best(n, -1);
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<int> cur(n, -1);
    std::vector<bool> used(m, false);
    std::function<void(int, double)> search = [&](int i, double acc) {
        if (i == n) {
            if (acc < best_cost) {
                best_cost = acc;
                best = cur;
            }
            return;
        }
        std::vector<int> avail;
        for (int s = 0; s < m; ++s) {
            if (!used[s] && std::isfinite(cost[i][s])) avail.push_back(s);
        }
        if (avail.empty()) {  // 无可用槽位：离群点，丢弃
            cur[i] = -1;
            search(i + 1, acc);
            return;
        }
        for (int s : avail) {
            used[s] = true;
            cur[i] = s;
            search(i + 1, acc + cost[i][s]);
            used[s] = false;
            cur[i] = -1;
        }
    };
    search(0, 0.0);
    return best;
}

// 角度归一化到 (-π, π]。
double wrapAngleToPi(double angle) {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

// 从板朝向直接判槽位。板表面法向满足 armor_orientation_yaw = -(θ + φ + π/2)（mod 2π），
// 反解相位 φ 后取最接近的 90° 整数倍即槽位。法向的 ±π 二义性会把结果推到对面槽（+2），
// 该歧义由调用处的位置马氏距离门限消解：对面槽预测位置相差约 2r，必被拒，退回全局匹配。
int slotFromOrientation(double armor_orientation_yaw, double theta_prior) {
    const double phi = wrapAngleToPi(-armor_orientation_yaw - theta_prior - kPi / 2.0);
    int slot = static_cast<int>(std::lround(phi / (kPi / 2.0)));
    slot %= 4;
    if (slot < 0) slot += 4;
    return slot;
}

}  // namespace

// 槽位 s 相对车体 yaw 的固定相位角（4 板均布，间隔 90°）。
double RotatingVehicleTracker::slotPhase(int slot) {
    return static_cast<double>(slot) * (kPi / 2.0);
}

// 槽位 s 对应的半径：四块板到旋转中心距离相同，统一用单一半径 kR。
double RotatingVehicleTracker::slotRadius(const Eigen::VectorXd& x, int slot) {
    (void)slot;
    return x(kR);
}

// 槽位 s 对应的高度状态索引。
int RotatingVehicleTracker::slotHeightIndex(int slot) {
    return slot == 0 ? kY0 : slot == 1 ? kY1 : slot == 2 ? kY2 : kY3;
}

RotatingVehicleTracker::RotatingVehicleTracker() = default;
RotatingVehicleTracker::~RotatingVehicleTracker() = default;

double RotatingVehicleTracker::wrapToPi(double angle) const {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

// 过程模型 f：等速平移 + 等角速度旋转（线性；yaw 的 wrap 只在 f 里做，Jacobian 忽略）。
//   θ' = θ + ω·dt   （匀角速度：θ 由 ω 积分推进）
//   ω' = ω          （角速度假设不变）
Eigen::VectorXd RotatingVehicleTracker::processModel(const Eigen::VectorXd& x,
                                                     double dt) const {
    Eigen::VectorXd xp = x;
    const double w = x(kYawRate);
    const double w2 = w * w;
    const double over = w2 > kVelDampOmega2Thresh ? w2 - kVelDampOmega2Thresh : 0.0;
    const double damp = std::exp(-kVelDamp * over * dt);
    xp(kCx) += x(kVx) * dt;                       // cx' = cx + vx·dt
    xp(kCz) += x(kVz) * dt;                       // cz' = cz + vz·dt
    xp(kVx) *= damp;                              // 快速旋转时中心速度被拉向 0（哨兵中心固定）
    xp(kVz) *= damp;
    xp(kYaw) = wrapToPi(x(kYaw) + x(kYawRate) * dt);  // θ' = wrap(θ + ω·dt)
    // 高度/半径/角速度均假设不变。
    return xp;
}

Eigen::MatrixXd RotatingVehicleTracker::processJacobian(const Eigen::VectorXd& x,
                                                        double dt) const {
    const double w = x(kYawRate);
    const double w2 = w * w;
    const double over = w2 > kVelDampOmega2Thresh ? w2 - kVelDampOmega2Thresh : 0.0;
    const double damp = std::exp(-kVelDamp * over * dt);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    F(kCx, kVx) = dt;            // ∂cx'/∂vx = dt
    F(kCz, kVz) = dt;            // ∂cz'/∂vz = dt
    F(kVx, kVx) = damp;          // ∂vx'/∂vx = damp（速度阻尼）
    F(kVz, kVz) = damp;          // ∂vz'/∂vz = damp
    F(kYaw, kYawRate) = dt;      // ∂θ'/∂ω = dt（θ' = θ + ω·dt）
    // ∂vx'/∂ω = vx · ∂damp/∂ω = vx · (-kVelDamp·2ω·dt·damp)（仅在 ω² 超阈值时非零）
    const double dw = w2 > kVelDampOmega2Thresh ? -kVelDamp * 2.0 * w * dt * damp : 0.0;
    F(kVx, kYawRate) = x(kVx) * dw;
    F(kVz, kYawRate) = x(kVz) * dw;
    return F;
}

// 观测模型 h：槽位 s 的装甲板在 gimbal 系的位置 (x 右, y 下, z 前)。
// 板 i 的绝对相位 = θ + φ_i。参考实现（armor/src/Planner.cpp 同款）用「中心 - r·(cos,sin)」
// 约定，即板在 (θ+φ) 指向的相反一侧：
//   x_pred = cx - r·cos(θ+φ)   （减号：板位于 θ+φ 方向的相反一侧）
//   z_pred = cz - r·sin(θ+φ)
// 这个减号与 armor_orientation_yaw 的映射一致（见 step() 里 yaw 直接观测），别单独改正负。
Eigen::VectorXd RotatingVehicleTracker::observationModel(const Eigen::VectorXd& x,
                                                         int slot) const {
    const double r = slotRadius(x, slot);
    const double phase = slotPhase(slot);
    const double yaw = x(kYaw);

    Eigen::VectorXd z(3);
    z(0) = x(kCx) - r * std::cos(yaw + phase);   // x（右）
    z(1) = x(slotHeightIndex(slot));             // y（下）
    z(2) = x(kCz) - r * std::sin(yaw + phase);   // z（前）
    return z;
}

Eigen::MatrixXd RotatingVehicleTracker::observationJacobian(const Eigen::VectorXd& x,
                                                            int slot) const {
    const double r = slotRadius(x, slot);
    const double phase = slotPhase(slot);
    const double yaw = x(kYaw);
    const double c = std::cos(yaw + phase);
    const double s = std::sin(yaw + phase);
    const int rIdx = kR;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kStateDim);
    // 行 0 = x：∂x/∂cx=1，∂x/∂r=-cos(θ+φ)，∂x/∂θ=+r·sin(θ+φ)。
    H(0, kCx) = 1.0;
    H(0, rIdx) = -c;                       // ∂(cx - r·c)/∂r = -c
    H(0, kYaw) = r * s;                    // ∂(cx - r·cos(θ+φ))/∂θ = +r·sin(θ+φ)
    // 行 1 = y：∂y/∂y_slot = 1。
    H(1, slotHeightIndex(slot)) = 1.0;
    // 行 2 = z：∂z/∂cz=1，∂z/∂r=-sin(θ+φ)，∂z/∂θ=-r·cos(θ+φ)。
    H(2, kCz) = 1.0;
    H(2, rIdx) = -s;                       // ∂(cz - r·s)/∂r = -s
    H(2, kYaw) = -r * c;                   // ∂(cz - r·sin(θ+φ))/∂θ = -r·cos(θ+φ)
    return H;
}

Eigen::MatrixXd RotatingVehicleTracker::processNoise(double dt) const {
    Eigen::VectorXd q = Eigen::VectorXd::Zero(kStateDim);
    q(kCx) = q(kCz) = kQPosition * dt;
    q(kVx) = q(kVz) = kQVelocity * dt;
    q(kY0) = q(kY1) = q(kY2) = q(kY3) = kQHeight * dt;
    q(kR) = kQRadius * dt;
    q(kYaw) = kQYaw * dt;
    q(kYawRate) = kQYawRate * dt;
    if (translation_mode_) {
        // 平移/静止：单板位置对 yaw/ω 几何不可观测，朝向又病态。冻结它们的过程噪声，
        // 防止 ω 随机游走 → 积分 → yaw 漂移 → 槽位跳变 → 同一块板换 id。
        q(kYaw) = 0.0;
        q(kYawRate) = 0.0;
    }
    return q.asDiagonal();
}

Eigen::MatrixXd RotatingVehicleTracker::measurementNoise() const {
    // position update 的观测噪声：读当前观测的 active_* 成员（step() 更新前设置）。
    return positionMeasurementNoise(active_reproj_error_, active_depth_m_);
}

Eigen::MatrixXd RotatingVehicleTracker::positionMeasurementNoise(double reproj_px,
                                                                double depth_m) const {
    // 位置观测噪声自适应缩放：重投影误差直接反映 4 角点拟合质量（角点→PnP 噪声的
    // 完整代理），深度平方放大像素噪声（近大远小 σ∝z²）。坏角点/远目标的观测被放大
    // 噪声降权，不再带飞 EKF；好观测噪声收紧、收敛更快。
    const double reproj_scale = 1.0 + kReprojNoiseGain * reproj_px * reproj_px;
    const double depth_scale =
        depth_m > 0.0 ? std::clamp((depth_m * depth_m) / (kRefDepthM * kRefDepthM),
                                   kDepthNoiseFloor, kDepthNoiseCeil)
                      : 1.0;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, 3);
    R(0, 0) = R(1, 1) = R(2, 2) = kRPosition * reproj_scale * depth_scale;
    return R;
}

// 数字 -> 槽位映射。TODO(student)：按自己机器人编号规则填写；
// 未知/无编号返回 -1，走最近邻关联。
int RotatingVehicleTracker::numberToSlot(int target_id) const {
    (void)target_id;
    return -1;
}

void RotatingVehicleTracker::initializeFromObservation(const ArmorPose& obs) {
    const cv::Vec3d& p = obs.position_gimbal_m;
    const int slot = numberToSlot(obs.detection.target_id);
    tracked_slot_ = slot >= 0 ? slot : 0;
    tracked_target_id_ = obs.detection.target_id;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateDim);
    const double phase = slotPhase(tracked_slot_);
    // 首帧用板朝向直接定 yaw（法向观测有 ±π 二义性，此处取主值作为 branch 初值，
    // 旋转起来后由位置序列校正），并按一致的中心反推让首帧近似零残差。
    const double yaw0 = wrapToPi(-obs.armor_orientation_yaw - phase - kPi / 2.0);
    x0(kCx) = p[0] + kRadiusPriorM * std::cos(yaw0 + phase);
    x0(kCz) = p[2] + kRadiusPriorM * std::sin(yaw0 + phase);
    x0(kY0) = x0(kY1) = x0(kY2) = x0(kY3) = p[1];
    x0(kR) = kRadiusPriorM;
    x0(kYaw) = yaw0;

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    P0(kCx, kCx) = P0(kCz, kCz) = 0.5 * 0.5;      // 中心不确定度高
    P0(kVx, kVx) = P0(kVz, kVz) = 2.0 * 2.0;
    P0(kYaw, kYaw) = 1.0 * 1.0;
    P0(kYawRate, kYawRate) = 2.0 * 2.0;
    P0(kR, kR) = 0.1 * 0.1;

    ekf_ = std::make_unique<ExtendedKalmanFilter>(
        [this](const Eigen::VectorXd& x) { return processModel(x, dt_); },
        [this](const Eigen::VectorXd& x) { return observationModel(x, active_slot_); },
        [this](const Eigen::VectorXd& x) { return processJacobian(x, dt_); },
        [this](const Eigen::VectorXd& x) { return observationJacobian(x, active_slot_); },
        [this]() { return processNoise(dt_); },
        [this](const Eigen::VectorXd&) { return measurementNoise(); },
        [](const Eigen::VectorXd& residual) { return residual; },  // 位置观测，残差线性
        P0, x0);

    initialized_ = true;
    frames_since_observation_ = 0;
    updates_ = 0;
    recent_pos_.clear();  // 新生命周期清空半径软观测窗，避免上一世的陈旧位置污染
    recent_front_pos_.clear();  // 同理清空位置窗 ω 观测窗
}

PredictionResult RotatingVehicleTracker::step(const std::vector<ArmorPose>& observations,
                                              const GimbalState& gimbal,
                                              double timestamp_seconds) {
    PredictionResult out{};
    const bool have_obs = !observations.empty();

    // 长时间无观测 -> 重置。
    if (initialized_ && !have_obs && last_time_ > 0.0 &&
        timestamp_seconds - last_time_ > kLongLossSeconds) {
        initialized_ = false;
        ekf_.reset();
        return out;  // NoTarget
    }

    // 首帧观测 -> 初始化。
    if (!initialized_) {
        if (!have_obs) {
            return out;  // 无观测无法初始化
        }
        initializeFromObservation(observations.front());
        last_time_ = timestamp_seconds;
    }

    // 帧间隔。
    double dt = last_time_ > 0.0 ? timestamp_seconds - last_time_ : 0.0;
    if (!(dt > 0.0) || dt > kMaxDtS) {
        dt = dt > kMaxDtS ? kMaxDtS : 0.0;
    }
    last_time_ = timestamp_seconds;
    dt_ = dt;

    // 平移/静止判断（用上一帧后验 ω，先于 predict 确定，供 processNoise 的 Q 使用）。
    translation_mode_ =
        std::fabs(ekf_->get_X()(kYawRate)) < kTranslationOmegaThresh;

    // 1. 预测到当前时刻。
    ekf_->predict();

    // 2. 全局关联 -> 门限 -> 串行更新（位置观测 + yaw 直接观测）。
    //    - target_id 命中 -> 直接查表定槽（按重投影误差升序抢占，冲突者退级到全局匹配）。
    //    - 其余观测 × 剩余槽位做最小代价匹配（马氏距离，穷举，规模 ≤ 4×4）。
    //    - 马氏距离超门限 kOutlierGate 的观测视为离群点丢弃。
    int used = 0;
    if (have_obs) {
        const Eigen::VectorXd& x_pri = ekf_->get_nochange_X_pri();
        const Eigen::MatrixXd& P_pri = ekf_->get_P_pri();
        // 观测对某槽位的马氏距离平方（基于同一先验，保证全局关联的一致性）。
        // 噪声按该观测自身的重投影误差/深度自适应缩放：坏角点的位置残差会被更大的 R
        // 吸收、不至于被误判成离群点拒收；真正的误关联（槽位错配，残差≈2r）远大于
        // 噪声量级，仍被门限拒掉。
        auto maha2 = [&](const ArmorPose& obs, int slot) -> double {
            const Eigen::VectorXd zp = observationModel(x_pri, slot);
            Eigen::VectorXd z(3);
            z(0) = obs.position_gimbal_m[0];
            z(1) = obs.position_gimbal_m[1];
            z(2) = obs.position_gimbal_m[2];
            const Eigen::VectorXd nu = z - zp;
            const Eigen::MatrixXd H = observationJacobian(x_pri, slot);
            const Eigen::MatrixXd R = positionMeasurementNoise(obs.reprojection_error,
                                                               obs.position_camera_m[2]);
            const Eigen::MatrixXd S = H * P_pri * H.transpose() + R;
            return nu.transpose() * S.inverse() * nu;
        };

        // (a) 收集有限位置的有效观测。
        // 槽位优先判定不再靠贴纸编号（四面贴纸相同时 numberToSlot 恒 -1），改用板朝向
        // 直接反解槽位相位（slotFromOrientation）。朝向 ±π 翻转会把槽位推到对面，
        // 由下面 (b) 步的位置马氏距离门限消解：对面槽预测位置相差约 2r，必被拒，
        // 从而退回 (c) 全局匹配。
        std::vector<const ArmorPose*> cands;
        std::vector<int> prefer;
        for (const ArmorPose& obs : observations) {
            const cv::Vec3d& p = obs.position_gimbal_m;
            if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
                cands.push_back(&obs);
                prefer.push_back(std::isfinite(obs.armor_orientation_yaw)
                                     ? slotFromOrientation(obs.armor_orientation_yaw,
                                                           x_pri(kYaw))
                                     : -1);
            }
        }
        const int n = static_cast<int>(cands.size());
        std::vector<int> slotOf(n, -1);
        std::vector<bool> slotUsed(4, false);

        // (b) target_id 直查表：按重投影误差升序抢占，越干净的观测越先锁定自己的槽。
        {
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return cands[a]->reprojection_error < cands[b]->reprojection_error;
            });
            for (int i : order) {
                const int s = prefer[i];
                if (s < 0 || s >= 4 || slotUsed[s]) continue;
                if (maha2(*cands[i], s) <= kOutlierGate) {
                    slotOf[i] = s;
                    slotUsed[s] = true;
                }
            }
        }

        // (c) 剩余观测 × 剩余槽位的全局最小代价匹配。
        std::vector<int> remObs, remSlot;
        for (int i = 0; i < n; ++i) if (slotOf[i] < 0) remObs.push_back(i);
        for (int s = 0; s < 4; ++s) if (!slotUsed[s]) remSlot.push_back(s);
        if (!remObs.empty() && !remSlot.empty()) {
            std::vector<std::vector<double>> cost(
                remObs.size(),
                std::vector<double>(remSlot.size(), std::numeric_limits<double>::infinity()));
            for (std::size_t i = 0; i < remObs.size(); ++i) {
                for (std::size_t s = 0; s < remSlot.size(); ++s) {
                    const double d2 = maha2(*cands[remObs[i]], remSlot[s]);
                    if (d2 <= kOutlierGate) cost[i][s] = d2;
                }
            }
            const std::vector<int> assign = optimalAssign(cost);
            for (std::size_t i = 0; i < assign.size(); ++i) {
                if (assign[i] >= 0) slotOf[remObs[i]] = remSlot[assign[i]];
            }
        }

        // 出现顺序编号：首次锁到某槽位时按先后顺序分配。转一圈后该槽位重新出现
        // 沿用旧编号（槽位是板的固定身份，与单帧丢失/遮挡无关）。槽位分配经马氏门限，
        // 只有真正锁上的槽位才计为「首次出现」。多车模式下编号来自共享全局计数器，
        // 使不同车的板按全局出现先后统一编号、互不重复。
        for (int i = 0; i < n; ++i) {
            const int s = slotOf[i];
            if (s < 0) continue;
            if (slot_to_id_[s] == 0) {
                slot_to_id_[s] = next_global_id_ ? (*next_global_id_)++
                                                 : next_appearance_id_++;
            }
            // 记录该板的贴纸数字：首次识别到非零数字即存下（槽位是板的固定身份，
            // 转一圈后沿用，与单帧丢失/遮挡无关）。
            const std::uint8_t digit =
                static_cast<std::uint8_t>(cands[i]->detection.target_id);
            if (digit > 0 && slot_to_digit_[s] == 0) {
                slot_to_digit_[s] = digit;
            }
        }

        // (d) 按分配结果串行更新：位置观测 + yaw 直接观测（不再注入 w 观测）。
        for (int i = 0; i < n; ++i) {
            const int slot = slotOf[i];
            if (slot < 0) continue;
            const ArmorPose& obs = *cands[i];
            const cv::Vec3d& p = obs.position_gimbal_m;

            Eigen::VectorXd z(3);
            z(0) = p[0];
            z(1) = p[1];
            z(2) = p[2];
            active_slot_ = slot;
            // 自适应 R：把本观测的重投影误差/相机深度写进成员，measurementNoise() 据此
            // 缩放位置观测噪声（update(z) 内部经 update_R 回调读取）。
            active_reproj_error_ = obs.reprojection_error;
            active_depth_m_ = obs.position_camera_m[2];
            ekf_->update(z);
            ekf_->usePostAsPri();  // 多观测时按后验继续下一个更新
            ++used;
            ++updates_;
            tracked_target_id_ = obs.detection.target_id;
            tracked_slot_ = slot;

            // 板朝向（armor_orientation_yaw）直接编码车体 yaw：
            //   yaw_obs = -armor_orientation_yaw - φ - π/2
            // 用它做 yaw 的直接观测（独立量测，静止时锚定 yaw，使其不随机游走）。
            // 法向观测有 ±π 二义性（正对时板朝向病态会翻 π），不能拿「上一帧原始
            // 观测」当基准逐帧比较（会锁死在错误 branch）。改为投影到先验 yaw 所在
            // branch：残差 = wrap(yaw_obs - yaw_prior)。正确 branch 的残差≈噪声，
            // 翻 π 的观测残差≈π，据此丢弃后者。
            const double angle = obs.armor_orientation_yaw;
            const double yaw_obs = wrapToPi(-angle - slotPhase(slot) - kPi / 2.0);
            const double yaw_prior = ekf_->get_nochange_X_pri()(kYaw);
            const double delta_yaw = wrapToPi(yaw_obs - yaw_prior);
            // 平移/静止时朝向病态，跳过 yaw 直接观测：板正对相机时 PnP 绕法向旋转
            // 病态，armor_orientation_yaw 在 ±π 附近抖动，注入只会带偏 yaw 使其漂移。
            if (!translation_mode_ && std::fabs(delta_yaw) < kMaxRotationDeltaRad) {
                // yaw 直接观测：update(z,H,R) 的残差 (z - H·x_pri) 不做 wrap，角度
                // 跨越 ±π 会被当成 ~2π 巨差。把 z 设成「先验 + wrap(观测-先验)」，
                // 让内部残差正好等于 wrap 后的角度差，处理 179°→-179° 跨越。
                //
                // 板正对相机时（法向≈视线反向）PnP 绕法向旋转病态、armor_orientation_yaw
                // 噪声巨大；侧对时法向清晰。slant=|cos(法向-视线)|：0=侧对(可信)、
                // 1=正对(病态)。用它放大 R_yaw，正对时自动降权 yaw 观测、改由位置观测
                // 锚定 θ，侧对时 yaw 观测拿回主导权。
                const double slant = std::abs(
                    std::cos(wrapToPi(obs.armor_orientation_yaw - obs.armor_yaw)));
                Eigen::MatrixXd H_yaw = Eigen::MatrixXd::Zero(1, kStateDim);
                H_yaw(0, kYaw) = 1.0;
                Eigen::MatrixXd R_yaw = Eigen::MatrixXd::Zero(1, 1);
                R_yaw(0, 0) = kRYawObs * (1.0 + kYawSlantGain * slant * slant);
                Eigen::VectorXd z_yaw(1);
                z_yaw(0) = yaw_prior + delta_yaw;
                ekf_->update(z_yaw, H_yaw, R_yaw);
                ekf_->usePostAsPri();
            }
        }

        // (e) ω 直接观测：板朝向相邻帧差分给出实测 ω，让 EKF 能跟上快速旋转、静止时
        // 把 ω 拉回 0。位置/yaw 观测都只测 θ，ω 只靠 θ↔ω 交叉协方差间接推——收敛慢、
        // 静止有偏（同一硬币两面）。差分放大噪声，R_w 取大压低增益，只用来「引导 ω
        // 方向」而非精确测 ω；与 yaw 观测同源相关，故放最后更新。这一观测与关联无关：
        // 即使位置/yaw 因门限被拒（转速太快、θ 预测滞后），ω 观测仍按原始朝向序列
        // 把 ω 拉回真值，从而打破「ω 卡住 → θ 滞后 → 观测被拒 → ω 更卡」的死锁。
        {
            const double yaw_proxy = -observations.front().armor_orientation_yaw;
            if (std::isfinite(yaw_proxy)) {
                if (has_prev_meas_yaw_ && timestamp_seconds > prev_meas_time_) {
                    const double dt_m = timestamp_seconds - prev_meas_time_;
                    const double d = wrapToPi(yaw_proxy - prev_meas_yaw_);
                    // 门限 kOmegaIncGateRad（0.15 rad）：丢弃 ±π 翻转/换板跳变（π/2）与
                    // 正对相机时朝向的小幅乱跳（十几度会掺进错误方向、把 ω 带负）；
                    // dt_m 下限 kMinMeasDtS 挡时间戳抖动的差分爆炸，上限 kMaxDtS 丢弃
                    // 遮挡盲区后的陈旧差分；瞬时 ω 再限幅兜底。
                    // 平移/静止时朝向病态，ω 差分观测同样不可靠，跳过注入（prev 时序仍滚动，
                    // 转起来后立即可差分）。
                    if (!translation_mode_ && dt_m >= kMinMeasDtS &&
                        dt_m <= kMaxDtS && std::fabs(d) < kOmegaIncGateRad) {
                        const double w_inst =
                            std::clamp(d / dt_m, -kMaxMeasWInst, kMaxMeasWInst);
                        meas_omega_rad_s_ = 0.3 * w_inst + 0.7 * meas_omega_rad_s_;
                        // 朝向差分噪声 ∝ σ_yaw/dt：板更正对（σ_yaw 大）或帧率更高
                        // （dt 小）时都放大，让 R_w 始终匹配真实差分噪声。
                        const double slant_w = std::abs(std::cos(wrapToPi(
                            observations.front().armor_orientation_yaw -
                            observations.front().armor_yaw)));
                        const double dt_ratio = kNominalDtS / dt_m;
                        Eigen::MatrixXd H_w = Eigen::MatrixXd::Zero(1, kStateDim);
                        H_w(0, kYawRate) = 1.0;
                        Eigen::MatrixXd R_w = Eigen::MatrixXd::Zero(1, 1);
                        R_w(0, 0) = kRWObs * (1.0 + kYawSlantGain * slant_w * slant_w) *
                                    dt_ratio * dt_ratio;
                        Eigen::VectorXd z_w(1);
                        z_w(0) = meas_omega_rad_s_;
                        ekf_->update(z_w, H_w, R_w);
                        ekf_->usePostAsPri();
                    }
                }
                prev_meas_yaw_ = yaw_proxy;
                prev_meas_time_ = timestamp_seconds;
                has_prev_meas_yaw_ = true;
            }
        }

        // (f) 半径软观测：单板位置对半径弱可观测（每帧板切向只移 ~1.5cm，比 3cm 噪声还
        // 小），EKF 会把半径拉塌（0.2 -> 0.05）。塌掉的半径对旋转车致命：板的切向速度
        // r·ω 被低估，预测板位置偏离真值 ~0.17m、被马氏门限拒收 -> 卡死 -> 1.5s 重置 ->
        // 再塌。但对平移车无害（中心/半径/yaw 退化，塌半径的同时中心会相应调整、板位置
        // 仍跟踪得上），强行锚回先验反而破坏其自洽解。因此仅「原地旋转」（|ω| 大且中心
        // 速度小，即哨兵）才注入半径软观测：用最近 ~1s 板位置窗的 RMS 离质心距离（整圈
        // RMS≈r）直接量测真实半径，随车体实际尺寸自适应；样本不足时退回先验 0.2。
        {
            for (const ArmorPose* c : cands) {
                const cv::Vec3d& p = c->position_gimbal_m;
                if (std::isfinite(p[0]) && std::isfinite(p[2])) {
                    recent_pos_.push_back({p[0], p[2]});
                }
            }
            while (recent_pos_.size() > kRadiusHintWindow) {
                recent_pos_.pop_front();
            }

            const double w_now = ekf_->get_X()(kYawRate);
            const double v_now = std::hypot(ekf_->get_X()(kVx), ekf_->get_X()(kVz));
            if (std::fabs(w_now) > kRadiusHintOmegaRadS &&
                v_now < kRadiusHintMaxCenterVel) {
                double r_target = kRadiusPriorM;   // 样本不足时退回先验
                if (recent_pos_.size() >= kRadiusHintMinSamples) {
                    double mx = 0.0, mz = 0.0;
                    for (const auto& p : recent_pos_) {
                        mx += p[0];
                        mz += p[1];
                    }
                    mx /= static_cast<double>(recent_pos_.size());
                    mz /= static_cast<double>(recent_pos_.size());
                    double s = 0.0;
                    for (const auto& p : recent_pos_) {
                        const double dx = p[0] - mx, dz = p[1] - mz;
                        s += dx * dx + dz * dz;
                    }
                    const double r_meas =
                        std::sqrt(s / static_cast<double>(recent_pos_.size()));
                    r_target = std::clamp(r_meas, kMinRadiusM, kMaxRadiusM);
                }

                // 软观测把单一半径 r 拉向 r_target。
                Eigen::MatrixXd H_r = Eigen::MatrixXd::Zero(1, kStateDim);
                H_r(0, kR) = 1.0;
                Eigen::MatrixXd R_r = Eigen::MatrixXd::Zero(1, 1);
                R_r(0, 0) = kRadiusHintNoiseM * kRadiusHintNoiseM;
                Eigen::VectorXd z_r(1);
                z_r(0) = r_target;
                ekf_->update(z_r, H_r, R_r);
                ekf_->usePostAsPri();
            } else {
                // 平移/静止车：半径从位置观测弱可观测，EKF 会把它拉塌到接近 0，使 yaw
                // 从位置不可观测（∂位置/∂θ ∝ r），进而靠噪声巨大的朝向差分把 ω 带飞
                // （v1 单半径实测塌到 0.05 → ω 误估 -2.3 rad/s → 锁丢失）。用弱先验锚回
                // 0.2 防塌：噪声放宽（kRadiusPriorNoiseM），只挡塌缩方向、不硬锁定中心/半径
                // /yaw 的退化自洽解。
                Eigen::MatrixXd H_prior = Eigen::MatrixXd::Zero(1, kStateDim);
                H_prior(0, kR) = 1.0;
                Eigen::MatrixXd R_prior = Eigen::MatrixXd::Zero(1, 1);
                R_prior(0, 0) = kRadiusPriorNoiseM * kRadiusPriorNoiseM;
                Eigen::VectorXd z_prior(1);
                z_prior(0) = kRadiusPriorM;
                ekf_->update(z_prior, H_prior, R_prior);
                ekf_->usePostAsPri();
            }
        }

#if 1  // 【已修复】启用 (g) ω 直接观测：改用「前板朝向扫角」而非位置扫角。
        // (g) 朝向窗 ω 软观测：板只在正对相机时被检测到（slant≈1），被检测板的位置几乎
        // 恒在正前方（位置角 std≈10°、不绕中心扫角），真正单调扫的是板的朝向
        // armor_orientation_yaw（可见弧内扫 ~105°）。故用朝向的 wrap 增量窗口累计：
        // 逐帧 wrapToPi 增量，|增量|>门限判换板/离群丢弃，总扫角/总时长即干净的 ω。
        // 不再用 translation_mode_ 门控：它会造死锁——初始化 ω=0 → |ω|<0.4 → translation_mode_=true
        // → (g) 被跳过 → ω 永远 0，旋转车永远转不起来（正是「EKF ω 卡 0、实测 ω 读 290」）。
        // w_fit 本身自门：静止/平移车朝向不扫、w_fit≈0（正确注入 0），旋转车朝向扫、
        // w_fit≈真 ω（把 ω 拉起）。平移车边平移时朝向会随视线方位缓慢变化、可能注入小假 ω，
        // 属已知代价，后续用 |V| 或扫角幅值门限单独处理。
        {
            if (!observations.empty()) {
                // -armor_orientation_yaw 作车体 yaw 代理（与 (e) 段 yaw_proxy 同约定）：
                // armor_orientation_yaw = -(θ+φ+π/2)，取负后按 +ω 方向单调扫。
                const double ori = -observations.front().armor_orientation_yaw;
                if (std::isfinite(ori)) {
                    recent_front_pos_.push_back({ori, timestamp_seconds, 0.0});
                }
            }
            while (recent_front_pos_.size() > kOmegaWindow) {
                recent_front_pos_.pop_front();
            }

            // 用有效增量的瞬时 ω（dth/dt）中位数替代 total_ang/total_dt 均值：均值对
            // 正对相机时朝向的小幅乱跳敏感（乱跳增量符号会反、幅度达十几度，掺进和里
            // 把 ω 拉负）；中位数由多数同号增量决定，天然鲁棒，配合收紧后的门限双保险。
            std::vector<double> rates;
            rates.reserve(recent_front_pos_.size());
            for (std::size_t i = 1; i < recent_front_pos_.size(); ++i) {
                const auto& a = recent_front_pos_[i - 1];
                const auto& b = recent_front_pos_[i];
                const double dt = b[1] - a[1];
                if (!(dt > 0.0) || dt > kMaxDtS) continue;       // 时间戳倒退/停顿
                const double dth = wrapToPi(b[0] - a[0]);
                if (std::fabs(dth) > kOmegaIncGateRad) continue;  // 换板/离群/正对乱跳
                rates.push_back(dth / dt);
            }
            if (rates.size() >= kOmegaMinSamples) {
                std::nth_element(rates.begin(), rates.begin() + rates.size() / 2,
                                 rates.end());
                const double w_fit = rates[rates.size() / 2];
                Eigen::MatrixXd H_w = Eigen::MatrixXd::Zero(1, kStateDim);
                H_w(0, kYawRate) = 1.0;
                Eigen::MatrixXd R_w = Eigen::MatrixXd::Zero(1, 1);
                R_w(0, 0) = kOmegaObsNoiseRadS * kOmegaObsNoiseRadS;
                Eigen::VectorXd z_w(1);
                z_w(0) = w_fit;
                ekf_->update(z_w, H_w, R_w);
                ekf_->usePostAsPri();
            }
        }
#endif  // (g) ω 直接观测结束（朝向窗扫角软观测）

    }

    // 3. 丢失计数。
    frames_since_observation_ = (used > 0) ? 0 : frames_since_observation_ + 1;
    if (frames_since_observation_ * dt > kLongLossSeconds) {
        initialized_ = false;
        ekf_.reset();
        return out;  // NoTarget
    }

    // yaw 规整回 [-π, π]：过程/观测反复推进可能让 θ 累积到几圈之外，规整后输出与
    // 下一帧先验都在主值区间，避免角度残差再出现 2π 跨越。
    ekf_->get_X()(kYaw) = wrapToPi(ekf_->get_X()(kYaw));

    // 半径物理约束：观测/关联偶发会把半径拉负（与 yaw 的 ±π 分支耦合），负半径会
    // 进一步把中心/速度带飞。夹回正区间打破这个正反馈；值域放宽，不硬性锚定先验。
    ekf_->get_X()(kR) = std::clamp(ekf_->get_X()(kR), kMinRadiusM, kMaxRadiusM);

    const Eigen::VectorXd x = ekf_->get_X();  // 后验（或纯预测）状态
    const bool stable = frames_since_observation_ == 0 && updates_ >= kStableMinUpdates;

    // EKF 仍持有有效估计（走到这里说明 initialized_ 为真且未触发 1.5s 重置），
    // 即使当前板暂时没检测到（state 将回 NoTarget），俯视图仍用预测状态绘制。
    out.tracking = true;

    // 4. 预测到子弹命中时刻：实测处理延迟 + 电控 bias + 弹道飞行时间。
    // 在线模式下 timestamp_seconds 是取帧时刻的 steady 时钟，减去 now 即真实处理耗时
    // （毫秒级）。离线视频模式 timestamp_seconds 是视频时间轴（0..几秒），与 steady
    // 时钟不同基准，直接相减会得到数千秒的假延迟、把预测点外推到几公里外；用上限夹住，
    // 基准不一致（差值异常大）时按 0 处理——离线回放本就不需要补偿实时处理延迟。
    const auto now_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    double measured_latency_s = 0.0;
    if (now_s > timestamp_seconds) {
        const double raw_latency = now_s - timestamp_seconds;
        measured_latency_s = raw_latency < kMaxMeasuredLatencyS ? raw_latency : 0.0;
    }

    const int aim_slot = tracked_slot_;
    const double r = slotRadius(x, aim_slot);              // 半径/板高在 CTRV 下不变
    const double phase = slotPhase(aim_slot);
    const double height_up = -x(slotHeightIndex(aim_slot));  // 板高不变，上为正

    // 飞行时间与命中点位置相互耦合：t_total 决定外推多远，外推出的命中距离又反过来
    // 改变飞行时间。与参考 Planner 一致做不动点迭代（慢目标 1~2 次即收敛）。
    const double rho0 = std::sqrt(x(kCx) * x(kCx) + x(kCz) * x(kCz));
    Bullet_Model bullet(static_cast<float>(rho0), static_cast<float>(height_up));
    bullet.velo = static_cast<float>(kBulletSpeedMps);
    double flight_s = bullet.bullet_iteration_cal();
    double beta = bullet.beta;
    for (int iter = 0; iter < 10; ++iter) {
        const double t_total = measured_latency_s + gimbal.prediction_bias_s + flight_s;
        const Eigen::VectorXd xp = processModel(x, t_total);
        const double pxi = xp(kCx) - r * std::cos(xp(kYaw) + phase);
        const double pzi = xp(kCz) - r * std::sin(xp(kYaw) + phase);
        Bullet_Model b2(static_cast<float>(std::sqrt(pxi * pxi + pzi * pzi)),
                        static_cast<float>(height_up));
        b2.velo = static_cast<float>(kBulletSpeedMps);
        const double flight_next = b2.bullet_iteration_cal();
        const double beta_next = b2.beta;
        const bool converged = std::fabs(flight_next - flight_s) < 1e-3;
        flight_s = flight_next;
        beta = beta_next;
        if (converged) {
            break;
        }
    }

    // 用收敛后的 t_total 外推命中点，再算云台指令角。
    const double t_total = measured_latency_s + gimbal.prediction_bias_s + flight_s;
    const Eigen::VectorXd x_pred = processModel(x, t_total);
    const double yaw_t = x_pred(kYaw);
    const double px = x_pred(kCx) - r * std::cos(yaw_t + phase);
    const double py = x_pred(slotHeightIndex(aim_slot));
    const double pz = x_pred(kCz) - r * std::sin(yaw_t + phase);

    // 5. 云台指令角度（gimbal 系）。
    const double rho_t = std::sqrt(px * px + pz * pz);
    const double target_yaw = std::atan2(px, pz);
    // pitch 取弹道模型迭代出的发射仰角 beta（含重力 + 空气阻力下坠补偿），不再用
    // 纯几何视线角 atan2(-py, rho_t)——纯视线角没补下坠，2~5m 会偏低 1°~2.5°。
    const double target_pitch = beta;

    // 6. 角速度（解析，参考 Planner::calculateTargetYawVelocity）。
    const double vx_a = x_pred(kVx) + r * std::sin(yaw_t + phase) * x_pred(kYawRate);
    const double vz_a = x_pred(kVz) - r * std::cos(yaw_t + phase) * x_pred(kYawRate);
    const double rho2 = px * px + pz * pz;
    // d/dt atan2(px, pz) = (pz*vx - px*vz)/rho²（对 atan2 的 y=px、x=pz 求导）。
    const double target_yaw_vel = rho2 > 1e-6 ? (pz * vx_a - px * vz_a) / rho2 : 0.0;
    const double rho_dot = rho2 > 1e-6 ? (px * vx_a + pz * vz_a) / std::sqrt(rho2) : 0.0;
    const double target_pitch_vel =
        (rho2 + py * py) > 1e-6 ? (py / (rho2 + py * py)) * rho_dot : 0.0;

    // 7. 开火许可。
    const double dist = std::sqrt(rho2 + py * py);
    const double yaw_err = std::fabs(wrapToPi(target_yaw - gimbal.yaw));
    const bool fire_ok = stable && yaw_err < kFireYawErrorRad &&
                         dist >= kFireMinDistanceM && dist <= kFireMaxDistanceM;

    // 8. 填结果（米/弧度）。协议 coo_x/coo_y 是地面平面两个轴，这里把 z(前) 记为 y。
    // 板丢失后的「及时删除编号」：连续无观测超过 kHoldLostSeconds（0.2s）就清空当前
    // 板编号/槽位并报 NoTarget，不再把旧编号挂满 1.5s（旋转过快、检测跟不上时旧编号
    // 早已失效）。0.2s 的宽容窗能桥过 1~8 帧的正常关联抖动，避免俯视图一帧一闪。
    // EKF 仍纯预测到 kLongLossSeconds 才重置，转回后沿用 slot_to_id_ 原编号、不重编。
    const bool locked = (frames_since_observation_ * dt <= kHoldLostSeconds);
    out.state = locked ? (stable ? TrackingState::Stable : TrackingState::Unstable)
                       : TrackingState::NoTarget;
    out.target_id = locked ? tracked_target_id_ : 0;
    out.tracked_slot = locked ? static_cast<std::uint8_t>(tracked_slot_) : 0;
    for (int s = 0; s < 4; ++s) {
        out.slot_ids[s] = slot_to_id_[s];
        out.slot_digits[s] = slot_to_digit_[s];
    }
    out.tracked_id = locked ? slot_to_id_[tracked_slot_] : 0;
    out.center_x_m = static_cast<float>(x(kCx));
    out.center_y_m = static_cast<float>(x(kCz));
    out.center_velocity_x_m_s = static_cast<float>(x(kVx));
    out.center_velocity_y_m_s = static_cast<float>(x(kVz));
    out.body_yaw = static_cast<float>(x(kYaw));
    out.body_yaw_velocity = static_cast<float>(x(kYawRate));
    out.radius_1_m = static_cast<float>(x(kR));
    out.radius_2_m = static_cast<float>(x(kR));  // 单一半径：协议保留两个字段，同值
    out.distance_m = static_cast<float>(dist);
    out.target_yaw = static_cast<float>(target_yaw);
    out.target_pitch = static_cast<float>(target_pitch);
    out.target_yaw_velocity = static_cast<float>(target_yaw_vel);
    out.target_pitch_velocity = static_cast<float>(target_pitch_vel);
    out.fire_allowed = fire_ok;
    return out;
}
