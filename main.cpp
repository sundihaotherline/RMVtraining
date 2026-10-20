#include "AppSupport.hpp"
#include "SerialPort.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

// 这个文件有意保留 aimbot_26/main.cpp 的组织方式：
// main 负责初始化，ReadFunction 负责收串口，OperationFunction 负责整条视觉流水线。
// 学生只需要替换 student/src/StudentTasks.cpp 中的四个任务函数。

namespace {

constexpr int kDefaultWebPort = 8080;
constexpr int kInitialReadAttempts = 5;
constexpr auto kIdleDelay = std::chrono::milliseconds(10);
// 控制器数据过期后，结果必须降级为无目标，不能继续允许开火。
constexpr auto kMessageMaxAge = std::chrono::milliseconds(500);

// 程序退出标志。原子变量同时供信号处理函数、读线程和操作线程使用。
static_assert(std::atomic<bool>::is_always_lock_free,
              "退出标志必须支持无锁原子操作");
std::atomic<bool> shutdown_requested{false};

// [离线调试] 串口是否已连接。为 false 时跳过串口收发与控制器依赖，
// 仅用相机跑实时检测/框选（敌方颜色见 debugEnemyColor()）。
bool serial_connected = false;

// 下面几个全局变量对应参考工程中的 camera、temp、translator、pic。
// 相机 SDK 的对象和生命周期由学生在 get_pic() 中自行管理，这里保存已标定参数。
CameraParameters camera;
Translator temp;       // 读线程写入的最新控制器帧
Translator translator; // 操作线程使用并回写的当前帧
cv::Mat pic;           // 操作线程当前取得的图像
// 上一帧 EKF 结果（一车一个，按出现顺序），供离线可视化窗口叠加多个俯视图与编号。
std::vector<PredictionResult> g_last_predictions;

// 实际转速（测量 ω，仅用于可视化对照，绝不喂给 EKF）。由相邻帧板朝向 yaw 差分
// 得到，跨帧保存上一步朝向与时间戳，轻 EMA 平滑后缓存在这里。
double g_meas_omega_rad_s = 0.0;
bool g_has_meas_yaw = false;
double g_prev_meas_yaw = 0.0;
double g_prev_meas_time = 0.0;
int g_meas_gap_frames = 0;

// 朝向差分的最小帧间隔：低于此值说明时间戳抖动（两帧几乎同刻），d/dt 会爆炸成几百
// rad/s 的假 ω，跳过该样本。瞬时 ω 限幅再兜底（步兵转速远达不到 10 rad/s）。
constexpr double kMinMeasDtS = 0.005;
constexpr double kMaxMeasWInst = 20.0;  // rad/s，约 1146°/s（~3 圈/秒，远高于步兵实际转速）

// 角度归一化到 (-π, π]，用于相邻帧朝向差分的 ±π 跳变处理。
static double wrapToPi(double angle) {
    while (angle > CV_PI) angle -= 2.0 * CV_PI;
    while (angle < -CV_PI) angle += 2.0 * CV_PI;
    return angle;
}

// 多车结果里选一个「主目标」用于串口回传与主日志：优先稳定锁定，其次任意锁定，
// 同状态取距离最近者；全无则返回 NoTarget 默认值。
static PredictionResult selectPrimary(const std::vector<PredictionResult>& preds) {
    const PredictionResult* best = nullptr;
    for (const PredictionResult& p : preds) {
        if (p.state == TrackingState::NoTarget) {
            continue;
        }
        if (best == nullptr ||
            (p.state == TrackingState::Stable && best->state != TrackingState::Stable) ||
            (p.state == best->state && p.distance_m < best->distance_m)) {
            best = &p;
        }
    }
    return best != nullptr ? *best : PredictionResult{};
}

// temp 及其时间戳由同一把锁保护，避免读线程写入时操作线程同时复制。
std::mutex message_mutex;
bool message_available = false;
std::chrono::steady_clock::time_point message_received_at{};

int web_port = kDefaultWebPort;

struct Options {
    std::string serial_device;
    int baud_rate{115200};
    std::string camera_config;
};

void request_shutdown(int) {
    shutdown_requested.store(true, std::memory_order_relaxed);
}

void printUsage(const char* program) {
    std::cout << "用法: " << program
              << " <串口设备> <波特率> <相机标定文件> [Web端口]\n"
              << "示例: " << program
              << " /dev/ttyACM0 115200 config/camera.yaml 8080\n";
}

bool parsePositiveInt(const char* text, int& value) {
    try {
        std::size_t parsed = 0;
        const int candidate = std::stoi(text, &parsed);
        if (text[parsed] != '\0' || candidate <= 0) {
            return false;
        }
        value = candidate;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseOptions(int argc, char** argv, Options& options) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return false;
    }
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return false;
    }

    options.serial_device = argv[1];
    options.camera_config = argv[3];
    if (!parsePositiveInt(argv[2], options.baud_rate)) {
        std::cerr << "波特率无效: " << argv[2] << '\n';
        return false;
    }
    if (argc == 5 && !parsePositiveInt(argv[4], web_port)) {
        std::cerr << "Web端口无效: " << argv[4] << '\n';
        return false;
    }
    return true;
}

bool hasFrameMarkers(const char* frame) noexcept {
    // 帧头和帧尾只用于确认边界；浮点数是否有效在操作线程中检查。
    return static_cast<std::uint8_t>(frame[0]) == kFrameHead &&
           static_cast<std::uint8_t>(frame[kCommunicationFrameSize - 1]) == kFrameTail;
}

bool hasFiniteControlInput(const MessData_AutoAim& message) noexcept {
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias);
}

// [离线调试] 无串口时决定敌方颜色，可用环境变量 RMV_ENEMY_COLOR=blue 覆盖，
// 默认红色敌方（对应参考工程 status==5 的场景）。
TeamColor debugEnemyColor() {
    static const TeamColor color = [] {
        const char* env = std::getenv("RMV_ENEMY_COLOR");
        return (env != nullptr && std::string(env) == "blue") ? TeamColor::Blue
                                                              : TeamColor::Red;
    }();
    return color;
}

void publishFrame(const Translator& incoming) {
    std::lock_guard<std::mutex> lock(message_mutex);
    temp = incoming;
    message_received_at = std::chrono::steady_clock::now();
    message_available = true;
}

bool copyLatestFrame(Translator& destination,
                     std::chrono::steady_clock::time_point& received_at) {
    std::lock_guard<std::mutex> lock(message_mutex);
    if (!message_available) {
        return false;
    }
    destination = temp;
    received_at = message_received_at;
    return true;
}

bool frameIsFresh(const std::chrono::steady_clock::time_point& received_at) {
    return std::chrono::steady_clock::now() - received_at <= kMessageMaxAge;
}

bool writeFrame(SerialPort& serialPort, Translator& frame) {
    // 统一设置协议边界，并保证一次写出完整的 64 字节帧。
    frame.message.head = kFrameHead;
    frame.message.tail = kFrameTail;
    const int written =
        serialPort.Write(frame.data, static_cast<int>(kCommunicationFrameSize));
    if (written != static_cast<int>(kCommunicationFrameSize)) {
        std::cerr << "串口写入失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void logPrediction(const PredictionResult& result) {
    WEB_LOG("target_found", result.state == TrackingState::NoTarget ? 0.0 : 1.0);
    WEB_LOG("ekf_state", static_cast<double>(result.state));
    // 始终输出全部字段：NoTarget 时 result 为默认全 0，仍会打出 0 值。这样网页端
    // 每张曲线卡自程序启动就存在，不会因为目标未锁上（NoTarget）而整张卡消失。
    WEB_LOG("target_id", result.target_id);
    WEB_LOG("tracked_slot", result.tracked_slot);
    WEB_LOG("tracked_id", result.tracked_id);
    WEB_LOG("target_yaw", result.target_yaw);
    WEB_LOG("target_pitch", result.target_pitch);
    WEB_LOG("target_distance", result.distance_m);
    WEB_LOG("ekf_x_m", result.center_x_m);
    WEB_LOG("ekf_y_m", result.center_y_m);
    WEB_LOG("ekf_vx_m_s", result.center_velocity_x_m_s);
    WEB_LOG("ekf_vy_m_s", result.center_velocity_y_m_s);
    WEB_LOG("ekf_yaw_rad", result.body_yaw);
    WEB_LOG("ekf_yaw_rate_rad_s", result.body_yaw_velocity);
    WEB_LOG("ekf_radius_1_m", result.radius_1_m);
    WEB_LOG("ekf_radius_2_m", result.radius_2_m);
    // 四个槽位的出现顺序编号与识别到的贴纸数字（0 = 尚未识别），网页端直接看
    // 「板编号是否稳定、贴纸数字是否读对」。
    WEB_LOG("slot_id0", result.slot_ids[0]);
    WEB_LOG("slot_id1", result.slot_ids[1]);
    WEB_LOG("slot_id2", result.slot_ids[2]);
    WEB_LOG("slot_id3", result.slot_ids[3]);
    WEB_LOG("slot_digit0", result.slot_digits[0]);
    WEB_LOG("slot_digit1", result.slot_digits[1]);
    WEB_LOG("slot_digit2", result.slot_digits[2]);
    WEB_LOG("slot_digit3", result.slot_digits[3]);
}

// [离线调试] 叠加 EKF 结果：每辆车一行状态文字 + 一个车体俯视示意图 + 主目标
// 预测瞄准点反投影准星。只用于可视化理解，不参与在线串口路径。
void drawEkfOverlay(cv::Mat& vis, const std::vector<PredictionResult>& preds,
                    const CameraParameters& cam) {
    // 预测转速（EKF 的 body_yaw_velocity）与实际转速（相邻帧朝向差分，仅显示）。
    char vel[160];
    std::snprintf(vel, sizeof(vel), "pred w=%.0f  meas w=%.0f deg/s",
                  preds.empty() ? 0.0 : preds.front().body_yaw_velocity * 180.0 / CV_PI,
                  g_meas_omega_rad_s * 180.0 / CV_PI);
    cv::putText(vis, vel, cv::Point(12, vis.rows - 68), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

    // 过滤出有效车辆：只要 EKF 还持有估计（tracking）就画俯视图——快速旋转的哨兵板
    // 转到背面被遮挡时 state 会暂时 NoTarget，但中心/yaw/半径/编号仍由 EKF 预测有效，
    // 俯视图继续绘制，避免整张图一隐一现（「跟不上」的观感来源）。真正的目标丢失由
    // 1.5s 无观测重置后 tracking=false 才停止绘制。
    std::vector<const PredictionResult*> valid;
    for (const PredictionResult& p : preds) {
        if (p.tracking) {
            valid.push_back(&p);
        }
    }
    if (valid.empty()) {
        cv::putText(vis, "EKF: no target", cv::Point(12, vis.rows - 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return;
    }

    const int box = 220;          // 单辆车俯视图边长（px），多车时略缩以并排
    const double scale = 300.0;   // 像素 / 米
    const double plate_half_len = 0.075;  // 半板宽（米），约小装甲板宽的一半
    const int gap = 28;           // 相邻俯视图间距（px）

    for (std::size_t k = 0; k < valid.size(); ++k) {
        const PredictionResult& p = *valid[k];
        const int car = static_cast<int>(k) + 1;  // 出现顺序车号，仅用于区分俯视图
        // 布局：从右上角向左依次排布各车俯视图。
        const cv::Point2f o(
            vis.cols - 20.0F - box / 2.0F - static_cast<float>(k) * (box + gap),
            box / 2.0F + 20.0F);

        // 每车一行状态文字（画在俯视图上方）。
        char text[320];
        std::snprintf(text, sizeof(text),
                      "car%d plate=%d slot=%d c=(%.2f,%.2f) v=(%.2f,%.2f) yaw=%.0f w=%.0f r=%.2f",
                      car, static_cast<int>(p.tracked_id), static_cast<int>(p.tracked_slot),
                      p.center_x_m, p.center_y_m,
                      p.center_velocity_x_m_s, p.center_velocity_y_m_s,
                      p.body_yaw * 180.0 / CV_PI, p.body_yaw_velocity * 180.0 / CV_PI,
                      p.radius_1_m);
        cv::putText(vis, text,
                    cv::Point(static_cast<int>(o.x - box / 2.0F),
                              static_cast<int>(o.y - box / 2.0F - 8.0F)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        // 坐标轴 + 轴标签。
        cv::line(vis, o, o + cv::Point2f(box / 2.0F - 10.0F, 0), cv::Scalar(120, 120, 120), 1);
        cv::line(vis, o, o + cv::Point2f(0, -(box / 2.0F - 10.0F)), cv::Scalar(120, 120, 120), 1);
        cv::putText(vis, "x", o + cv::Point2f(box / 2.0F - 14.0F, 14.0F),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(140, 140, 140), 1);
        cv::putText(vis, "z", o + cv::Point2f(6.0F, -(box / 2.0F - 16.0F)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(140, 140, 140), 1);
        // 单一半径参考圆（一辆车四块板到旋转中心距离相同）。
        if (p.radius_1_m > 0) {
            cv::circle(vis, o, static_cast<int>(p.radius_1_m * scale), cv::Scalar(120, 120, 120), 1);
        }
        // 四块装甲板：俯视下每块板是一条切向短线段（板面），标出现顺序编号，跟踪中的板高亮。
        for (int s = 0; s < 4; ++s) {
            const double phase = s * (CV_PI / 2.0);
            const double r = p.radius_1_m;  // 单一半径
            const double rel_x = -r * std::cos(p.body_yaw + phase);
            const double rel_z = -r * std::sin(p.body_yaw + phase);
            const cv::Point2f pt(static_cast<float>(o.x + rel_x * scale),
                                 static_cast<float>(o.y - rel_z * scale));
            // 板面切向单位向量（屏幕坐标）：径向 (-cos,-sin) 的垂向 (+sin,-cos)。
            const float tx = static_cast<float>(std::sin(p.body_yaw + phase));
            const float tz = static_cast<float>(-std::cos(p.body_yaw + phase));
            const float hl = static_cast<float>(plate_half_len * scale);
            // 只有真正锁定（state != NoTarget）才高亮当前板；NoTarget 时 tracked_slot 恒 0，
            // 若仍高亮会把 0 号板误标成跟踪中。
            const bool is_tracked = p.state != TrackingState::NoTarget &&
                                    (s == static_cast<int>(p.tracked_slot));
            const cv::Scalar color =
                is_tracked ? cv::Scalar(0, 255, 0) : cv::Scalar(210, 210, 210);
            cv::line(vis, pt - cv::Point2f(tx * hl, -tz * hl), pt + cv::Point2f(tx * hl, -tz * hl),
                     color, is_tracked ? 3 : 2);
            // 出现顺序编号（id）与识别到的贴纸数字（digit）：id/digit 一并显示。
            const int pid = static_cast<int>(p.slot_ids[s]);
            if (pid > 0) {
                const double rr = r + 0.05;
                const cv::Point2f lp(
                    static_cast<float>(o.x + (-rr * std::cos(p.body_yaw + phase)) * scale),
                    static_cast<float>(o.y - (-rr * std::sin(p.body_yaw + phase)) * scale));
                const int pdigit = static_cast<int>(p.slot_digits[s]);
                const std::string tag = pdigit > 0
                                            ? std::to_string(pid) + "/" + std::to_string(pdigit)
                                            : std::to_string(pid);
                cv::putText(vis, tag, lp,
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
            }
        }
        cv::circle(vis, o, 3, cv::Scalar(0, 0, 255), -1);  // 车体中心
        cv::line(vis, o, o + cv::Point2f(static_cast<float>(p.center_velocity_x_m_s * scale),
                                         static_cast<float>(-p.center_velocity_y_m_s * scale)),
                 cv::Scalar(0, 0, 255), 2);  // 速度向量
        // 该车瞄准点星标（相对车中心）。
        const double rho = p.distance_m * std::cos(p.target_pitch);
        const double aim_x = rho * std::sin(p.target_yaw);
        const double aim_z = rho * std::cos(p.target_yaw);
        const cv::Point aim_pt(static_cast<int>(o.x + (aim_x - p.center_x_m) * scale),
                               static_cast<int>(o.y - (aim_z - p.center_y_m) * scale));
        cv::drawMarker(vis, aim_pt, cv::Scalar(0, 255, 0), cv::MARKER_STAR, 12, 2);
    }

    // 主目标（首辆有效车）的预测瞄准点反投影回图像，画准星，直观对比 EKF 预测位置与检测框。
    if (cam.valid()) {
        const PredictionResult& p = *valid.front();
        const double rho = p.distance_m * std::cos(p.target_pitch);
        const double aim_x = rho * std::sin(p.target_yaw);
        const double aim_y = -p.distance_m * std::sin(p.target_pitch);
        const double aim_z = rho * std::cos(p.target_yaw);
        cv::Mat aim_gimbal = (cv::Mat_<double>(3, 1) << aim_x, aim_y, aim_z);
        cv::Mat aim_cam = cam.rotation_camera_to_gimbal.t() *
                          (aim_gimbal - cam.translation_camera_to_gimbal);
        std::vector<cv::Point3d> obj{cv::Point3d(aim_cam.at<double>(0),
                                                 aim_cam.at<double>(1),
                                                 aim_cam.at<double>(2))};
        std::vector<cv::Point2d> img;
        cv::projectPoints(obj, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
                          cam.camera_matrix, cam.distortion_coefficients, img);
        if (!img.empty() && img[0].x >= 0 && img[0].y >= 0 &&
            img[0].x < vis.cols && img[0].y < vis.rows) {
            cv::drawMarker(vis, cv::Point(static_cast<int>(img[0].x),
                                          static_cast<int>(img[0].y)),
                           cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);
        }
    }
}

// main 中启动线程前先同步读一帧，保持参考工程的初始化顺序。
// 返回 1 表示读到合法帧，0 表示暂时超时，-1 表示串口出错。
int readInitialFrame(SerialPort& serialPort, Translator& frame) {
    for (int attempt = 0; attempt < kInitialReadAttempts &&
                            !shutdown_requested.load(std::memory_order_relaxed);
         ++attempt) {
        Translator incoming{};
        const int length =
            serialPort.Read(incoming.data, static_cast<int>(kCommunicationFrameSize));
        if (length < 0) {
            return -1;
        }
        if (length == static_cast<int>(kCommunicationFrameSize) &&
            hasFrameMarkers(incoming.data)) {
            frame = incoming;
            return 1;
        }
    }
    return 0;
}

}  // 匿名命名空间

// 串口读线程：只负责读取完整帧并更新共享的 temp。
void *ReadFunction(void *arg);
// 操作线程：按照“复制 -> 取流 -> 检测 -> PnP -> EKF -> 写回”的顺序处理。
void *OperationFunction(void *arg);

int main(int argc, char** argv) {
    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);

    Options options;
    if (!parseOptions(argc, argv, options)) {
        return argc == 2 && std::string(argv[1]) == "--help" ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::string camera_error;
    if (!loadCameraParameters(options.camera_config, camera, camera_error)) {
        // [离线调试] 相机未标定时不退出，仍可跑实时检测/框选（PnP 会被跳过）。
        std::cerr << "[离线调试] " << camera_error << '\n';
    }

    // 与参考工程一样，串口对象在 main 中创建，再把指针传给两个线程。
    // [离线调试] 串口打不开时不退出，仅关闭串口收发。
    SerialPort* serialPort = new SerialPort(options.serial_device);
    serial_connected = serialPort->isOpen();
    if (!serial_connected) {
        std::cerr << "[离线调试] 未打开串口设备: " << options.serial_device
                  << "，仅运行相机实时检测\n";
    } else if (!serialPort->InitSerialPort(options.baud_rate, 8, 1, 'N')) {
        std::cerr << "串口配置失败\n";
        delete serialPort;
        return EXIT_FAILURE;
    }

    // 先同步读取首帧并缓存控制器状态；后续每帧仍会重新读取 status 来确定敌方颜色。
    // [离线调试] 无串口时跳过首帧读取。
    if (serial_connected) {
        Translator first_frame{};
        const int initial_result = readInitialFrame(*serialPort, first_frame);
        if (initial_result < 0) {
            std::cerr << "读取串口首帧失败\n";
            delete serialPort;
            return EXIT_FAILURE;
        }
        if (initial_result > 0) {
            publishFrame(first_frame);
            std::cout << "收到首帧，status = "
                      << static_cast<unsigned int>(first_frame.message.status) << '\n';
        } else {
            // 首帧超时不直接退出，读线程启动后仍会继续等待控制器数据。
            std::cerr << "未在启动阶段收到合法首帧，继续等待串口数据\n";
        }
    }

    pthread_t readThread{};
    pthread_t operationThread{};
    bool read_started = false;
    bool operation_started = false;

    // 保留参考工程的 pthread 双线程结构：先创建读线程，再创建操作线程。
    // [离线调试] 无串口时跳过读线程，但仍创建操作线程跑相机检测。
    if (serial_connected) {
        if (pthread_create(&readThread, nullptr, ReadFunction, serialPort) == 0) {
            read_started = true;
        } else {
            std::cerr << "创建串口读线程失败\n";
        }
    }
    if ((!serial_connected || read_started) &&
        pthread_create(&operationThread, nullptr, OperationFunction, serialPort) == 0) {
        operation_started = true;
    } else {
        std::cerr << "创建视觉操作线程失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
    }

    // 先等待操作线程结束，再通知读线程退出，和原 main 的生命周期一致。
    if (operation_started) {
        pthread_join(operationThread, nullptr);
    }
    shutdown_requested.store(true, std::memory_order_relaxed);
    if (read_started) {
        pthread_join(readThread, nullptr);
    }

    STOP_WEB_SERVER();
    delete serialPort;
    return operation_started ? EXIT_SUCCESS : EXIT_FAILURE;
}

void *ReadFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        return nullptr;
    }

    // SerialPort::Read 已经负责短读缓存；这里额外保留滑动缓冲，
    // 使程序从半帧启动或遇到干扰字节时仍能重新找到 0x71 ... 0x4C。
    std::vector<char> stream_buffer;
    stream_buffer.reserve(kCommunicationFrameSize * 2);

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        std::array<char, kCommunicationFrameSize> chunk{};
        const int length =
            serialPort->Read(chunk.data(), static_cast<int>(chunk.size()));
        if (length < 0) {
            std::cerr << "串口读取失败\n";
            shutdown_requested.store(true, std::memory_order_relaxed);
            break;
        }
        if (length == 0) {
            // 超时不是错误；短暂让出 CPU 后继续等待下一帧。
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }
        if (length > static_cast<int>(chunk.size())) {
            continue;
        }

        stream_buffer.insert(stream_buffer.end(), chunk.begin(), chunk.begin() + length);
        std::size_t consumed = 0;
        while (stream_buffer.size() - consumed >= kCommunicationFrameSize) {
            const char* candidate = stream_buffer.data() + consumed;
            if (!hasFrameMarkers(candidate)) {
                ++consumed;
                continue;
            }

            Translator incoming{};
            std::memcpy(incoming.data, candidate, kCommunicationFrameSize);
            consumed += kCommunicationFrameSize;
            publishFrame(incoming);
        }
        if (consumed != 0U) {
            stream_buffer.erase(stream_buffer.begin(),
                                stream_buffer.begin() + static_cast<std::ptrdiff_t>(consumed));
        }
    }
    return nullptr;
}

void *OperationFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        return nullptr;
    }

    // Web 服务和相机 SDK 都属于操作线程的资源，生命周期与原工程一致。
    START_WEB_SERVER(web_port);

    double dt = 0.0;
    double last_time_stamp = 0.0;
    auto last_frame_time = std::chrono::steady_clock::time_point{};
    auto last_viz_time = std::chrono::steady_clock::time_point{};
    int empty_frame_count = 0;

    // [离线视频] 处理视频文件：RMV_TEST_VIDEO 指定输入，RMV_TEST_OUT 指定带标注的
    // 输出视频路径，RMV_TEST_CSV 指定逐帧信号落盘路径（供离线画曲线验证）。这三者
    // 只在无串口的离线调试模式下有意义。
    const char* out_path_env = std::getenv("RMV_TEST_OUT");
    const std::string out_video_path = out_path_env != nullptr ? out_path_env : "";
    cv::VideoWriter out_writer;
    bool out_writer_ready = false;

    const char* csv_path_env = std::getenv("RMV_TEST_CSV");
    const std::string csv_path = csv_path_env != nullptr ? csv_path_env : "";
    std::ofstream csv_out;
    long long frame_index = 0;
    if (!csv_path.empty()) {
        csv_out.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv_out.is_open()) {
            csv_out << "frame_idx,t_sec,armor_count,detection_id,pnp_reproj,"
                       "pnp_dist_m,armor_orientation_yaw,armor_position_yaw,"
                       "meas_w,ekf_state,tracked_slot,tracked_id,slot_id0,slot_id1,slot_id2,slot_id3,slot_digit0,slot_digit1,slot_digit2,slot_digit3,target_id,"
                       "ekf_x_m,ekf_y_m,ekf_vx,ekf_vy,ekf_yaw,ekf_w,"
                       "radius_1_m,radius_2_m,target_yaw,target_pitch,target_distance,"
                       "bars_count,pair_rejected,pair_ratio,vehicle_count";
            for (int v = 1; v <= 3; ++v) {
                csv_out << ",v" << (v + 1) << "_tracked_id"
                        << ",v" << (v + 1) << "_tracked_slot"
                        << ",v" << (v + 1) << "_slot0"
                        << ",v" << (v + 1) << "_slot1"
                        << ",v" << (v + 1) << "_slot2"
                        << ",v" << (v + 1) << "_slot3"
                        << ",v" << (v + 1) << "_digit0"
                        << ",v" << (v + 1) << "_digit1"
                        << ",v" << (v + 1) << "_digit2"
                        << ",v" << (v + 1) << "_digit3"
                        << ",v" << (v + 1) << "_ekf_x"
                        << ",v" << (v + 1) << "_ekf_y"
                        << ",v" << (v + 1) << "_ekf_yaw"
                        << ",v" << (v + 1) << "_ekf_w"
                        << ",v" << (v + 1) << "_r1"
                        << ",v" << (v + 1) << "_r2";
            }
            csv_out << "\n";
        }
    }

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        const auto process_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point received_at{};

        GimbalState gimbal{};
        if (serial_connected) {
            // 对应 aimbot_26 中的 MManager.copy(temp, translator)。
            if (!copyLatestFrame(translator, received_at)) {
                std::this_thread::sleep_for(kIdleDelay);
                continue;
            }

            // 不支持的状态或异常姿态不进入学生算法，直接回传无目标帧。
            if (!frameIsFresh(received_at) || !isAutoAimStatus(translator.message.status) ||
                !hasFiniteControlInput(translator.message)) {
                setSafeOutput(translator);
                writeFrame(*serialPort, translator);
                // 同一份过期快照可能会被重复看到，避免无休止地高速回包。
                std::this_thread::sleep_for(kIdleDelay);
                continue;
            }
            gimbal = gimbalStateFromMessage(translator.message);
        }
        // [离线调试] 无串口时 gimbal 保持 0，敌方颜色由 debugEnemyColor() 决定。

        // 对应参考工程中的 camera.get_pic(&pic, ...)。具体 SDK 初始化和取流由学生完成。
        pic.release();
        const bool got_pic = get_pic(pic);
        if (!got_pic && is_video_source() && video_source_ended()) {
            // 离线视频播完：正常退出，落盘输出视频/CSV 后停止。
            shutdown_requested.store(true, std::memory_order_relaxed);
            break;
        }
        if (!got_pic || pic.empty()) {
            // 相机断连/取流失败：不跑检测流水线，回传安全帧后短暂等待并重试重连。
            // get_pic() 已在失败时释放失效的相机句柄，下一轮会重新 open() 枚举设备。
            ++empty_frame_count;
            WEB_LOG("empty_frame", static_cast<double>(empty_frame_count));
            if (empty_frame_count == 1) {
                std::cerr << "相机断连/取流失败，等待重连...\n";
            }
            if (serial_connected) {
                setSafeOutput(translator);
                writeFrame(*serialPort, translator);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        if (empty_frame_count != 0) {
            std::cerr << "相机已重连，恢复取流\n";
        }
        empty_frame_count = 0;
        WEB_LOG("empty_frame", 0.0);
        // 实际取流分辨率：与 config/camera.yaml 的 1440x1080 不一致时，内参 cx/cy/fx/fy
        // 全部对不上，PnP 位置会整体缩放/偏移，正是「一辆车被拆成多辆」的常见根因。
        WEB_LOG("frame_width", static_cast<double>(pic.cols));
        WEB_LOG("frame_height", static_cast<double>(pic.rows));

        const auto frame_time = std::chrono::steady_clock::now();
        // 离线视频模式下，EKF/显示 ω 用的时间戳取自视频自身时间轴（逐帧单调、dt 真实），
        // 而非壁钟——否则全速回放时壁钟 dt 远小于视频 dt，ω 会被系统性低估。
        const bool video_mode = is_video_source();
        double frame_timestamp_sec = 0.0;
        double camera_fps = 0.0;
        if (video_mode) {
            frame_timestamp_sec = get_pic_timestamp_sec();
            camera_fps = get_video_fps();
            WEB_LOG("camera_fps", camera_fps);
        } else {
            frame_timestamp_sec =
                std::chrono::duration<double>(frame_time.time_since_epoch()).count();
            if (last_frame_time.time_since_epoch().count() != 0) {
                const double frame_seconds =
                    std::chrono::duration<double>(frame_time - last_frame_time).count();
                if (frame_seconds > 0.0) {
                    camera_fps = 1.0 / frame_seconds;
                    WEB_LOG("camera_fps", camera_fps);
                }
            }
            last_frame_time = frame_time;
        }

        // 1. 装甲板检测。检测结果为空时不提前结束，继续向下传递空观测。
        const TeamColor enemy_color =
            serial_connected ? enemyColorFromStatus(translator.message.status)
                             : debugEnemyColor();
        const std::vector<ArmorDetection> detections = armor_detect(pic, enemy_color);
        WEB_LOG("armor_count", static_cast<double>(detections.size()));
        if (!detections.empty()) {
            // 记录第一块候选装甲板，便于在网页中快速判断检测器是否工作。
            const ArmorDetection& first_detection = detections.front();
            WEB_LOG("detection_id", first_detection.target_id);
            WEB_LOG("detection_confidence", first_detection.confidence);
            WEB_LOG("detection_size", first_detection.size == ArmorSize::Large ? 1.0 : 0.0);
            const cv::Point2f center =
                (first_detection.corners[0] + first_detection.corners[1] +
                 first_detection.corners[2] + first_detection.corners[3]) * 0.25F;
            WEB_LOG("target_x_px", center.x);
            WEB_LOG("target_y_px", center.y);
        }

        // 2. PnP 位姿解算。PnP 失败时同样保留空 observations，交给 EKF 做纯预测。
        const std::vector<ArmorPose> observations =
            armor_solve(detections, camera, gimbal);
        if (!observations.empty()) {
            WEB_LOG("pnp_reprojection_error", observations.front().reprojection_error);
            WEB_LOG("pnp_distance_m", cv::norm(observations.front().position_gimbal_m));
            // 诊断：板朝向（法向 yaw，喂给 EKF 的 yaw 直接观测）与板位置角。
            // 静止板若这两条随时间漂移，说明是 PnP 输入在漂；若它们恒定而
            // ekf_yaw_rate_rad_s 仍非零，说明是 ω 弱可观测/收敛慢。
            WEB_LOG("armor_orientation_yaw", observations.front().armor_orientation_yaw);
            WEB_LOG("armor_position_yaw", observations.front().armor_yaw);
        } else {
            WEB_LOG("pnp_reprojection_error", -1.0);
        }

        // 实际转速（测量 ω，仅显示）：对板表面朝向 yaw 做相邻帧差分。用
        // -armor_orientation_yaw 作车体 yaw 的代理（槽位相位 φ 是常数，在差分中抵消），
        // 其导数就是干净的 ω；比位置角更不受车体中心偏移影响。±π 朝向翻转样本
        // 用 0.5rad 门限丢弃，结果轻 EMA 平滑后只用于对照显示，保持 EKF「ω 无
        // 直接观测」的独立性（绝不回喂给 EKF）。
        if (!observations.empty()) {
            const double yaw_meas = -observations.front().armor_orientation_yaw;
            const double t_meas = frame_timestamp_sec;
            if (g_has_meas_yaw && t_meas > g_prev_meas_time) {
                const double dt_meas = t_meas - g_prev_meas_time;
                const double d = wrapToPi(yaw_meas - g_prev_meas_yaw);
                // |d|<0.5 跳过 ±π 翻转 / 换板跳变；dt 下限挡时间戳抖动造成的差分爆炸
                // （dt 极小 → d/dt 到几百 rad/s）；瞬时 ω 限幅再兜底。
                if (dt_meas >= kMinMeasDtS && std::fabs(d) < 0.5) {
                    const double w_inst =
                        std::clamp(d / dt_meas, -kMaxMeasWInst, kMaxMeasWInst);
                    g_meas_omega_rad_s = 0.3 * w_inst + 0.7 * g_meas_omega_rad_s;
                }
            }
            g_prev_meas_yaw = yaw_meas;
            g_prev_meas_time = t_meas;
            g_has_meas_yaw = true;
            g_meas_gap_frames = 0;
        } else if (++g_meas_gap_frames > 15) {
            // 连续约 0.5s 无观测，判为「当前无实测转速」，归零并等待下一次初始化。
            g_meas_omega_rad_s = 0.0;
            g_has_meas_yaw = false;
        }
        WEB_LOG("meas_yaw_rate_rad_s", g_meas_omega_rad_s);

        // [离线调试] 实时框选可视化，ESC 退出。按 ~30fps 节流刷新窗口，
        // 其余帧跳过 imshow/重绘，避免把处理流水线拖慢。写输出视频时逐帧全画、不节流。
        const bool writing_output = !out_video_path.empty();
        if (!serial_connected && !pic.empty() &&
            (writing_output ||
             std::chrono::steady_clock::now() - last_viz_time >=
                 std::chrono::milliseconds(33))) {
            last_viz_time = std::chrono::steady_clock::now();
            cv::Mat vis = pic.clone();
            // 灯条级中间结果：即使没配成装甲板也画出来，便于判断检测器工作状态。
            const auto bars = armor_detect_debug(pic, enemy_color);
            for (const auto& b : bars) {
                cv::line(vis, b.top, b.bottom, cv::Scalar(0, 255, 255), 2);
                cv::circle(vis, b.center, 4, cv::Scalar(0, 255, 255), -1);
            }
            // 出现顺序编号：槽位由「板位置 + 所属车 EKF 中心/yaw」反解（无 ±π 二义性），
            // 槽位是板的固定身份；转一圈后重新出现的板沿用旧编号。多车时先按板位置到
            // 各车中心最近者确定所属车，再读该车的 slot_ids。
            auto appearanceIdOf = [&](const ArmorDetection& d) -> int {
                if (g_last_predictions.empty()) {
                    return 0;
                }
                const cv::Point2f dc = (d.corners[0] + d.corners[1] + d.corners[2] +
                                        d.corners[3]) * 0.25F;
                // 角点中心距最近的观测位姿（≤16px 视为同一块板）。
                const ArmorPose* best_pose = nullptr;
                double best_px = 1e9;
                for (const ArmorPose& p : observations) {
                    const cv::Point2f pc = (p.detection.corners[0] + p.detection.corners[1] +
                                            p.detection.corners[2] + p.detection.corners[3]) * 0.25F;
                    const double dx = dc.x - pc.x;
                    const double dy = dc.y - pc.y;
                    const double dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < best_px) {
                        best_px = dist;
                        best_pose = &p;
                    }
                }
                if (best_pose == nullptr || best_px > 16.0) {
                    return 0;
                }
                // 所属车：板位置到各车中心最近者。
                int best_v = -1;
                double best_d = 1e9;
                for (std::size_t v = 0; v < g_last_predictions.size(); ++v) {
                    const PredictionResult& pr = g_last_predictions[v];
                    if (pr.state == TrackingState::NoTarget) {
                        continue;
                    }
                    const double ddx = best_pose->position_gimbal_m[0] - pr.center_x_m;
                    const double ddz = best_pose->position_gimbal_m[2] - pr.center_y_m;
                    const double dd = std::sqrt(ddx * ddx + ddz * ddz);
                    if (dd < best_d) {
                        best_d = dd;
                        best_v = static_cast<int>(v);
                    }
                }
                if (best_v < 0) {
                    return 0;
                }
                const PredictionResult& pr = g_last_predictions[best_v];
                const double rel_yaw = std::atan2(
                    best_pose->position_gimbal_m[0] - pr.center_x_m,
                    best_pose->position_gimbal_m[2] - pr.center_y_m);
                const double phi = wrapToPi(-CV_PI / 2.0 - rel_yaw - pr.body_yaw);
                int slot = static_cast<int>(std::lround(phi / (CV_PI / 2.0)));
                slot %= 4;
                if (slot < 0) {
                    slot += 4;
                }
                return pr.slot_ids[slot];
            };
            for (const ArmorDetection& d : detections) {
                // 保持帧（held）用黄色框区分：绿色 = 本帧新检测，黄色 = 沿用上一帧框
                // 抹平单帧漏检的保持框，让「框稳不稳」一眼可辨。
                const cv::Scalar box_color = d.held ? cv::Scalar(0, 255, 255)
                                                    : cv::Scalar(0, 255, 0);
                std::vector<cv::Point> pts;
                for (const cv::Point2f& c : d.corners) {
                    pts.emplace_back(static_cast<int>(std::lround(c.x)),
                                     static_cast<int>(std::lround(c.y)));
                }
                cv::polylines(vis, std::vector<std::vector<cv::Point>>{pts}, true,
                              box_color, 2);
                // 四个角点画成实心圆点。
                for (const cv::Point2f& c : d.corners) {
                    cv::circle(vis, cv::Point(static_cast<int>(std::lround(c.x)),
                                              static_cast<int>(std::lround(c.y))),
                               5, cv::Scalar(0, 0, 255), -1);
                }
                const cv::Point2f center = (d.corners[0] + d.corners[1] +
                                            d.corners[2] + d.corners[3]) * 0.25F;
                // 标签：同时展示 id 编号与识别数字（id/数字）+ 大小 + 保持标记。
                const int app_id = appearanceIdOf(d);
                std::string label =
                    (app_id > 0 ? std::to_string(app_id) : "?") + "/";
                label += (d.target_id > 0 ? std::to_string(d.target_id) : "?");
                label += (d.size == ArmorSize::Large ? "L" : "S");
                label += (d.held ? "H" : "");
                cv::putText(vis, label, center + cv::Point2f(4, 0),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2);
            }
            // PnP 位姿读数：每块装甲板标注距离、云台系偏航/俯仰角与 (x,y,z) 坐标。
            for (const ArmorPose& p : observations) {
                const cv::Point2f center = (p.detection.corners[0] +
                                            p.detection.corners[1] +
                                            p.detection.corners[2] +
                                            p.detection.corners[3]) * 0.25F;
                char pose[128];
                std::snprintf(pose, sizeof(pose),
                              "d=%.2fm yaw=%.1f pitch=%.1f orient=%.1fdeg",
                              cv::norm(p.position_gimbal_m),
                              p.armor_yaw * 180.0 / CV_PI,
                              p.armor_pitch * 180.0 / CV_PI,
                              p.armor_orientation_yaw * 180.0 / CV_PI);
                cv::putText(vis, pose, center + cv::Point2f(4, 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                char xyz[128];
                std::snprintf(xyz, sizeof(xyz), "xyz=(%.2f, %.2f, %.2f)m",
                              p.position_gimbal_m[0], p.position_gimbal_m[1],
                              p.position_gimbal_m[2]);
                cv::putText(vis, xyz, center + cv::Point2f(4, 42),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }
            // 状态文字：帧率、灯条数、装甲数、敌方颜色。
            char status[160];
            std::snprintf(status, sizeof(status), "fps=%.1f bars=%zu armor=%zu enemy=%s",
                          camera_fps, bars.size(), detections.size(),
                          enemy_color == TeamColor::Red ? "red" : "blue");
            cv::putText(vis, status, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2);
            // PnP 诊断：对首个检测框逐项显示每道拒绝门槛的通过情况与数值。
            if (!detections.empty()) {
                const PnPDiagnostic diag = armor_solve_debug(detections.front(), camera);
                const char* size_name =
                    detections.front().size == ArmorSize::Large ? "L" : "S";
                char line[256];
                std::snprintf(line, sizeof(line),
                              "PnP: size=%s cam=%d corner=%d solved=%d depth=%.2f(in=%d) reproj=%.1f(ok=%d)",
                              size_name, diag.camera_valid ? 1 : 0,
                              diag.corners_inside ? 1 : 0,
                              diag.solved ? 1 : 0, diag.depth_m,
                              diag.depth_in_range ? 1 : 0, diag.reprojection_error_px,
                              diag.reprojection_ok ? 1 : 0);
                cv::putText(vis, line, cv::Point(12, 58), cv::FONT_HERSHEY_SIMPLEX,
                            0.7, cv::Scalar(0, 255, 255), 2);
            }
            // 配对诊断：始终显示一行。有灯条对时给 ratio 与各门槛；不足两根灯条时
            // 明确提示「无配对」，一眼区分是「配对失败」还是「灯条根本没提出来」。
            {
                const ArmorDetector::PairDiagnostic pd = armor_pair_diagnostic();
                char pline[256];
                if (pd.valid) {
                    std::snprintf(pline, sizeof(pline),
                                  "pair: ratio=%.2f lenr=%.2f ang=%.0fdeg off=%.2f %s",
                                  pd.pair_ratio, pd.length_ratio, pd.angle_diff,
                                  pd.along_offset, pd.rejected ? "REJECTED" : "ok");
                    cv::putText(vis, pline, cv::Point(12, 84), cv::FONT_HERSHEY_SIMPLEX,
                                0.7, pd.rejected ? cv::Scalar(0, 0, 255)
                                                 : cv::Scalar(0, 255, 0), 2);
                } else {
                    std::snprintf(pline, sizeof(pline), "pair: no pair (bars<2)");
                    cv::putText(vis, pline, cv::Point(12, 84), cv::FONT_HERSHEY_SIMPLEX,
                                0.7, cv::Scalar(0, 0, 255), 2);
                }
            }
            drawEkfOverlay(vis, g_last_predictions, camera);

            if (writing_output) {
                // 写输出视频：懒初始化一次 writer，逐帧写全分辨率标注帧，不做窗口显示。
                if (!out_writer_ready) {
                    out_writer.open(out_video_path,
                                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                    get_video_fps(), cv::Size(vis.cols, vis.rows));
                    out_writer_ready = out_writer.isOpened();
                }
                if (out_writer_ready) {
                    out_writer.write(vis);
                }
            } else {
                // 缩小后再显示，降低窗口渲染开销（框/文字仍按原分辨率坐标绘制）。
                cv::Mat display;
                if (vis.cols > 960) {
                    const double scale = 960.0 / static_cast<double>(vis.cols);
                    cv::resize(vis, display, cv::Size(), scale, scale, cv::INTER_AREA);
                } else {
                    display = vis;
                }
                cv::imshow("armor debug", display);
                if (cv::waitKey(1) == 27) {
                    shutdown_requested.store(true, std::memory_order_relaxed);
                }
            }
        }

        // 3. 计算帧间隔和通信延迟，再调用 EKF。即使 observations 为空也必须调用。
        const double time_stamp = frame_timestamp_sec;
        dt = last_time_stamp > 0.0 ? time_stamp - last_time_stamp : 0.0;
        if (dt < 0.0 || !std::isfinite(dt)) {
            dt = 0.0;
        }
        last_time_stamp = time_stamp;
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(frame_time - process_start).count());
        WEB_LOG("frame_dt_ms", dt * 1000.0);

        const std::vector<PredictionResult> predictions =
            ekf_predict(observations, gimbal, time_stamp);
        g_last_predictions = predictions;
        WEB_LOG("vehicle_count", static_cast<double>(predictions.size()));
        if (!predictions.empty()) {
            WEB_LOG("ekf_center_x_m", predictions.front().center_x_m);
            WEB_LOG("ekf_center_y_m", predictions.front().center_y_m);
            WEB_LOG("ekf_radius_m", predictions.front().radius_1_m);
        }
        const PredictionResult primary = selectPrimary(predictions);
        bool prediction_valid = applyPrediction(translator, primary);

        // 学生算法耗时过长时，丢弃本次结果，避免旧控制器数据触发动作。
        if (!frameIsFresh(received_at)) {
            setSafeOutput(translator);
            prediction_valid = false;
        }

        const auto process_end = std::chrono::steady_clock::now();
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(process_end - process_start).count());
        WEB_LOG("process_time_ms", translator.message.latency);
        logPrediction(prediction_valid ? primary : PredictionResult{});

        // [离线信号落盘] 逐帧记录关键量，供离线画曲线交叉验证（meas_w/ekf_w、
        // orientation 锯齿、position_yaw 摆动等）。仅在设置了 RMV_TEST_CSV 时启用。
        if (csv_out.is_open()) {
            double orientation_yaw = 0.0;
            double position_yaw = 0.0;
            double pnp_reproj = -1.0;
            double pnp_dist = 0.0;
            if (!observations.empty()) {
                orientation_yaw = observations.front().armor_orientation_yaw;
                position_yaw = observations.front().armor_yaw;
                pnp_reproj = observations.front().reprojection_error;
                pnp_dist = cv::norm(observations.front().position_gimbal_m);
            }
            // 第一列块描述「主车」（predictions[0]，出现顺序第一辆，身份稳定）。
            const PredictionResult p0 =
                predictions.empty() ? PredictionResult{} : predictions.front();
            csv_out << frame_index << ',' << frame_timestamp_sec << ','
                    << detections.size() << ','
                    << (detections.empty() ? 0 : detections.front().target_id) << ','
                    << pnp_reproj << ',' << pnp_dist << ','
                    << orientation_yaw << ',' << position_yaw << ','
                    << g_meas_omega_rad_s << ','
                    << static_cast<int>(p0.state) << ','
                    << static_cast<int>(p0.tracked_slot) << ','
                    << static_cast<int>(p0.tracked_id) << ','
                    << static_cast<int>(p0.slot_ids[0]) << ','
                    << static_cast<int>(p0.slot_ids[1]) << ','
                    << static_cast<int>(p0.slot_ids[2]) << ','
                    << static_cast<int>(p0.slot_ids[3]) << ','
                    << static_cast<int>(p0.slot_digits[0]) << ','
                    << static_cast<int>(p0.slot_digits[1]) << ','
                    << static_cast<int>(p0.slot_digits[2]) << ','
                    << static_cast<int>(p0.slot_digits[3]) << ','
                    << static_cast<int>(p0.target_id) << ','
                    << p0.center_x_m << ',' << p0.center_y_m << ','
                    << p0.center_velocity_x_m_s << ','
                    << p0.center_velocity_y_m_s << ','
                    << p0.body_yaw << ',' << p0.body_yaw_velocity << ','
                    << p0.radius_1_m << ',' << p0.radius_2_m << ','
                    << p0.target_yaw << ',' << p0.target_pitch << ','
                    << p0.distance_m << ','
                    << static_cast<double>(armor_detect_debug(pic, enemy_color).size()) << ','
                    << (armor_pair_diagnostic().rejected ? 1 : 0) << ','
                    << (armor_pair_diagnostic().valid ? armor_pair_diagnostic().pair_ratio : -1.0)
                    << ',' << predictions.size();
            for (std::size_t v = 1; v <= 3; ++v) {
                const PredictionResult pv =
                    v < predictions.size() ? predictions[v] : PredictionResult{};
                csv_out << ',' << static_cast<int>(pv.tracked_id)
                        << ',' << static_cast<int>(pv.tracked_slot)
                        << ',' << static_cast<int>(pv.slot_ids[0])
                        << ',' << static_cast<int>(pv.slot_ids[1])
                        << ',' << static_cast<int>(pv.slot_ids[2])
                        << ',' << static_cast<int>(pv.slot_ids[3])
                        << ',' << static_cast<int>(pv.slot_digits[0])
                        << ',' << static_cast<int>(pv.slot_digits[1])
                        << ',' << static_cast<int>(pv.slot_digits[2])
                        << ',' << static_cast<int>(pv.slot_digits[3])
                        << ',' << pv.center_x_m << ',' << pv.center_y_m
                        << ',' << pv.body_yaw << ',' << pv.body_yaw_velocity
                        << ',' << pv.radius_1_m << ',' << pv.radius_2_m;
            }
            csv_out << '\n';
        }
        ++frame_index;

        // 4. 将结果写回原 64 字节结构体。
        if (serial_connected) {
            writeFrame(*serialPort, translator);
        }
    }

    // 相机在操作线程内打开，也在此线程结束前显式关闭，避免进程退出阶段
    // 静态析构时再调 MVS SDK 导致段错误。
    if (out_writer_ready) {
        out_writer.release();
    }
    if (csv_out.is_open()) {
        csv_out.close();
    }
    close_camera();
    STOP_WEB_SERVER();
    return nullptr;
}
