#include "AppSupport.hpp"
#include "SerialPort.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    if (result.state == TrackingState::NoTarget) {
        return;
    }
    WEB_LOG("target_id", result.target_id);
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

        const auto frame_time = std::chrono::steady_clock::now();
        double camera_fps = 0.0;
        if (last_frame_time.time_since_epoch().count() != 0) {
            const double frame_seconds =
                std::chrono::duration<double>(frame_time - last_frame_time).count();
            if (frame_seconds > 0.0) {
                camera_fps = 1.0 / frame_seconds;
                WEB_LOG("camera_fps", camera_fps);
            }
        }
        last_frame_time = frame_time;

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
        } else {
            WEB_LOG("pnp_reprojection_error", -1.0);
        }

        // [离线调试] 实时框选可视化，ESC 退出。按 ~30fps 节流刷新窗口，
        // 其余帧跳过 imshow/重绘，避免把处理流水线拖慢。
        if (!serial_connected && !pic.empty() &&
            std::chrono::steady_clock::now() - last_viz_time >=
                std::chrono::milliseconds(33)) {
            last_viz_time = std::chrono::steady_clock::now();
            cv::Mat vis = pic.clone();
            // 灯条级中间结果：即使没配成装甲板也画出来，便于判断检测器工作状态。
            const auto bars = armor_detect_debug(pic, enemy_color);
            for (const auto& b : bars) {
                cv::line(vis, b.top, b.bottom, cv::Scalar(0, 255, 255), 2);
                cv::circle(vis, b.center, 4, cv::Scalar(0, 255, 255), -1);
            }
            for (const ArmorDetection& d : detections) {
                std::vector<cv::Point> pts;
                for (const cv::Point2f& c : d.corners) {
                    pts.emplace_back(static_cast<int>(std::lround(c.x)),
                                     static_cast<int>(std::lround(c.y)));
                }
                cv::polylines(vis, std::vector<std::vector<cv::Point>>{pts}, true,
                              cv::Scalar(0, 255, 0), 2);
                // 四个角点画成实心圆点。
                for (const cv::Point2f& c : d.corners) {
                    cv::circle(vis, cv::Point(static_cast<int>(std::lround(c.x)),
                                              static_cast<int>(std::lround(c.y))),
                               5, cv::Scalar(0, 0, 255), -1);
                }
                const cv::Point2f center = (d.corners[0] + d.corners[1] +
                                            d.corners[2] + d.corners[3]) * 0.25F;
                cv::putText(vis, std::to_string(d.target_id),
                            center + cv::Point2f(4, 0), cv::FONT_HERSHEY_SIMPLEX,
                            0.9, cv::Scalar(0, 255, 0), 2);
            }
            // PnP 位姿读数：每块装甲板标注距离、云台系偏航/俯仰角与 (x,y,z) 坐标。
            for (const ArmorPose& p : observations) {
                const cv::Point2f center = (p.detection.corners[0] +
                                            p.detection.corners[1] +
                                            p.detection.corners[2] +
                                            p.detection.corners[3]) * 0.25F;
                char pose[128];
                std::snprintf(pose, sizeof(pose), "d=%.2fm yaw=%.1fdeg pitch=%.1fdeg",
                              cv::norm(p.position_gimbal_m),
                              p.armor_yaw * 180.0 / CV_PI,
                              p.armor_pitch * 180.0 / CV_PI);
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

        // 3. 计算帧间隔和通信延迟，再调用 EKF。即使 observations 为空也必须调用。
        const double time_stamp =
            std::chrono::duration<double>(frame_time.time_since_epoch()).count();
        dt = last_time_stamp > 0.0 ? time_stamp - last_time_stamp : 0.0;
        if (dt < 0.0 || !std::isfinite(dt)) {
            dt = 0.0;
        }
        last_time_stamp = time_stamp;
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(frame_time - process_start).count());
        WEB_LOG("frame_dt_ms", dt * 1000.0);

        const PredictionResult prediction = ekf_predict(observations, gimbal, time_stamp);
        bool prediction_valid = applyPrediction(translator, prediction);

        // 学生算法耗时过长时，丢弃本次结果，避免旧控制器数据触发动作。
        if (!frameIsFresh(received_at)) {
            setSafeOutput(translator);
            prediction_valid = false;
        }

        const auto process_end = std::chrono::steady_clock::now();
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(process_end - process_start).count());
        WEB_LOG("process_time_ms", translator.message.latency);
        logPrediction(prediction_valid ? prediction : PredictionResult{});

        // 4. 将结果写回原 64 字节结构体。
        if (serial_connected) {
            writeFrame(*serialPort, translator);
        }
    }

    // 相机在操作线程内打开，也在此线程结束前显式关闭，避免进程退出阶段
    // 静态析构时再调 MVS SDK 导致段错误。
    close_camera();
    STOP_WEB_SERVER();
    return nullptr;
}
