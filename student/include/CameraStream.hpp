#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// 相机取流封装，同时支持：
//   1. HIKROBOT MVS 工业相机（编译时定义 HAS_MVS_SDK 时启用）；
//   2. 离线测试：从图片文件/目录循环读取，模拟连续取流；
//   3. 离线视频：从 AVI/MP4 视频逐帧读取，用于无相机时联调整条流水线。
//
// 离线测试通过环境变量配置：
//   RMV_TEST_IMAGE_DIR  图片目录，按文件名排序循环读取
//   RMV_TEST_IMAGE      单张图片，循环返回同一张
//   RMV_TEST_VIDEO      视频文件，按视频帧率逐帧读取，读完后 eof() 为 true
//
// open() 优先使用环境变量指定的离线源（视频 > 图片目录 > 单图）；未设置时尝试
// 打开 MVS 相机。取流失败或超时时 grab() 返回 false，由上层 get_pic() 决定降级策略。
class CameraStream {
public:
    CameraStream() = default;
    ~CameraStream();

    CameraStream(const CameraStream&) = delete;
    CameraStream& operator=(const CameraStream&) = delete;

    bool open();
    bool isOpen() const { return mode_ != Mode::None; }
    // 获取一帧 BGR 图像。失败返回 false 且不修改 bgr。
    bool grab(cv::Mat& bgr);
    void close();

    // 视频源状态（离线视频模式专用，供上层用真实视频时间戳跑 EKF/显示 ω）。
    bool isVideo() const { return mode_ == Mode::Video; }
    bool eof() const { return eof_; }
    double fps() const { return video_fps_; }
    // 当前帧在视频时间轴上的时间戳（秒）。逐帧单调递增，供 EKF 计算 dt。
    double timestampSec() const { return current_ts_; }

private:
    enum class Mode { None, Image, Mvs, Video };

    bool openImageSource(const std::string& path);
    bool openVideoSource(const std::string& path);
    bool openMvs();
    bool grabImage(cv::Mat& bgr);
    bool grabVideo(cv::Mat& bgr);
    bool grabMvs(cv::Mat& bgr);

    Mode mode_{Mode::None};

    // 图片源状态。
    std::vector<std::string> image_paths_;
    std::size_t image_index_{0};

    // 视频源状态。
    cv::VideoCapture capture_;
    double video_fps_{0.0};
    long long video_frame_index_{0};
    double current_ts_{0.0};
    bool eof_{false};

    // MVS 状态。
    void* handle_{nullptr};
};
