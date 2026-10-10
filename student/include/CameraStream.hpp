#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// 相机取流封装，同时支持：
//   1. HIKROBOT MVS 工业相机（编译时定义 HAS_MVS_SDK 时启用）；
//   2. 离线测试：从图片文件/目录循环读取，模拟连续取流。
//
// 离线测试通过环境变量配置：
//   RMV_TEST_IMAGE_DIR  图片目录，按文件名排序循环读取
//   RMV_TEST_IMAGE      单张图片，循环返回同一张
//
// open() 优先使用环境变量指定的图片源；未设置时尝试打开 MVS 相机。
// 取流失败或超时时 grab() 返回 false，由上层 get_pic() 决定降级策略。
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

private:
    enum class Mode { None, Image, Mvs };

    bool openImageSource(const std::string& path);
    bool openMvs();
    bool grabImage(cv::Mat& bgr);
    bool grabMvs(cv::Mat& bgr);

    Mode mode_{Mode::None};

    // 图片源状态。
    std::vector<std::string> image_paths_;
    std::size_t image_index_{0};

    // MVS 状态。
    void* handle_{nullptr};
};
