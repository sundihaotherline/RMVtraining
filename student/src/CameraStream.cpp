#include "CameraStream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef HAS_MVS_SDK
#include "MvCameraControl.h"
#endif

namespace {

const char* envOrEmpty(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

}  // namespace

CameraStream::~CameraStream() {
    close();
}

bool CameraStream::open() {
    close();

    // 1. 离线视频源（逐帧回放，时间戳取视频自身，供 EKF 用真实 dt）。
    const std::string video_file = envOrEmpty("RMV_TEST_VIDEO");
    if (!video_file.empty()) {
        return openVideoSource(video_file);
    }

    // 2. 离线图片源（便于在没有相机硬件时联调检测/PnP）。
    const std::string image_dir = envOrEmpty("RMV_TEST_IMAGE_DIR");
    const std::string image_file = envOrEmpty("RMV_TEST_IMAGE");
    if (!image_dir.empty()) {
        return openImageSource(image_dir);
    }
    if (!image_file.empty()) {
        return openImageSource(image_file);
    }

    // 3. 真实相机。
#ifdef HAS_MVS_SDK
    return openMvs();
#else
    return false;
#endif
}

bool CameraStream::openVideoSource(const std::string& path) {
    capture_.open(path);
    if (!capture_.isOpened()) {
        capture_.release();
        return false;
    }
    video_fps_ = capture_.get(cv::CAP_PROP_FPS);
    if (!(video_fps_ > 1.0) || !std::isfinite(video_fps_)) {
        video_fps_ = 30.0;  // 兜底：容器未标帧率时按 30fps 回放。
    }
    video_frame_index_ = 0;
    current_ts_ = 0.0;
    eof_ = false;
    mode_ = Mode::Video;
    return true;
}

bool CameraStream::openImageSource(const std::string& path) {
    image_paths_.clear();
    image_index_ = 0;

    // 目录：枚举常见图片后缀。
    const bool is_dir = !path.empty() && path.back() == '/';
    if (is_dir) {
        const char* kExtensions[] = {".bmp", ".png", ".jpg", ".jpeg"};
        for (const char* ext : kExtensions) {
            std::vector<cv::String> files;
            cv::glob(path + "*" + ext, files, false);
            for (const cv::String& f : files) {
                image_paths_.push_back(f);
            }
        }
    } else {
        image_paths_.push_back(path);
    }

    std::sort(image_paths_.begin(), image_paths_.end());
    if (image_paths_.empty()) {
        return false;
    }
    mode_ = Mode::Image;//切换为读图模式
    return true;
}

bool CameraStream::openMvs() {
#ifdef HAS_MVS_SDK
    MV_CC_DEVICE_INFO_LIST device_list{};
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list) != MV_OK ||
        device_list.nDeviceNum == 0) {
           
        return false;
    }

    MV_CC_DEVICE_INFO* device = device_list.pDeviceInfo[0]; //取零号相机
    if (!MV_CC_IsDeviceAccessible(device, MV_ACCESS_Exclusive)) {
        return false;
    }

    void* handle = nullptr;
    if (MV_CC_CreateHandle(&handle, device) != MV_OK) {
        return false;
    }
    if (MV_CC_OpenDevice(handle, MV_ACCESS_Exclusive, 0) != MV_OK) {
        MV_CC_DestroyHandle(handle);
        return false;
    }

    // 连续采集：关闭触发，固定曝光、关闭增益，输出 BayerRG8 彩色。
    // 灯条是强光源，自动曝光/增益会把它们过曝成白色，导致颜色掩膜抓不到；
    // 因此关闭增益并把曝光固定到 4000us。
    MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    MV_CC_SetEnumValue(handle, "ExposureAuto", 0);              // 0 = Off
    MV_CC_SetFloatValue(handle, "ExposureTime", 4000.0F);
    MV_CC_SetEnumValue(handle, "GainAuto", 0);                  // 0 = Off
    MV_CC_SetFloatValue(handle, "Gain", 0.0F);
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BayerRG8);
    // 显式固定输出分辨率 = 标定的 1440x1080，并居中 ROI。相机上电默认可能是全分辨率
    // （带宽被吃满+