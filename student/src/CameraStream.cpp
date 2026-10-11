#include "CameraStream.hpp"

#include <algorithm>
#include <cstdlib>

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

    // 1. 离线图片源（便于在没有相机硬件时联调检测/PnP）。
    const std::string image_dir = envOrEmpty("RMV_TEST_IMAGE_DIR");
    const std::string image_file = envOrEmpty("RMV_TEST_IMAGE");
    if (!image_dir.empty()) {
        return openImageSource(image_dir);
    }
    if (!image_file.empty()) {
        return openImageSource(image_file);
    }

    // 2. 真实相机。
#ifdef HAS_MVS_SDK
    return openMvs();
#else
    return false;
#endif
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
    mode_ = Mode::Image;
    return true;
}

bool CameraStream::openMvs() {
#ifdef HAS_MVS_SDK
    MV_CC_DEVICE_INFO_LIST device_list{};
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list) != MV_OK ||
        device_list.nDeviceNum == 0) {
        return false;
    }

    MV_CC_DEVICE_INFO* device = device_list.pDeviceInfo[0];
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
    MV_CC_SetEnumValue(handle, "ExposureAuto", 0);  // 0 = Off
    MV_CC_SetFloatValue(handle, "ExposureTime", 4000.0F);
    MV_CC_SetEnumValue(handle, "GainAuto", 0);      // 0 = Off
    MV_CC_SetFloatValue(handle, "Gain", 0.0F);
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BayerRG8);

    if (MV_CC_StartGrabbing(handle) != MV_OK) {
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        return false;
    }

    handle_ = handle;
    mode_ = Mode::Mvs;
    return true;
#else
    (void)0;
    return false;
#endif
}

bool CameraStream::grabImage(cv::Mat& bgr) {
    if (image_paths_.empty()) {
        return false;
    }
    const std::string& path = image_paths_[image_index_];
    image_index_ = (image_index_ + 1) % image_paths_.size();

    const cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
    if (frame.empty()) {
        return false;
    }
    bgr = frame;
    return true;
}

bool CameraStream::grabMvs(cv::Mat& bgr) {
#ifdef HAS_MVS_SDK
    if (handle_ == nullptr) {
        return false;
    }
    MV_FRAME_OUT frame{};
    if (MV_CC_GetImageBuffer(handle_, &frame, 1000) != MV_OK) {
        return false;
    }

    const MV_FRAME_OUT_INFO_EX& info = frame.stFrameInfo;
    const int width = info.nWidth;
    const int height = info.nHeight;
    const unsigned char* data = frame.pBufAddr;

    cv::Mat converted;
    switch (info.enPixelType) {
        case PixelType_Gvsp_Mono8: {
            const cv::Mat gray(height, width, CV_8UC1, const_cast<unsigned char*>(data));
            cv::cvtColor(gray, converted, cv::COLOR_GRAY2BGR);
            break;
        }
        case PixelType_Gvsp_BayerRG8: {
            // 该型号相机报告的 RG 相位实际对应 OpenCV 的 BG，直接按 BG 解拜耳，
            // 否则 R/B 通道互换（红色灯条会被显示成蓝色）。
            const cv::Mat bayer(height, width, CV_8UC1, const_cast<unsigned char*>(data));
            cv::cvtColor(bayer, converted, cv::COLOR_BayerBG2BGR);
            break;
        }
        case PixelType_Gvsp_BayerGB8: {
            const cv::Mat bayer(height, width, CV_8UC1, const_cast<unsigned char*>(data));
            cv::cvtColor(bayer, converted, cv::COLOR_BayerGB2BGR);
            break;
        }
        case PixelType_Gvsp_BayerGR8: {
            const cv::Mat bayer(height, width, CV_8UC1, const_cast<unsigned char*>(data));
            cv::cvtColor(bayer, converted, cv::COLOR_BayerGR2BGR);
            break;
        }
        case PixelType_Gvsp_BayerBG8: {
            const cv::Mat bayer(height, width, CV_8UC1, const_cast<unsigned char*>(data));
            cv::cvtColor(bayer, converted, cv::COLOR_BayerBG2BGR);
            break;
        }
        case PixelType_Gvsp_RGB8_Packed: {
            const cv::Mat rgb(height, width, CV_8UC3, const_cast<unsigned char*>(data));
            cv::cvtColor(rgb, converted, cv::COLOR_RGB2BGR);
            break;
        }
        case PixelType_Gvsp_BGR8_Packed: {
            converted = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char*>(data));
            break;
        }
        default:
            MV_CC_FreeImageBuffer(handle_, &frame);
            return false;
    }

    MV_CC_FreeImageBuffer(handle_, &frame);
    if (converted.empty()) {
        return false;
    }
    bgr = converted.clone();
    return true;
#else
    (void)bgr;
    return false;
#endif
}

bool CameraStream::grab(cv::Mat& bgr) {
    switch (mode_) {
        case Mode::Image:
            return grabImage(bgr);
        case Mode::Mvs:
            return grabMvs(bgr);
        default:
            return false;
    }
}

void CameraStream::close() {
#ifdef HAS_MVS_SDK
    if (handle_ != nullptr) {
        MV_CC_StopGrabbing(handle_);
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
#endif
    image_paths_.clear();
    image_index_ = 0;
    mode_ = Mode::None;
}
