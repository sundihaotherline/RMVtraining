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
    MV_CC_SetFloatValue(handle, "ExposureTime", 6000.0F);
    MV_CC_SetEnumValue(handle, "GainAuto", 0);                  // 0 = Off
    MV_CC_SetFloatValue(handle, "Gain", 0.0F);
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BayerRG8);
    // 显式固定输出分辨率 = 标定的 1440x1080，并居中 ROI。相机上电默认可能是全分辨率
    // （带宽被吃满 → 帧率掉到 ~24）；固定到标定分辨率后带宽需求下降，帧率才能回得去。
    MVCC_INTVALUE_EX wmax{}, hmax{};
    if (MV_CC_GetIntValueEx(handle, "Width", &wmax) == MV_OK &&
        MV_CC_GetIntValueEx(handle, "Height", &hmax) == MV_OK) {
        std::cerr << "[相机] 传感器最大分辨率 = " << wmax.nMax << " x " << hmax.nMax << "\n";
        if (wmax.nMax >= 1440 && hmax.nMax >= 1080) {
            MV_CC_SetIntValueEx(handle, "OffsetX", (wmax.nMax - 1440) / 2);
            MV_CC_SetIntValueEx(handle, "OffsetY", (hmax.nMax - 1080) / 2);
        }
    }
    MV_CC_SetIntValueEx(handle, "Width", 1440);
    MV_CC_SetIntValueEx(handle, "Height", 1080);
    MVCC_INTVALUE_EX wset{}, hset{};
    if (MV_CC_GetIntValueEx(handle, "Width", &wset) == MV_OK &&
        MV_CC_GetIntValueEx(handle, "Height", &hset) == MV_OK) {
        std::cerr << "[相机] 分辨率设置后 = " << wset.nCurValue << " x " << hset.nCurValue << "\n";
    }
    MVCC_FLOATVALUE fps_res{};
    if (MV_CC_GetFloatValue(handle, "ResultingFrameRate", &fps_res) == MV_OK) {
        std::cerr << "[相机] 设分辨率后可达帧率 = " << fps_res.fCurValue << " fps\n";
    }
    // 显式固定采集帧率 170fps（与离线素材 166fps 一致）。曝光 4000us 远小于帧周期
    // ~5.9ms，不会互相冲突。
    MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 170.0F);

    // 读回相机实际生效值，判断 170fps 是否真的生效：曝光偏大说明自动曝光没关；
    // ResultingFrameRate 偏低说明被带宽/曝光/传感器限制住了。
    MVCC_FLOATVALUE exposure{};
    if (MV_CC_GetFloatValue(handle, "ExposureTime", &exposure) == MV_OK) {
        std::cerr << "[相机] 实际曝光 = " << exposure.fCurValue << " us\n";
    }
    MVCC_FLOATVALUE fps{};
    if (MV_CC_GetFloatValue(handle, "ResultingFrameRate", &fps) == MV_OK) {
        std::cerr << "[相机] 实际可达帧率 = " << fps.fCurValue << " fps\n";
    } else if (MV_CC_GetFloatValue(handle, "AcquisitionFrameRate", &fps) == MV_OK) {
        std::cerr << "[相机] 帧率设置读回 = " << fps.fCurValue << " fps\n";
    }

    // GigE 链路带宽：之前能到 170、现在只有 ~24，多半是相机的 DeviceLinkThroughputLimit
    // 用了保守默认值（MVS 客户端能到 170 就是因为它会把吞吐放开）。这里读回当前值、
    // 放开到该节点的最大值，再重读 ResultingFrameRate 确认是否回升。
    MVCC_FLOATVALUE thr{};
    if (MV_CC_GetFloatValue(handle, "DeviceLinkThroughputLimit", &thr) == MV_OK) {
        std::cerr << "[相机] 吞吐限制(改前) = " << thr.fCurValue
                  << " B/s（范围 " << thr.fMin << " ~ " << thr.fMax << "）\n";
        MV_CC_SetFloatValue(handle, "DeviceLinkThroughputLimit", thr.fMax);
        MVCC_FLOATVALUE thr2{};
        if (MV_CC_GetFloatValue(handle, "DeviceLinkThroughputLimit", &thr2) == MV_OK) {
            std::cerr << "[相机] 吞吐限制(改后) = " << thr2.fCurValue << " B/s\n";
        }
        MVCC_FLOATVALUE fps2{};
        if (MV_CC_GetFloatValue(handle, "ResultingFrameRate", &fps2) == MV_OK) {
            std::cerr << "[相机] 放开吞吐后可达帧率 = " << fps2.fCurValue << " fps\n";
        }
    } else {
        std::cerr << "[相机] 无 DeviceLinkThroughputLimit 节点（可能是 USB 相机或命名不同）\n";
    }
    MVCC_FLOATVALUE pkt{};
    if (MV_CC_GetFloatValue(handle, "GevSCPSPacketSize", &pkt) == MV_OK) {
        std::cerr << "[相机] GigE 包大小 = " << pkt.fCurValue << " B（网卡 MTU 需大于该值）\n";
    }

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

bool CameraStream::grabVideo(cv::Mat& bgr) {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        eof_ = true;
        return false;
    }
    // 时间戳由帧序号/帧率自算，避免依赖容器里可能缺失或抖动的 POS_MSEC。
    current_ts_ = static_cast<double>(video_frame_index_) / video_fps_;
    ++video_frame_index_;
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

    // 打印一次实际输出分辨率/像素格式，判断「帧率上不去」是否因为相机默认输出
    // 全分辨率（openMvs 里没有显式设置 Width/Height/ROI，分辨率取决于相机上电默认值）。
    static bool res_logged = false;
    if (!res_logged) {
        res_logged = true;
        std::cerr << "[相机] 实际输出分辨率 = " << width << " x " << height
                  << "，像素格式枚举 = " << info.enPixelType << "\n";
    }

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
        case Mode::Video:
            return grabVideo(bgr);
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
    if (capture_.isOpened()) {
        capture_.release();
    }
    image_paths_.clear();
    image_index_ = 0;
    mode_ = Mode::None;
}
