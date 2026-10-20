#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "CameraStream.hpp"
#include "NumberClassifier.hpp"
#include "RotatingVehicleTracker.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 装甲板物理尺寸（mm，宽 x 高）。见 装甲板尺寸.txt。
constexpr double kSmallArmorWidthM = 0.135;
constexpr double kSmallArmorHeightM = 0.057;
constexpr double kLargeArmorWidthM = 0.230;
constexpr double kLargeArmorHeightM = 0.057;

// PnP 观测拒绝阈值。
constexpr double kMinDepthM = 0.2;
constexpr double kMaxDepthM = 15.0;
constexpr double kMaxReprojectionErrorPx = 6.0;

// 数字分类数据集目录。默认相对当前工作目录，可用环境变量覆盖。
std::string datasetDirectory() {
    const char* env = std::getenv("RMV_DATASET_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return "datasets";
}

// 共享检测器单例：armor_detect 与可视化必须用同一个实例，使 detect() 缓存的
// 灯条、以及时间平滑状态（prev_boxes_）保持一致，避免可视化再跑一遍检测流水线。
ArmorDetector& sharedDetector() {
    static NumberClassifier classifier;
    static bool classifier_attempted = false;
    static ArmorDetector detector;
    if (!classifier_attempted) {
        classifier_attempted = true;
        const bool loaded = classifier.load(datasetDirectory());
        detector.setClassifier(loaded ? &classifier : nullptr);
        std::cerr << "[数字分类] "
                  << (loaded ? "模板加载成功"
                             : "模板加载失败：datasets 目录缺失或 1..9 类别不全，贴纸数字将不显示")
                  << '\n';
    }
    return detector;
}

// 以装甲板中心为原点建立物点：x 向右、y 向下、z 指向相机，单位米。
// 角点顺序与 ArmorDetection::corners 一致（左上、右上、右下、左下）。
std::vector<cv::Point3d> buildObjectPoints(ArmorSize size) {
    const double half_w =
        (size == ArmorSize::Large ? kLargeArmorWidthM : kSmallArmorWidthM) / 2.0;
    const double half_h =
        (size == ArmorSize::Large ? kLargeArmorHeightM : kSmallArmorHeightM) / 2.0;
    return {
        cv::Point3d(-half_w, -half_h, 0.0),  // 左上
        cv::Point3d(half_w, -half_h, 0.0),   // 右上
        cv::Point3d(half_w, half_h, 0.0),    // 右下
        cv::Point3d(-half_w, half_h, 0.0),   // 左下
    };
}

bool cornersInsideImage(const ArmorDetection& detection, const CameraParameters& camera) {
    for (const cv::Point2f& corner : detection.corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y) ||
            corner.x < 0.0F || corner.y < 0.0F ||
            corner.x >= camera.image_width || corner.y >= camera.image_height) {
            return false;
        }
    }
    return true;
}

// 相机取流单例。生命周期显式管理：get_pic 按需打开，close_camera 在操作线程
// 结束前关闭，避免 CameraStream 析构函数在进程退出阶段调用 MVS SDK——
// 此时 SDK 全局状态可能已析构，close() 会直接段错误。
CameraStream g_camera_stream;

}  // namespace

bool get_pic(cv::Mat& pic) {
    // 相机 SDK 的对象和生命周期在 g_camera_stream 中统一管理，取流结束后释放 SDK 缓冲。
    CameraStream& stream = g_camera_stream;
    if (!stream.isOpen()) {
        if (stream.eof()) {
            // 离线视频已播完：不要再 reopen 从头循环，让上层据此退出。
            pic.release();
            return false;
        }
        if (!stream.open()) {
            pic.release();
            return false;
        }
    }

    cv::Mat frame;
    if (!stream.grab(frame) || frame.empty() || frame.channels() != 3) {
        if (stream.isVideo() && stream.eof()) {
            // 视频自然结束：保持句柄不 close，使 is_video_source()/video_source_ended()
            // 仍可查询，主循环据此优雅退出而不是反复 reopen。
            pic.release();
            return false;
        }
        // 取流失败（相机可能已断开）：释放失效句柄并把模式复位，使下一帧重新
        // open() 枚举设备重连，而不是用旧 handle 无限空转。
        stream.close();
        pic.release();
        return false;
    }
    pic = frame;
    return true;
}

void close_camera() {
    g_camera_stream.close();
}

bool is_video_source() {
    return g_camera_stream.isVideo();
}

bool video_source_ended() {
    return g_camera_stream.eof();
}

double get_pic_timestamp_sec() {
    return g_camera_stream.timestampSec();
}

double get_video_fps() {
    return g_camera_stream.fps();
}

std::vector<ArmorDetection> armor_detect(const cv::Mat& pic, TeamColor enemy_color) {
    return sharedDetector().detect(pic, enemy_color);
}

std::vector<ArmorDetector::LightBar> armor_detect_debug(const cv::Mat&,
                                                        TeamColor) {
    // 直接返回 detect() 已提取并缓存的灯条，避免可视化重复跑掩膜/轮廓。
    return sharedDetector().lastBars();
}

ArmorDetector::PairDiagnostic armor_pair_diagnostic() {
    return sharedDetector().lastPairDiagnostic();
}

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                   const CameraParameters& camera,
                                   const GimbalState& gimbal) {
    std::vector<ArmorPose> poses;
    if (!camera.valid()) {
        return poses;
    }

    const cv::Mat r_cg = camera.rotation_camera_to_gimbal;
    const cv::Mat t_cg = camera.translation_camera_to_gimbal;

    for (const ArmorDetection& detection : detections) {
        // 保持帧是上一帧框的拷贝，只用于可视化防闪；不重复喂给 PnP/EKF，
        // 否则会把同一份旧量测重复计数，污染 EKF 的量测更新与 meas_w 曲线。
        if (detection.held) {
            continue;
        }
        if (!cornersInsideImage(detection, camera)) {
            continue;
        }

        const std::vector<cv::Point3d> object_points = buildObjectPoints(detection.size);
        std::vector<cv::Point2f> image_points(detection.corners.begin(),
                                              detection.corners.end());

        cv::Mat rvec, tvec;
        const bool solved =
            cv::solvePnP(object_points, image_points, camera.camera_matrix,
                         camera.distortion_coefficients, rvec, tvec, false,
                         cv::SOLVEPNP_ITERATIVE);
        if (!solved || tvec.rows != 3 || tvec.cols != 1) {
            continue;
        }

        const cv::Vec3d position_camera(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
                                        tvec.at<double>(2, 0));
        const double depth = position_camera[2];
        if (!std::isfinite(depth) || depth < kMinDepthM || depth > kMaxDepthM) {
            continue;
        }

        // 重投影误差：超过阈值视为外点。object_points 为 double，投影结果同为 double。
        std::vector<cv::Point2d> projected;
        cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                          camera.distortion_coefficients, projected);
        double reprojection_error = 0.0;
        for (std::size_t i = 0; i < projected.size(); ++i) {
            const double dx = projected[i].x - image_points[i].x;
            const double dy = projected[i].y - image_points[i].y;
            reprojection_error += std::sqrt(dx * dx + dy * dy);
        }
        reprojection_error /= static_cast<double>(projected.size());
        if (!std::isfinite(reprojection_error) ||
            reprojection_error > kMaxReprojectionErrorPx) {
            continue;
        }

        // 相机坐标系 -> 云台坐标系：p_gimbal = R * p_camera + t。
        cv::Mat position_camera_mat(position_camera);
        const cv::Mat position_gimbal_mat = r_cg * position_camera_mat + t_cg;
        const cv::Vec3d position_gimbal(position_gimbal_mat.at<double>(0, 0),
                                        position_gimbal_mat.at<double>(1, 0),
                                        position_gimbal_mat.at<double>(2, 0));
        if (!cv::checkRange(position_gimbal_mat)) {
            continue;
        }

        // 角点沿用检测器给出的灯条端点矩形，不再把整块装甲板外轮廓反投影回图像，
        // 保证输出框与“只框灯条间部分”的约定一致。
        ArmorPose pose;
        pose.detection = detection;
        pose.position_camera_m = position_camera;
        pose.position_gimbal_m = position_gimbal;
        // 云台坐标系偏航角：假设 x 向右、z 向前。
        pose.armor_yaw = std::atan2(position_gimbal[0], position_gimbal[2]);
        // 板表面法向（物点 z=0，法向 +z 指向相机）转到云台系，得到板朝向的 yaw。
        // 自转时板法向单调旋转，是比位置角更干净的旋转证据。
        cv::Mat R_obj_cam;
        cv::Rodrigues(rvec, R_obj_cam);
        const cv::Mat n_gimbal = r_cg * R_obj_cam.col(2);
        pose.armor_orientation_yaw =
            std::atan2(n_gimbal.at<double>(0, 0), n_gimbal.at<double>(2, 0));
        // 俯仰角：y 向下，故向上为正；对 y 取负，水平距离取 sqrt(x^2+z^2)。
        pose.armor_pitch = std::atan2(-position_gimbal[1],
                                      std::sqrt(position_gimbal[0] * position_gimbal[0] +
                                                position_gimbal[2] * position_gimbal[2]));
        pose.reprojection_error = reprojection_error;
        poses.push_back(pose);
    }
    return poses;
}

PnPDiagnostic armor_solve_debug(const ArmorDetection& detection,
                                const CameraParameters& camera) {
    PnPDiagnostic diag;
    diag.camera_valid = camera.valid();
    if (!diag.camera_valid) {
        return diag;
    }

    diag.corners_inside = cornersInsideImage(detection, camera);
    if (!diag.corners_inside) {
        return diag;
    }

    const std::vector<cv::Point3d> object_points = buildObjectPoints(detection.size);
    const std::vector<cv::Point2f> image_points(detection.corners.begin(),
                                                detection.corners.end());
    cv::Mat rvec, tvec;
    diag.solved =
        cv::solvePnP(object_points, image_points, camera.camera_matrix,
                     camera.distortion_coefficients, rvec, tvec, false,
                     cv::SOLVEPNP_ITERATIVE);
    if (!diag.solved || tvec.rows != 3 || tvec.cols != 1) {
        return diag;
    }

    diag.depth_m = tvec.at<double>(2, 0);
    diag.depth_in_range = std::isfinite(diag.depth_m) && diag.depth_m >= kMinDepthM &&
                          diag.depth_m <= kMaxDepthM;

    std::vector<cv::Point2d> projected;
    cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                      camera.distortion_coefficients, projected);
    double reprojection = 0.0;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        const double dx = projected[i].x - image_points[i].x;
        const double dy = projected[i].y - image_points[i].y;
        reprojection += std::sqrt(dx * dx + dy * dy);
    }
    diag.reprojection_error_px = reprojection / static_cast<double>(projected.size());
    diag.reprojection_ok = std::isfinite(diag.reprojection_error_px) &&
                           diag.reprojection_error_px <= kMaxReprojectionErrorPx;
    return diag;
}

namespace {
// 单车辆跟踪器（简化：默认只有一辆车，只对准它的四块板）。必须跨帧持久：
// ekf_predict 每次只拿到当帧观测，上一帧的状态、协方差与板编号都保存在这个静态
// 对象里，不要做成每次调用新建。
RotatingVehicleTracker g_vehicle_tracker;
}  // namespace

std::vector<PredictionResult> ekf_predict(const std::vector<ArmorPose>& observations,
                                          const GimbalState& gimbal,
                                          double timestamp_seconds) {
    // 简化：只维护一辆车。返回值保持 vector（main 里的绘制/选择/CSV 都按列表处理），
    // 但永远只有一个元素；tracking 标志表示是否已锁定。
    std::vector<PredictionResult> results;
    results.push_back(g_vehicle_tracker.step(observations, gimbal, timestamp_seconds));
    return results;
}
