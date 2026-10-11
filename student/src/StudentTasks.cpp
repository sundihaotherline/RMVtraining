#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "CameraStream.hpp"
#include "NumberClassifier.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
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

}  // namespace

bool get_pic(cv::Mat& pic) {
    // 相机 SDK 的对象和生命周期在此函数内管理，取流结束后释放 SDK 缓冲。
    static CameraStream stream;
    if (!stream.isOpen()) {
        if (!stream.open()) {
            pic.release();
            return false;
        }
    }

    cv::Mat frame;
    if (!stream.grab(frame) || frame.empty() || frame.channels() != 3) {
        pic.release();
        return false;
    }
    pic = frame;
    return true;
}

std::vector<ArmorDetection> armor_detect(const cv::Mat& pic, TeamColor enemy_color) {
    static NumberClassifier classifier;
    static bool classifier_attempted = false;
    static ArmorDetector detector;

    if (!classifier_attempted) {
        classifier_attempted = true;
        classifier.load(datasetDirectory());
        detector.setClassifier(classifier.loaded() ? &classifier : nullptr);
    }

    return detector.detect(pic, enemy_color);
}

std::vector<ArmorDetector::LightBar> armor_detect_debug(const cv::Mat& pic,
                                                        TeamColor enemy_color) {
    static NumberClassifier classifier;
    static bool classifier_attempted = false;
    static ArmorDetector detector;

    if (!classifier_attempted) {
        classifier_attempted = true;
        classifier.load(datasetDirectory());
        detector.setClassifier(classifier.loaded() ? &classifier : nullptr);
    }

    return detector.debugExtractBars(pic, enemy_color);
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

        // 用已知物理尺寸把整块装甲板矩形反投影回图像，得到精准完整角点：
        // solvePnP 求出的位姿结合 object_points 的完整宽高，能正则化掉
        // 灯条端点截短造成的框偏小，保证角点覆盖整块板（中央数字完整）。
        // 精度取决于标定内参；内参未标定时只起到矩形正则化作用。
        std::vector<cv::Point2d> refined_corners;
        cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                          camera.distortion_coefficients, refined_corners);
        ArmorDetection refined_detection = detection;
        if (refined_corners.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                refined_detection.corners[i] =
                    cv::Point2f(static_cast<float>(refined_corners[i].x),
                                static_cast<float>(refined_corners[i].y));
            }
        }

        ArmorPose pose;
        pose.detection = refined_detection;
        pose.position_camera_m = position_camera;
        pose.position_gimbal_m = position_gimbal;
        // 云台坐标系偏航角：假设 x 向右、z 向前。
        pose.armor_yaw = std::atan2(position_gimbal[0], position_gimbal[2]);
        pose.reprojection_error = reprojection_error;
        poses.push_back(pose);
    }
    return poses;
}

PredictionResult ekf_predict(const std::vector<ArmorPose>&, const GimbalState&, double) {
    // TODO(student)：旋转车体 EKF 跟踪与预测属于后续步骤，暂保持无目标安全输出。
    return {};
}
