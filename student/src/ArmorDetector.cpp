#include "ArmorDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---- HSV 颜色阈值（OpenCV H:[0,180], S/V:[0,255]） ----
constexpr int kRedHueLow1 = 0;
constexpr int kRedHueHigh1 = 12;
constexpr int kRedHueLow2 = 156;
constexpr int kRedHueHigh2 = 180;
constexpr int kBlueHueLow = 100;
constexpr int kBlueHueHigh = 124;
constexpr int kColorSaturationMin = 40;
constexpr int kColorValueMin = 30;
// ---- BGR 通道差掩膜阈值 ----
constexpr int kChannelDiffMin = 25;

// ---- 灯条候选过滤 ----
constexpr double kMinLengthRatio = 0.04;   // 相对图像高度
constexpr double kMinAspectRatio = 2.0;    // 长度/宽度
constexpr double kMaxAngleDeg = 30.0;      // 相对竖直方向
constexpr double kMinFillRatio = 0.4;      // 轮廓面积 / 外接矩形面积
constexpr double kMinBarArea = 80.0;

// ---- 左右灯条匹配 ----
constexpr double kMaxLengthRatio = 2.0;    // 长度比
constexpr double kMaxAngleDiffDeg = 10.0;
constexpr double kMinPairWidthRatio = 1.2; // 中心距/平均长度
constexpr double kMaxPairWidthRatio = 6.0;
constexpr double kMaxVerticalOffsetRatio = 0.5; // 中心纵向偏移/平均长度
// 大小装甲板分界：中心距/灯条长度（小 135/57≈2.37，大 230/57≈4.04）。
constexpr double kLargeArmorRatio = 3.0;

// ---- 数字分类 ----
constexpr float kMinNumberConfidence = 0.5F;

double dist(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::norm(a - b);
}

// 将四个角点按“左上 -> 右上 -> 右下 -> 左下”顺时针排序。
// 适用于相对竖直方向倾斜不超过 45° 的四边形。
void orderCornersClockwise(std::array<cv::Point2f, 4>& corners) {
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a,
                                                 const cv::Point2f& b) {
        return (a.y + a.x) < (b.y + b.x);
    });
    // corners[0] = 左上，corners[3] = 右下。
    // 剩余两点用 (y - x) 区分右上（最小）和左下（最大）。
    if (corners[1].y - corners[1].x > corners[2].y - corners[2].x) {
        std::swap(corners[1], corners[2]);
    }
    // 此时顺序为 [左上, 右上, 左下, 右下]，交换后两元素得到顺时针顺序。
    std::swap(corners[2], corners[3]);
}

}  // namespace

cv::Mat ArmorDetector::buildMask(const cv::Mat& bgr, TeamColor enemy_color) const {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat color_mask;
    if (enemy_color == TeamColor::Red) {
        cv::Mat part1, part2;
        cv::inRange(hsv, cv::Scalar(kRedHueLow1, kColorSaturationMin, kColorValueMin),
                    cv::Scalar(kRedHueHigh1, 255, 255), part1);
        cv::inRange(hsv, cv::Scalar(kRedHueLow2, kColorSaturationMin, kColorValueMin),
                    cv::Scalar(kRedHueHigh2, 255, 255), part2);
        color_mask = part1 | part2;
    } else {
        cv::inRange(hsv, cv::Scalar(kBlueHueLow, kColorSaturationMin, kColorValueMin),
                    cv::Scalar(kBlueHueHigh, 255, 255), color_mask);
    }

    // BGR 通道差掩膜：敌色通道明显强于对侧通道，与绝对亮度无关，
    // 对较暗/较远的灯条同样鲁棒（红：R-B，蓝：B-R）。
    std::vector<cv::Mat> bgr_channels;
    cv::split(bgr, bgr_channels);
    cv::Mat channel_diff;
    if (enemy_color == TeamColor::Red) {
        channel_diff = bgr_channels[2] - bgr_channels[0]; // R - B
    } else {
        channel_diff = bgr_channels[0] - bgr_channels[2]; // B - R
    }
    cv::Mat diff_mask;
    cv::threshold(channel_diff, diff_mask, kChannelDiffMin, 255, cv::THRESH_BINARY);

    cv::Mat mask = color_mask & diff_mask;

    // 开运算去噪，闭运算填充灯条内部的过曝亮斑造成的空洞。
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);
    return mask;
}

std::vector<ArmorDetector::LightBar> ArmorDetector::extractLightBars(
    const cv::Mat& mask) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double min_length = kMinLengthRatio * mask.rows;
    std::vector<LightBar> bars;

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinBarArea) {
            continue;
        }

        // 用直线拟合主轴，再投影轮廓点求端点，比 minAreaRect 更稳定。
        cv::Vec4f line;
        cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
        float vx = line[0];
        float vy = line[1];
        const float norm = std::sqrt(vx * vx + vy * vy);
        if (norm < 1e-6F) {
            continue;
        }
        vx /= norm;
        vy /= norm;
        // 让方向指向下方（vy >= 0），使倾角落在 [-90, 90] 而非接近 ±180。
        if (vy < 0.0F) {
            vx = -vx;
            vy = -vy;
        }

        const float x0 = line[2];
        const float y0 = line[3];
        float t_min = std::numeric_limits<float>::max();
        float t_max = -std::numeric_limits<float>::max();
        for (const auto& p : contour) {
            const float t = (p.x - x0) * vx + (p.y - y0) * vy;
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
        }

        LightBar bar;
        bar.length = t_max - t_min;
        if (bar.length < min_length) {
            continue;
        }
        bar.width = static_cast<float>(area / bar.length);
        if (bar.width < 1.0F || bar.length / bar.width < kMinAspectRatio) {
            continue;
        }

        // 角度相对竖直方向；atan2(vx, vy) 在竖直时为 0。
        bar.angle_deg = static_cast<float>(std::atan2(vx, vy) * 180.0 / kPi);
        if (std::abs(bar.angle_deg) > kMaxAngleDeg) {
            continue;
        }

        cv::Point2f p_top(x0 + t_min * vx, y0 + t_min * vy);
        cv::Point2f p_bottom(x0 + t_max * vx, y0 + t_max * vy);
        if (p_top.y > p_bottom.y) {
            std::swap(p_top, p_bottom);
        }
        bar.top = p_top;
        bar.bottom = p_bottom;
        bar.center = cv::Point2f((p_top.x + p_bottom.x) * 0.5F,
                                 (p_top.y + p_bottom.y) * 0.5F);

        const cv::RotatedRect rect = cv::minAreaRect(contour);
        const double rect_area = rect.size.width * rect.size.height;
        bar.fill_ratio = static_cast<float>(rect_area > 0.0 ? area / rect_area : 0.0);
        if (bar.fill_ratio < kMinFillRatio) {
            continue;
        }

        bars.push_back(bar);
    }
    return bars;
}

bool ArmorDetector::buildDetection(const LightBar& left, const LightBar& right,
                                   ArmorDetection& detection) const {
    detection.corners = {left.top, right.top, right.bottom, left.bottom};
    orderCornersClockwise(detection.corners);

    // 图像边界检查：角点必须在图像内或仅略微越界。
    const double avg_length = (left.length + right.length) * 0.5;
    const double pair_ratio = dist(left.center, right.center) / avg_length;
    detection.size = pair_ratio > kLargeArmorRatio ? ArmorSize::Large : ArmorSize::Small;
    return true;
}

bool ArmorDetector::classifyCenter(const cv::Mat& bgr, ArmorDetection& detection) const {
    if (classifier_ == nullptr || !classifier_->loaded()) {
        return false;
    }

    // 透视校正到正视矩形，再裁剪中央数字区域。
    const float top_width = dist(detection.corners[0], detection.corners[1]);
    const float bottom_width = dist(detection.corners[3], detection.corners[2]);
    const float left_height = dist(detection.corners[0], detection.corners[3]);
    const float right_height = dist(detection.corners[1], detection.corners[2]);
    const int warp_w = std::max(1, static_cast<int>(std::max(top_width, bottom_width)));
    const int warp_h = std::max(1, static_cast<int>(std::max(left_height, right_height)));

    const std::array<cv::Point2f, 4> src = {detection.corners[0], detection.corners[1],
                                            detection.corners[2], detection.corners[3]};
    const std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0.0F, 0.0F), cv::Point2f(static_cast<float>(warp_w - 1), 0.0F),
        cv::Point2f(static_cast<float>(warp_w - 1), static_cast<float>(warp_h - 1)),
        cv::Point2f(0.0F, static_cast<float>(warp_h - 1))};

    const cv::Mat transform = cv::getPerspectiveTransform(src.data(), dst.data());
    cv::Mat warped;
    cv::warpPerspective(bgr, warped, transform, cv::Size(warp_w, warp_h));

    // 数字占据装甲板中央区域。
    const cv::Rect roi(static_cast<int>(warp_w * 0.22), static_cast<int>(warp_h * 0.12),
                       static_cast<int>(warp_w * 0.56), static_cast<int>(warp_h * 0.76));
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > warp_w ||
        roi.y + roi.height > warp_h) {
        return false;
    }

    float confidence = 0.0F;
    const int number = classifier_->classify(warped(roi), confidence);
    if (number < 1 || number > 5 || confidence < kMinNumberConfidence) {
        return false;
    }
    detection.target_id = number;
    detection.confidence = confidence;
    return true;
}

std::vector<ArmorDetection> ArmorDetector::detect(const cv::Mat& bgr,
                                                  TeamColor enemy_color) const {
    std::vector<ArmorDetection> results;
    if (bgr.empty()) {
        return results;
    }

    const cv::Mat mask = buildMask(bgr, enemy_color);
    std::vector<LightBar> bars = extractLightBars(mask);
    if (bars.size() < 2) {
        return results;
    }

    // 两两匹配并打分，用于简单的非极大抑制。
    struct Candidate {
        ArmorDetection detection;
        double score;
        int left;
        int right;
    };
    std::vector<Candidate> candidates;

    for (std::size_t i = 0; i < bars.size(); ++i) {
        for (std::size_t j = i + 1; j < bars.size(); ++j) {
            LightBar a = bars[i];
            LightBar b = bars[j];
            // 以 x 坐标区分左右。
            if (a.center.x > b.center.x) {
                std::swap(a, b);
            }

            const double avg_length = (a.length + b.length) * 0.5;
            const double length_ratio =
                std::max(a.length, b.length) / std::min(a.length, b.length);
            const double angle_diff = std::abs(a.angle_deg - b.angle_deg);
            const double center_dist = dist(a.center, b.center);
            const double pair_ratio = center_dist / avg_length;
            const double vertical_offset =
                std::abs(a.center.y - b.center.y) / avg_length;

            if (length_ratio > kMaxLengthRatio || angle_diff > kMaxAngleDiffDeg ||
                pair_ratio < kMinPairWidthRatio || pair_ratio > kMaxPairWidthRatio ||
                vertical_offset > kMaxVerticalOffsetRatio) {
                continue;
            }

            ArmorDetection detection;
            buildDetection(a, b, detection);
            const bool classified = classifyCenter(bgr, detection);

            // 有分类器时只有有效编号才算目标；无分类器时保留候选但编号未知。
            if (classifier_ != nullptr && classifier_->loaded() && !classified) {
                continue;
            }

            // 分数：越接近标准装甲板宽高比、灯条越等长、倾角越一致越好。
            const double score =
                -(std::abs(pair_ratio - 2.4) + std::abs(length_ratio - 1.0) +
                  angle_diff / 10.0);
            candidates.push_back({detection, score, static_cast<int>(i),
                                  static_cast<int>(j)});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                  return lhs.score > rhs.score;
              });

    // 贪心非极大抑制：同一灯条只属于一个装甲板。
    std::vector<bool> used(bars.size(), false);
    for (const auto& candidate : candidates) {
        if (used[candidate.left] || used[candidate.right]) {
            continue;
        }
        used[candidate.left] = true;
        used[candidate.right] = true;
        results.push_back(candidate.detection);
    }
    return results;
}
