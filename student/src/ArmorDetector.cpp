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
constexpr int kColorSaturationMin = 80;
constexpr int kColorValueMin = 30;
// ---- BGR 通道差掩膜阈值 ----
constexpr int kChannelDiffMin = 40;

// ---- 灯条候选过滤 ----
constexpr double kMinLengthRatio = 0.04;   // 相对图像高度
constexpr double kMinAspectRatio = 2.0;    // 长度/宽度
constexpr double kMaxAngleDeg = 60.0;      // 相对竖直方向：允许装甲板绕中心 roll 到 60°
constexpr double kMinFillRatio = 0.4;      // 轮廓面积 / 外接矩形面积
constexpr double kMinBarArea = 40.0;       // 侧对时灯条变窄，放宽最小面积
// 灯条端点晕光裁剪：沿主轴采样颜色强度，强度低于该灯条峰值强度的
// 该比例处视为 bloom 边界。近端饱和 LED 的 bloom 更亮、更拖尾，
// 用绝对阈值会把远端真灯条一起误伤，故按每根灯条自身峰值做相对裁剪。
constexpr double kBarTrimStrengthRatio = 0.6;

// ---- 左右灯条匹配 ----
// 配对条件整体收紧：多灯条场景下避免把不同装甲板的灯条错配成一块，
// 只在几何上足够吻合时才生成候选框。
constexpr double kMaxLengthRatio = 1.6;      // 长度比：同板两灯条长度应接近
constexpr double kMaxAngleDiffDeg = 12.0;    // 角度差：两灯条应近乎平行
// 中心距/平均长度。下限收到 1.0（侧对时两灯条在图像上靠得更近，但不允许近到
// 几乎重叠）；上限收到 4.5（大装甲 230/57≈4.04），避免把相邻装甲板灯条错配。
constexpr double kMinPairWidthRatio = 1.0;
constexpr double kMaxPairWidthRatio = 4.5;
constexpr double kMaxAlongAxisOffsetRatio = 0.35; // 中心沿灯条主轴偏移/平均长度
// 大小装甲板分界：中心距/灯条长度（小 135/57≈2.37，大 230/57≈4.04）。
constexpr double kLargeArmorRatio = 3.0;// 取中间值，避免小装甲板被误判为大装甲板。

// ---- 数字分类 ----
constexpr float kMinNumberConfidence = 0.5F;//图像贴纸与数据集相似度底线

// ---- 时间平滑（EMA）----
constexpr float kSmoothAlpha = 0.6F;   // 当前帧权重；上一帧权重 = 1 - alpha
constexpr float kAssocDistPx = 60.0F;  // 相邻帧同目标关联的最大中心位移（像素）

double dist(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::norm(a - b);
}

// 装甲板框中心（四角均值）。
cv::Point2f boxCenter(const ArmorDetection& d) {
    cv::Point2f c(0.0F, 0.0F);
    for (const cv::Point2f& p : d.corners) {
        c += p;
    }
    return c * 0.25F;
}

// 将四个角点按“左上 -> 右上 -> 右下 -> 左下”顺时针排序。
// 用绕质心的极角排序，对任意 roll 倾斜角都稳健，不再要求倾斜 ≤45°。
void orderCornersClockwise(std::array<cv::Point2f, 4>& corners) {
    cv::Point2f center(0.0F, 0.0F);
    for (const cv::Point2f& p : corners) {
        center += p;
    }
    center *= 0.25F;

    // 图像坐标 y 向下，atan2 角度递增方向即顺时针方向，得到凸四边形环序。
    std::sort(corners.begin(), corners.end(),
              [&center](const cv::Point2f& a, const cv::Point2f& b) {
                  return std::atan2(a.y - center.y, a.x - center.x) <
                         std::atan2(b.y - center.y, b.x - center.x);
              });

    // 以“左上”角（x+y 最小）为起点，其余沿顺时针环序跟随。
    const auto top_left = std::min_element(
        corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return (a.x + a.y) < (b.x + b.y);
        });
    std::rotate(corners.begin(), top_left, corners.end());
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
    // 各向同性方形核：灯条是一排 LED，LED 间有几像素暗缝，相机滚转使灯条在
    // 画面里倾斜时，这些暗缝方向也随之倾斜，竖直细长核（3x15）桥接不到、
    // 灯条被切成一堆短碎片被 min_length 砍掉。方形核任意方向都能桥接 LED 暗缝；
    // 左右两根灯条中心距在有效距离内都远大于 9px，不会把两根粘成一体。
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);
    return mask;
}

cv::Mat ArmorDetector::buildStrength(const cv::Mat& bgr) const {
    // 亮度图 V = max(R,G,B)。灯条过曝时核心发白（R≈G≈B），若用颜色差
    // (R-B) 会在白核处掉到 0，把远端真灯条误裁成 0；亮度在 LED 核心最亮、
    // 向两端 bloom 渐暗，是判断真实灯条边界的更稳指标。
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);
    return cv::max(ch[0], cv::max(ch[1], ch[2]));
}

std::vector<ArmorDetector::LightBar> ArmorDetector::extractLightBars(
    const cv::Mat& mask, const cv::Mat& strength) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double min_length = kMinLengthRatio * mask.rows;
    std::vector<LightBar> bars;

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinBarArea) {
            continue;
        }

        // 主轴方向用轮廓二阶中心矩（PCA）估计，比 fitLine / minAreaRect 对
        // 轮廓边界噪声与单侧 bloom 更稳，避免侧对时近端灯条主轴被带歪。
        const cv::Moments m = cv::moments(contour);
        if (m.m00 < 1e-6) {
            continue;
        }
        const double angle = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02);
        float vx = static_cast<float>(std::cos(angle));
        float vy = static_cast<float>(std::sin(angle));
        // 让方向指向下方（vy >= 0），使倾角落在 [-90, 90] 而非接近 ±180。
        if (vy < 0.0F) {
            vx = -vx;
            vy = -vy;
        }

        const float x0 = static_cast<float>(m.m10 / m.m00);
        const float y0 = static_cast<float>(m.m01 / m.m00);
        float t_min = std::numeric_limits<float>::max();
        float t_max = -std::numeric_limits<float>::max();
        for (const auto& p : contour) {
            const float t = (p.x - x0) * vx + (p.y - y0) * vy;
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
        }

        // 晕光裁剪：近端过曝 LED 的 bloom 会沿主轴向外拖尾，把灯条长度撑大。
        // 先沿主轴扫一遍亮度峰值，再从两端向内找亮度首次达到峰值 * kBarTrimStrengthRatio
        // 处，即为真实灯条边界，剪掉外侧 bloom。按每根灯条自身峰值取相对阈值，
        // 避免绝对阈值误伤较暗的远端灯条；用亮度而非颜色差，避免过曝白核被误裁。
        if (!strength.empty()) {
            const auto sample = [&](float t) -> float {
                const int px = static_cast<int>(std::lround(x0 + t * vx));
                const int py = static_cast<int>(std::lround(y0 + t * vy));
                if (px < 0 || py < 0 || px >= strength.cols || py >= strength.rows) {
                    return 0.0F;
                }
                return static_cast<float>(strength.at<uchar>(py, px));
            };

            float peak = 0.0F;
            for (float t = t_min; t <= t_max; t += 1.0F) {
                peak = std::max(peak, sample(t));
            }
            if (peak > 1.0F) {
                const float cutoff = peak * static_cast<float>(kBarTrimStrengthRatio);

                float t_top = t_min;
                for (float t = t_min; t <= 0.0F; t += 1.0F) {
                    if (sample(t) >= cutoff) {
                        t_top = t;
                        break;
                    }
                }
                float t_bottom = t_max;
                for (float t = t_max; t >= 0.0F; t -= 1.0F) {
                    if (sample(t) >= cutoff) {
                        t_bottom = t;
                        break;
                    }
                }
                t_min = t_top;
                t_max = t_bottom;
            }
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
    // 角点直接取两根灯条的四个端点：框只覆盖灯条围出的矩形区域，
    // 不再向外扩出整块装甲板（含数字/板面）的完整外轮廓。
    detection.corners = {left.top, right.top, right.bottom, left.bottom};
    orderCornersClockwise(detection.corners);

    // 大小装甲板分界：中心距 / 平均灯条长度。
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
    // 有效编号扩到 1..9：1..5 是装甲板，6/7/8/9 是前哨/哨兵/基地/负样本，
    // 都作为已识别目标如实填写编号；只有未识别或低置信度才返回 false。
    if (number < 1 || number > 9 || confidence < kMinNumberConfidence) {
        return false;
    }
    detection.target_id = number;
    detection.confidence = confidence;
    return true;
}

std::vector<ArmorDetection> ArmorDetector::detect(const cv::Mat& bgr,
                                                  TeamColor enemy_color) {
    std::vector<ArmorDetection> results;
    if (bgr.empty()) {
        return results;
    }

    const cv::Mat mask = buildMask(bgr, enemy_color);
    const cv::Mat strength = buildStrength(bgr);
    std::vector<LightBar> bars = extractLightBars(mask, strength);

    if (bars.size() < 2) {
        return results;
    }

    // 第一步：先框出目标。仅用几何信息两两匹配生成候选框，再打分做非极大抑制。
    // 分类不再参与候选生成，避免“灯条识别得了但贴纸分类失败就把框丢掉”的问题。
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

            // 灯条主轴单位向量（top->bottom）。真实场景灯条可能倾斜，
            // 用“沿轴偏移 / 垂直轴偏移”取代固定竖直方向的比较，保证旋转不变。
            cv::Point2f axis = a.bottom - a.top;
            const float axis_len = cv::norm(axis);
            if (axis_len < 1e-3F) {
                continue;
            }
            axis.x /= axis_len;
            axis.y /= axis_len;
            const cv::Point2f center_vec = b.center - a.center;
            // 沿灯条方向的分量：两灯条应近乎等高（越小越好）。
            const double along_axis = std::abs(center_vec.x * axis.x +
                                               center_vec.y * axis.y);
            // 垂直灯条方向的分量：即装甲板宽度方向的中心距。
            const double perp_axis = std::abs(center_vec.x * axis.y -
                                              center_vec.y * axis.x);
            const double pair_ratio = perp_axis / avg_length;
            const double along_offset = along_axis / avg_length;

            if (length_ratio > kMaxLengthRatio || angle_diff > kMaxAngleDiffDeg ||
                pair_ratio < kMinPairWidthRatio || pair_ratio > kMaxPairWidthRatio ||
                along_offset > kMaxAlongAxisOffsetRatio) {
                continue;
            }

            ArmorDetection detection;
            buildDetection(a, b, detection);

            // 分数：越接近标准装甲板宽高比、灯条越等长、倾角越一致、等高越准越好。
            const double score =
                -(std::abs(pair_ratio - 2.4) + std::abs(length_ratio - 1.0) +
                  angle_diff / 10.0 + along_offset);
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

    // 第二步：对已框出的目标识别中央贴纸，填写编号/置信度。
    // 识别失败不丢弃框：保留框但 target_id 保持 0（未知目标）。
    for (ArmorDetection& detection : results) {
        classifyCenter(bgr, detection);
    }

    // 第三步：时间平滑。角点直接取自灯条端点，逐帧有 2~5px 噪声，用 EMA 与上一帧
    // 同目标角点混合，消除“框位置对但四角跳”。只在检测到目标时平滑；检测丢失
    // 立即清空 prev_boxes_，绝不保留残影。
    for (ArmorDetection& det : results) {
        int best = -1;
        float best_d = kAssocDistPx;
        const cv::Point2f center = boxCenter(det);
        for (std::size_t k = 0; k < prev_boxes_.size(); ++k) {
            const float d =
                static_cast<float>(cv::norm(boxCenter(prev_boxes_[k]) - center));
            if (d < best_d) {
                best_d = d;
                best = static_cast<int>(k);
            }
        }
        if (best < 0) {
            continue;  // 无上一帧同目标：本帧直接用原始角点。
        }
        const std::array<cv::Point2f, 4> prev_corners = prev_boxes_[best].corners;
        for (int i = 0; i < 4; ++i) {
            // 按最近角点（而非序号）匹配，避免角点序号翻转导致框扭曲。
            int bi = 0;
            float bd = std::numeric_limits<float>::max();
            for (int j = 0; j < 4; ++j) {
                const float d =
                    static_cast<float>(cv::norm(det.corners[i] - prev_corners[j]));
                if (d < bd) {
                    bd = d;
                    bi = j;
                }
            }
            det.corners[i].x = kSmoothAlpha * det.corners[i].x +
                               (1.0F - kSmoothAlpha) * prev_corners[bi].x;
            det.corners[i].y = kSmoothAlpha * det.corners[i].y +
                               (1.0F - kSmoothAlpha) * prev_corners[bi].y;
        }
    }
    prev_boxes_ = results;

    return results;
}
