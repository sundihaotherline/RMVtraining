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
// 侧对时 LED 发光方向性强、亮度掉得厉害，饱和度和通道差阈值若太高会把偏离
// 正对较远的灯条直接滤掉，导致可测角度窗口过窄。放低这两项以扩大可测角度
// （样本更多 -> ω 收敛更快），代价是更容易误检，靠后面的配对 + PnP 重投影兜底。
constexpr int kColorSaturationMin = 55;
// V（亮度）阈值：正对相机时 LED 最亮（核心 V 饱和到 255），其 bloom 沿**横向**（垂直
// 灯条主轴）拖出一圈蓝色光晕——光晕保持蓝色（S≥55、B−R≥20 都过，是散射的蓝光不是白
// 光），只有 V 从核心 255 沿横向平滑衰减到 ~42。V=20/40 都会把这圈横向光晕收进来，
// 把两根竖直灯条各自横向撑胖：实测 1440x1080 正对帧，左灯条横向宽到 124px（长宽比
// 0.75）、右灯条 62px（长宽比 1.40），都 < kMinAspectRatio(1.5) 被拒 → 正对一根都检不出。
// 纵向 bloom 已有主轴裁剪（kBarTrimStrengthRatio）处理，横向没有，只能靠 V 阈值压。
// V≥70 两根才彻底分开，V=80 时左长宽比 3.15、右 1.77（余量足）。旋转车前板只在 ±16°
// 内被检、LED 亮度始终 >200，不受影响；若还要检大角度侧对板（LED 更暗）可回调到 30。
constexpr int kColorValueMin = 80;
// ---- BGR 通道差掩膜阈值 ----
constexpr int kChannelDiffMin = 20;

// ---- 灯条候选过滤 ----
constexpr double kMinLengthRatio = 0.03;   // 相对图像高度（1080p≈32px，放开识别距离）
constexpr double kMinAspectRatio = 1.5;    // 长度/宽度（侧对时灯条变胖，放宽）
constexpr double kMaxAngleDeg = 60.0;      // 相对竖直方向：允许装甲板绕中心 roll 到 60°
constexpr double kMinFillRatio = 0.35;     // 轮廓面积 / 外接矩形面积
constexpr double kMinBarArea = 25.0;       // 侧对时灯条变窄变暗，进一步放宽最小面积
// 灯条端点晕光裁剪：沿主轴采样颜色强度，强度低于该灯条峰值强度的
// 该比例处视为 bloom 边界。近端饱和 LED 的 bloom 更亮、更拖尾，
// 用绝对阈值会把远端真灯条一起误伤，故按每根灯条自身峰值做相对裁剪。
constexpr double kBarTrimStrengthRatio = 0.6;

// ---- 左右灯条匹配 ----
// 配对条件整体收紧：多灯条场景下避免把不同装甲板的灯条错配成一块，
// 只在几何上足够吻合时才生成候选框。
constexpr double kMaxLengthRatio = 1.35;     // 长度比：同板两灯条长度应接近
constexpr double kMaxAngleDiffDeg = 10.0;    // 角度差：两灯条应近乎平行
// 中心距/平均长度。下限放开到 0.6：侧对时两灯条在图像上靠得更近，0.6 对应约
// 75° 侧对角度；再低（0.4~0.5）只多换 3~5°，却让不同装甲板灯条错配风险陡增，
// 反而伤精度，故停在 0.6。上限放宽到 6.0：标准大装甲 230/57≈4.04，但暗场景下
// 灯条 bloom 裁剪会把实测长度裁短、抬高比值到 ~4.5~5.5（v4 实测大量帧落在
// 4.42~5.0 被旧阈值 4.5 误拒）；而真正错配的「异板/异物远对」比值都在 15 以上，
// 6.0 与 15 之间留了足够间隙，不会把错配放进来。
constexpr double kMinPairWidthRatio = 0.6;
constexpr double kMaxPairWidthRatio = 6.0;
constexpr double kMaxAlongAxisOffsetRatio = 0.30; // 中心沿灯条主轴偏移/平均长度
// 大小装甲板分界：中心距/灯条长度（小 135/57≈2.37，大 230/57≈4.04）。
constexpr double kLargeArmorRatio = 3.0;// 取中间值，避免小装甲板被误判为大装甲板。
// 标准宽长比（仅用于配对打分，不做硬门槛）：打分同时接纳两种尺寸，
// 避免大装甲板（≈4.04）被固定 2.4 的目标误扣分，在贪心匹配里输给错配的候选对。
constexpr double kSmallArmorStdRatio = 135.0 / 57.0;  // ≈2.37
constexpr double kLargeArmorStdRatio = 230.0 / 57.0;  // ≈4.04

// ---- 数字分类 ----
constexpr float kMinNumberConfidence = 0.5F;//图像贴纸与数据集相似度底线

// ---- 时间平滑（EMA）----
constexpr float kSmoothAlpha = 0.6F;   // 当前帧权重；上一帧权重 = 1 - alpha
constexpr float kAssocDistPx = 60.0F;  // 相邻帧同目标关联的最大中心位移（像素）
// 漏检保持帧数：166fps 下目标 6ms 内几乎不动，单帧/双帧漏检大多是阈值边缘抖动而非
// 目标真的出画。保持上一帧框最多 5 帧（~30ms），把这种闪灭抹平；真正出画（长间隔）
// 超过此上限后清空，不留残影。
constexpr int kMaxHoldFrames = 5;

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
    // 灯条被切成一堆短碎片被 min_length 砍掉。方形核任意方向都能桥接 LED 暗缝。
    // 距离越远灯条越小、暗缝的相对像素占比越大，9x9 在 1.6m 附近已桥不住，故
    // 加大到 13x13；两灯条中心距在有效距离内 >150px，不会把两根粘成一体。
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
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
            // 双线性插值采样：亮度剖面在亚像素 t 上平滑，端点定位不再被 lround 量化到
            // 整数像素。原实现 t 整数步进 + lround，端点被量化到 ~1px，是角点→PnP→EKF
            // 的高频抖动主源。
            const auto sample = [&](float t) -> float {
                const float px = x0 + t * vx;
                const float py = y0 + t * vy;
                if (px < 0.0F || py < 0.0F || px >= strength.cols - 1.0F ||
                    py >= strength.rows - 1.0F) {
                    return 0.0F;
                }
                const int xi = static_cast<int>(px);
                const int yi = static_cast<int>(py);
                const float fx = px - static_cast<float>(xi);
                const float fy = py - static_cast<float>(yi);
                const float v00 = static_cast<float>(strength.at<uchar>(yi, xi));
                const float v01 = static_cast<float>(strength.at<uchar>(yi, xi + 1));
                const float v10 = static_cast<float>(strength.at<uchar>(yi + 1, xi));
                const float v11 = static_cast<float>(strength.at<uchar>(yi + 1, xi + 1));
                const float top = v00 + (v01 - v00) * fx;
                const float bot = v10 + (v11 - v10) * fx;
                return top + (bot - top) * fy;
            };

            float peak = 0.0F;
            for (float t = t_min; t <= t_max; t += 1.0F) {
                peak = std::max(peak, sample(t));
            }
            if (peak > 1.0F) {
                const float cutoff = peak * static_cast<float>(kBarTrimStrengthRatio);

                // 阈值跨越点做线性插值取亚像素：先整数步进找到跨越区间，再用区间两端
                // 亮度线性内插出 cutoff 对应的精确 t，端点不再吸附到整数像素。
                float t_top = t_min;
                for (float t = t_min; t <= 0.0F; t += 1.0F) {
                    const float s1 = sample(t);
                    if (s1 >= cutoff) {
                        const float s0 = sample(t - 1.0F);
                        t_top = s0 < cutoff ? (t - 1.0F) + (cutoff - s0) / (s1 - s0) : t;
                        break;
                    }
                    t_top = t;
                }
                float t_bottom = t_max;
                for (float t = t_max; t >= 0.0F; t -= 1.0F) {
                    const float s1 = sample(t);
                    if (s1 >= cutoff) {
                        const float s0 = sample(t + 1.0F);
                        t_bottom = s0 < cutoff ? t + (s1 - cutoff) / (s1 - s0) : t;
                        break;
                    }
                    t_bottom = t;
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
    last_pair_diag_ = {};
    if (bgr.empty()) {
        return holdOrClear();
    }

    const cv::Mat mask = buildMask(bgr, enemy_color);
    const cv::Mat strength = buildStrength(bgr);
    std::vector<LightBar> bars = extractLightBars(mask, strength);
    last_bars_ = bars;

    if (bars.size() < 2) {
        return holdOrClear();
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

    PairDiagnostic pair_diag;
    double best_perp = -1.0;

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

            // 记录「最宽灯条对」：即使被拒也记录，便于可视化定位是哪个门槛把框卡掉。
            const bool reject =
                length_ratio > kMaxLengthRatio || angle_diff > kMaxAngleDiffDeg ||
                pair_ratio < kMinPairWidthRatio || pair_ratio > kMaxPairWidthRatio ||
                along_offset > kMaxAlongAxisOffsetRatio;
            if (perp_axis > best_perp) {
                best_perp = perp_axis;
                pair_diag.valid = true;
                pair_diag.pair_ratio = pair_ratio;
                pair_diag.length_ratio = length_ratio;
                pair_diag.angle_diff = angle_diff;
                pair_diag.along_offset = along_offset;
                pair_diag.rejected = reject;
            }

            if (reject) {
                continue;
            }

            ArmorDetection detection;
            buildDetection(a, b, detection);

            // 分数：越接近标准装甲板宽高比、灯条越等长、倾角越一致、等高越准越好。
            // 宽高比取「到小/大装甲标准值的最小距离」，让两种尺寸都能拿到高分，
            // 否则固定 2.4 会让大装甲板（≈4.04）被错配候选对挤掉。
            const double ratio_error =
                std::min(std::abs(pair_ratio - kSmallArmorStdRatio),
                         std::abs(pair_ratio - kLargeArmorStdRatio));
            const double score =
                -(ratio_error + std::abs(length_ratio - 1.0) +
                  angle_diff / 10.0 + along_offset);
            candidates.push_back({detection, score, static_cast<int>(i),
                                  static_cast<int>(j)});
        }
    }

    last_pair_diag_ = pair_diag;

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

    // 灯条够多但没配出任何有效框：同样走保持逻辑，避免单帧配对失败导致框闪灭。
    if (results.empty()) {
        return holdOrClear();
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
        // 大小装甲板粘滞：板尺寸是恒定物理属性，而「中心距/灯条长」比例会随偏航角
        // 透视缩短、只会变小不会变大。所以上一帧已判为大装甲板（比例>3.0）的板，本帧
        // 因侧对角度增大、比例跌到阈值下被判成小，属于误判——沿用大板，避免 PnP 用错
        // 物点宽度（0.135 vs 0.230 m）导致深度 1.7 倍跳变。该跳变会让多车关联把同一
        // 辆车的板误拆成第二辆车（俯视图越开越多、编号乱）。
        if (prev_boxes_[best].size == ArmorSize::Large) {
            det.size = ArmorSize::Large;
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
    hold_frames_ = 0;

    return results;
}

std::vector<ArmorDetection> ArmorDetector::holdOrClear() {
    // 上一帧有框且保持未超限：沿用上一帧框，标记 held，供下游（可视化/EKF）区分
    // 这是「保持帧」而非新检测；超限后清空，避免目标出画后仍留残影。
    if (!prev_boxes_.empty() && hold_frames_ < kMaxHoldFrames) {
        std::vector<ArmorDetection> held = prev_boxes_;
        for (ArmorDetection& d : held) {
            d.held = true;
        }
        ++hold_frames_;
        return held;
    }
    prev_boxes_.clear();
    hold_frames_ = 0;
    return {};
}
