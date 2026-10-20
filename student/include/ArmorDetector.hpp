#pragma once

#include "AutoAimTypes.hpp"
#include "NumberClassifier.hpp"

#include <vector>

#include <opencv2/core.hpp>

// 传统视觉装甲板检测器。
//
// 流程：
//   1. HSV 颜色掩膜 + BGR 亮度掩膜，合成后做开、闭运算；
//   2. 提取细长灯条轮廓，过滤角度/面积/长宽比/填充率；
//   3. 左右灯条两两匹配（几何条件收紧），生成装甲板候选框；
//   4. 生成顺时针排列的四个角点，按宽高比区分大小装甲板；
//   5. 对已框出的目标裁剪中央区域做数字分类（编号 1..9），识别失败保留框但编号置 0。
class ArmorDetector {
public:
    // 单个灯条候选（沿主轴拟合的线段）。
    struct LightBar {
        cv::Point2f top;       // 顶部端点
        cv::Point2f bottom;    // 底部端点
        cv::Point2f center;    // 中心
        float angle_deg{0.0F}; // 相对竖直方向的倾角（度）
        float length{0.0F};    // 主轴长度（像素）
        float width{0.0F};     // 垂直方向宽度（像素）
        float fill_ratio{0.0F};// 轮廓面积 / 外接矩形面积
    };

    // 配对诊断：记录每帧「最宽灯条对」（perp_axis 最大，通常就是被观察的装甲板）
    // 的各几何量与被拒标志，供可视化判断框丢失是哪个门槛卡住。
    struct PairDiagnostic {
        bool valid{false};
        double pair_ratio{0.0};     // 垂直轴中心距 / 平均灯条长
        double length_ratio{0.0};   // 两灯条长度比
        double angle_diff{0.0};     // 倾角差（度）
        double along_offset{0.0};   // 沿轴偏移 / 平均灯条长
        bool rejected{false};       // 该对是否被任一几何条件拒绝
    };

    // 设置分类器；传入 nullptr 可禁用数字分类。
    void setClassifier(const NumberClassifier* classifier) { classifier_ = classifier; }

    std::vector<ArmorDetection> detect(const cv::Mat& bgr, TeamColor enemy_color);

    // 最近一次 detect() 提取的灯条（含未配对的），供可视化直接读取，避免可视化
    // 再跑一遍 buildMask/extract 的重复计算。
    const std::vector<LightBar>& lastBars() const { return last_bars_; }

    // 最近一次 detect() 的配对诊断（见 PairDiagnostic），供可视化读取。
    const PairDiagnostic& lastPairDiagnostic() const { return last_pair_diag_; }

private:
    cv::Mat buildMask(const cv::Mat& bgr, TeamColor enemy_color) const;
    // 亮度图（V = max(R,G,B)），供灯条端点晕光裁剪。过曝灯条核心发白，
    // 颜色差在白核处掉到 0 会误裁真灯条，亮度在核心最亮、bloom 渐暗更稳。
    cv::Mat buildStrength(const cv::Mat& bgr) const;
    std::vector<LightBar> extractLightBars(const cv::Mat& mask,
                                           const cv::Mat& strength) const;
    // 从匹配的左右灯条生成角点与目标。返回 true 表示生成有效候选。
    bool buildDetection(const LightBar& left, const LightBar& right,
                        ArmorDetection& detection) const;
    // 裁剪中央数字区域，分类并填写编号（1..9）/置信度；未识别返回 false。
    bool classifyCenter(const cv::Mat& bgr, ArmorDetection& detection) const;

    const NumberClassifier* classifier_{nullptr};
    // 时间平滑状态：上一帧的框，用于对同目标做 EMA 角点平滑。
    // 配合 hold_frames_：检测短暂丢失时不立即清空，而是沿用上一帧框保持若干帧，
    // 抹平 166fps 下的单帧/双帧漏检抖动；真正出画（长间隔）后再清空。
    std::vector<ArmorDetection> prev_boxes_;
    int hold_frames_{0};
    // 未检测到目标时的统一出口：保持帧数未超限则返回上一帧框的 held 拷贝，否则清空。
    std::vector<ArmorDetection> holdOrClear();
    // 最近一次 detect() 提取的灯条缓存，供可视化读取（见 lastBars()）。
    std::vector<LightBar> last_bars_;
    // 最近一次 detect() 的配对诊断缓存，供可视化读取（见 lastPairDiagnostic()）。
    PairDiagnostic last_pair_diag_;
};
