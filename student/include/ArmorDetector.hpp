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
//   3. 左右灯条两两匹配得到装甲板候选；
//   4. 生成顺时针排列的四个角点，按宽高比区分大小装甲板；
//   5. 裁剪中央区域做数字分类，只有编号 1..5 才判定为有效目标。
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

    // 设置分类器；传入 nullptr 可禁用数字分类。
    void setClassifier(const NumberClassifier* classifier) { classifier_ = classifier; }

    std::vector<ArmorDetection> detect(const cv::Mat& bgr, TeamColor enemy_color) const;

    // 调试：构建掩膜并提取灯条（不配对），供实时可视化反馈灯条级中间结果。
    std::vector<LightBar> debugExtractBars(const cv::Mat& bgr,
                                           TeamColor enemy_color) const {
        const cv::Mat mask = buildMask(bgr, enemy_color);
        return extractLightBars(mask);
    }

private:
    cv::Mat buildMask(const cv::Mat& bgr, TeamColor enemy_color) const;
    std::vector<LightBar> extractLightBars(const cv::Mat& mask) const;
    // 从匹配的左右灯条生成角点与目标。返回 true 表示生成有效候选。
    bool buildDetection(const LightBar& left, const LightBar& right,
                        ArmorDetection& detection) const;
    // 裁剪中央数字区域，分类并填写编号/置信度；无有效编号返回 false。
    bool classifyCenter(const cv::Mat& bgr, ArmorDetection& detection) const;

    const NumberClassifier* classifier_{nullptr};
};
