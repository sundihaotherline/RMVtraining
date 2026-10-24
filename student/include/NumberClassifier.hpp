#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// 装甲板中央图案分类器（传统「平均模板 + 归一化互相关 matchTemplate」）。
//
// 数据集（datasets/）中每张图片都是 20x28 灰度图，亮像素表示数字/图案（黑底白字）。
// 类别含义：
//   1..5   -> 有效装甲板编号
//   6outpost/7guard/8base -> 前哨站/哨兵/基地（非装甲板目标）
//   9neg   -> 负样本
//
// load() 逐类读图，用与 classify 完全相同的 preprocess（2/98% 拉伸 -> OTSU -> 前景
// bbox 裁剪居中 -> 20x28）累加求平均，得到 9 个 CV_32F 平均模板；classify() 对输入做
// 同一 preprocess 得到 query，再用 TM_CCOEFF_NORMED 与每个模板比归一化互相关取最高分。
// 相比 ONNX CNN：分数就是「query 与模板的相似度」，直观、可预期，不会像过拟合 CNN 那样
// 对域外输入自信地乱判。
class NumberClassifier {
public:
    static constexpr int kWidth = 20;
    static constexpr int kHeight = 28;

    // 从 dataset_dir 逐类读取 1..9 图片并生成平均模板。9 类齐全才返回 true。
    bool load(const std::string& dataset_dir);

    bool loaded() const { return loaded_; }

    // 对中央 ROI（单通道灰度图）分类。返回类别 1..9，置信度（TM_CCOEFF_NORMED 得分
    // 映射到 [0,1]）写入 confidence；ROI 为空/无前景/未加载时返回 0。
    int classify(const cv::Mat& roi, float& confidence) const;

    // 最近一次 classify() 的二值化图（OTSU 后、bbox 裁剪前，ROI 原生分辨率），
    // 供可视化验证阈值与透视校正是否正确。未分类过时为空。
    const cv::Mat& lastBinary() const { return last_binary_; }

    // 最近一次 classify() 的 query（bbox 裁剪后缩放到 kWidth x kHeight 的 CV_32F，
    // [0,1]）——即左边弹出的那张 20x28 小图，classify 就是拿它去和模板比。
    const cv::Mat& lastQuery() const { return last_query_; }

    // 最近一次 classify() 的 9 类归一化互相关得分（映射到 [0,1]，下标 0..8 对应 1..9）。
    // 用于定位「数字对应错」：正确类的分数是否紧咬第一、是否整体偏低（无信息）。
    const std::array<float, 9>& lastProbs() const { return last_probs_; }

private:
    // 阈值化并缩放到 kWidth x kHeight，输出 CV_32F 归一化到 [0,1]。
    // 可选把 OTSU 二值写到 binary_out 供调试可视化。
    static cv::Mat preprocess(const cv::Mat& roi, cv::Mat* binary_out = nullptr);

    bool loaded_{false};
    // 9 个平均模板（CV_32F，20x28，[0,1]），下标 0..8 对应类别 1..9。
    std::vector<cv::Mat> templates_;
    // 最近一次分类的二值化中间图（mutable：classify 是 const 方法）。
    mutable cv::Mat last_binary_;
    // 最近一次分类的 query（mutable：classify 是 const 方法）。
    mutable cv::Mat last_query_;
    // 最近一次分类的 9 类互相关得分（mutable：classify 是 const 方法）。
    mutable std::array<float, 9> last_probs_{};
};
