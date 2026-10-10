#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// 装甲板中央图案分类器。
//
// 数据集（datasets.zip）中的每张图片都是 20x28 的灰度图，亮像素表示
// 数字/图案（黑底白字）。类别含义：
//   1..5   -> 有效装甲板编号
//   6outpost/7guard/8base -> 前哨站/哨兵/基地（非装甲板目标）
//   9neg   -> 负样本
//
// 运行时按目录读取样本并计算每类平均模板，再用归一化互相关与待分类 ROI
// 比较。置信度为归一化后的相关系数，范围 [0, 1]。
class NumberClassifier {
public:
    static constexpr int kWidth = 20;
    static constexpr int kHeight = 28;

    // 从 dataset_dir 加载样本。该目录应包含 1,2,3,4,5,6outpost,7guard,8base,
    // 9neg 九个子目录。全部加载成功返回 true。
    bool load(const std::string& dataset_dir);

    bool loaded() const { return loaded_; }

    // 对中央 ROI（单通道灰度图）分类。返回类别 1..9，置信度写入 confidence；
    // ROI 为空或未加载时返回 0。
    int classify(const cv::Mat& roi, float& confidence) const;

private:
    // 阈值化并缩放到 kWidth x kHeight，输出 CV_32F 归一化到 [0, 1]。
    static cv::Mat preprocess(const cv::Mat& roi);

    bool loaded_{false};
    // 索引 0..8 依次对应类别 1..9 的平均模板（CV_32F，kWidth x kHeight）。
    std::vector<cv::Mat> templates_;
};
