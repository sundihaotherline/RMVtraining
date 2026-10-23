#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// 装甲板中央图案分类器。
//
// 数据集（datasets.zip）中的每张图片都是 20x28 的灰度图，亮像素表示
// 数字/图案（黑底白字）。类别含义：
//   1..5   -> 有效装甲板编号
//   6outpost/7guard/8base -> 前哨站/哨兵/基地（非装甲板目标）
//   9neg   -> 负样本
//
// 轻量 CNN 分类器（tools/train_digit_cnn.py 训练，ONNX 导出），由 OpenCV cv::dnn
// 推理。相比原「平均模板 + matchTemplate」，对 live ROI 的尺度/光照/角度域差更鲁棒。
// 置信度为 softmax 概率，范围 [0, 1]。
class NumberClassifier {
public:
    static constexpr int kWidth = 20;
    static constexpr int kHeight = 28;

    // 从 dataset_dir 加载 ONNX 模型（dataset_dir/digit_cnn.onnx）。加载成功返回 true。
    bool load(const std::string& dataset_dir);

    bool loaded() const { return loaded_; }

    // 对中央 ROI（单通道灰度图）分类。返回类别 1..9，置信度写入 confidence；
    // ROI 为空或未加载时返回 0。
    int classify(const cv::Mat& roi, float& confidence) const;

private:
    // 阈值化并缩放到 kWidth x kHeight，输出 CV_32F 归一化到 [0, 1]。
    static cv::Mat preprocess(const cv::Mat& roi);

    bool loaded_{false};
    // cv::dnn::Net::setInput/forward 非 const（会改内部状态），classify 是 const 方法，
    // 故标记 mutable。
    mutable cv::dnn::Net net_;
};
