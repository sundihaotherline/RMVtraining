#include "NumberClassifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 类别目录名，按类别编号 1..9 排列（仅用于文档/对照，CNN 输出 9 类即对应 1..9）。
const std::array<const char*, 9> kClassDirs = {
    "1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg",
};

}  // namespace

cv::Mat NumberClassifier::preprocess(const cv::Mat& roi) {
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else if (roi.channels() == 4) {
        cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = roi;
    }

    // 数字是亮图案：先自适应阈值得到黑白图。
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 只保留亮像素的包围盒并居中，再缩放到网络输入尺寸。与训练脚本
    // tools/train_digit_cnn.py 的 preprocess() 保持一致，保证训练/推理域对齐。
    std::vector<cv::Point> foreground;
    cv::findNonZero(binary, foreground);
    if (foreground.empty()) {
        return cv::Mat::zeros(kHeight, kWidth, CV_32F);
    }

    cv::Rect box = cv::boundingRect(foreground);
    // 外扩一点边距，避免数字笔画贴边被裁掉。
    const int pad_x = std::max(1, box.width / 8);
    const int pad_y = std::max(1, box.height / 8);
    box.x = std::max(0, box.x - pad_x);
    box.y = std::max(0, box.y - pad_y);
    box.width = std::min(binary.cols - box.x, box.width + 2 * pad_x);
    box.height = std::min(binary.rows - box.y, box.height + 2 * pad_y);

    cv::Mat digit = binary(box);
    cv::Mat resized;
    cv::resize(digit, resized, cv::Size(kWidth, kHeight), 0, 0,
               cv::INTER_AREA);

    cv::Mat result;
    resized.convertTo(result, CV_32F, 1.0 / 255.0);
    return result;
}

bool NumberClassifier::load(const std::string& dataset_dir) {
    net_ = cv::dnn::Net();
    loaded_ = false;

    // 轻量 CNN 分类器：由 tools/train_digit_cnn.py 训练并导出为 ONNX（9 类，含负样本）。
    const std::string model_path = dataset_dir + "/digit_cnn.onnx";
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception&) {
        net_ = cv::dnn::Net();
    }

    loaded_ = !net_.empty();
    return loaded_;
}

int NumberClassifier::classify(const cv::Mat& roi, float& confidence) const {
    confidence = 0.0F;
    if (!loaded_ || roi.empty()) {
        return 0;
    }

    const cv::Mat query = preprocess(roi);

    // NCHW 输入：[1, 1, kHeight, kWidth]。query 已是 kHeight x kWidth 单通道 CV_32F。
    cv::Mat blob = cv::dnn::blobFromImage(query, 1.0, cv::Size(kWidth, kHeight),
                                          cv::Scalar(), false, false);
    net_.setInput(blob);
    const cv::Mat out = net_.forward();  // shape (1, 9) logits

    const float* logits = out.ptr<float>(0);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 9; ++i) {
        max_logit = std::max(max_logit, static_cast<double>(logits[i]));
    }
    double probs[9];
    double sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[i]) - max_logit);
        sum += probs[i];
    }
    int best = 0;
    for (int i = 0; i < 9; ++i) {
        probs[i] /= sum;
        if (probs[i] > probs[best]) {
            best = i;
        }
    }

    confidence = static_cast<float>(probs[best]);
    return best + 1;  // 1..9
}
