#include "NumberClassifier.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 类别目录名，按类别编号 1..9 排列。
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

    // 只保留亮像素的包围盒并居中，再缩放到模板尺寸。模板与待分类图都走同一套
    // 处理，消除数字在 ROI 内位置/大小不同造成的误分类（这是“只有 2 能被认出”的主因）。
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
    templates_.clear();
    loaded_ = false;

    // 逐类读取全部样本，累加得到平均模板。
    for (int cls = 0; cls < static_cast<int>(kClassDirs.size()); ++cls) {
        std::vector<cv::String> files;
        try {
            cv::glob(dataset_dir + "/" + kClassDirs[cls] + "/*.*", files, false);
        } catch (const cv::Exception&) {
            // 目录不存在时 cv::glob 会抛异常，按空目录处理。
            files.clear();
        }
        if (files.empty()) {
            continue;
        }

        cv::Mat sum = cv::Mat::zeros(kHeight, kWidth, CV_64F);
        std::size_t count = 0;
        for (const cv::String& file : files) {
            cv::Mat sample = cv::imread(file, cv::IMREAD_GRAYSCALE);
            if (sample.empty()) {
                continue;
            }
            // 与 classify 使用同一套预处理（阈值 + 居中 + 缩放），保证模板与查询对齐。
            cv::Mat normalized;
            preprocess(sample).convertTo(normalized, CV_64F);
            sum += normalized;
            ++count;
        }

        if (count == 0) {
            continue;
        }

        cv::Mat mean;
        sum.convertTo(mean, CV_32F, 1.0 / static_cast<double>(count));
        templates_.push_back(mean);
    }

    // 需要九个类别都加载成功才算就绪。
    loaded_ = templates_.size() == kClassDirs.size();
    return loaded_;
}

int NumberClassifier::classify(const cv::Mat& roi, float& confidence) const {
    confidence = 0.0F;
    if (!loaded_ || roi.empty()) {
        return 0;
    }

    const cv::Mat query = preprocess(roi);
    int best_class = 0;
    double best_score = -1.0;

    for (std::size_t cls = 0; cls < templates_.size(); ++cls) {
        // 模板与查询同尺寸，TM_CCOEFF_NORMED 会输出单一相关值。
        cv::Mat result;
        cv::matchTemplate(query, templates_[cls], result, cv::TM_CCOEFF_NORMED);
        const double score = result.at<float>(0, 0);
        if (score > best_score) {
            best_score = score;
            best_class = static_cast<int>(cls) + 1;
        }
    }

    // 相关系数范围 [-1, 1]，映射到 [0, 1]。
    confidence = static_cast<float>(std::max(0.0, (best_score + 1.0) / 2.0));
    return best_class;
}
