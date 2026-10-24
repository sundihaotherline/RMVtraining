#!/usr/bin/env python3
"""重新生成 9 类平均像素模板，与 C++ NumberClassifier::load() 的均值算法完全一致。

C++ 端是运行时按目录读全部样本做平均（本脚本与其等价），这里额外把平均模板
落盘成 datasets/templates/<类名>.png（可视化）和 .npy（raw float32），供人工
检查模板长什么样、确认数据集没被污染。

用法：
    python3 tools/generate_templates.py [--dataset datasets] [--out datasets/templates]
"""
import argparse
import glob
import os

import cv2
import numpy as np

K_WIDTH = 20
K_HEIGHT = 28
CLASS_DIRS = ["1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"]


def preprocess(gray):
    # 复刻 C++ NumberClassifier::preprocess：OTSU -> 前景 bbox -> pad -> resize -> /255。
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((K_HEIGHT, K_WIDTH), dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    pad_x = max(1, w // 8)
    pad_y = max(1, h // 8)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(binary.shape[1] - 1, x1 + pad_x)
    y1 = min(binary.shape[0] - 1, y1 + pad_y)
    digit = binary[y0:y1 + 1, x0:x1 + 1]
    resized = cv2.resize(digit, (K_WIDTH, K_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets")
    ap.add_argument("--out", default="datasets/templates")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for cls, name in enumerate(CLASS_DIRS):
        files = sorted(glob.glob(os.path.join(args.dataset, name, "*.*")))
        if not files:
            print(f"[{name:8s}] 无样本，跳过")
            continue
        acc = np.zeros((K_HEIGHT, K_WIDTH), dtype=np.float64)
        n = 0
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            acc += preprocess(img)
            n += 1
        if n == 0:
            print(f"[{name:8s}] 样本全部读失败，跳过")
            continue
        mean = (acc / n).astype(np.float32)
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), (mean * 255).astype(np.uint8))
        np.save(os.path.join(args.out, f"{name}.npy"), mean)
        print(f"[{name:8s}] n={n:5d} 已存 {args.out}/{name}.png/.npy "
              f"亮像素占比={(mean > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
