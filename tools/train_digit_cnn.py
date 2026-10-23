#!/usr/bin/env python3
"""训练装甲板数字识别轻量 CNN，导出 ONNX 供 OpenCV cv::dnn 推理。

与 C++ NumberClassifier::preprocess() 保持完全一致（灰度 -> OTSU -> 前景 bbox ->
pad -> resize 到 28x20 -> 归一化到 [0,1]），训练/推理域对齐。类别 1..9（其中
6outpost/7guard/8base 为非装甲板目标，9neg 为负样本）。

用法：
    python3 tools/train_digit_cnn.py [--epochs N] [--out datasets/digit_cnn.onnx]
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

K_WIDTH = 20
K_HEIGHT = 28
CLASS_DIRS = ["1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"]


def preprocess(gray: np.ndarray) -> np.ndarray:
    """复刻 C++ NumberClassifier::preprocess：OTSU -> 前景 bbox -> pad -> resize。"""
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


def augment(gray: np.ndarray) -> np.ndarray:
    """在原始灰度图上做域差桥接增广（旋转/缩放/平移/亮度/模糊/噪声）。"""
    h, w = gray.shape
    # 仿射：旋转 ±10°、缩放 0.9~1.1、平移 ±15%。
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-0.15, 0.15) * w
    ty = random.uniform(-0.15, 0.15) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 亮度抖动（OTSU 对此有一定鲁棒，但模拟真实 ROI 的光照差异仍有帮助）。
    out = np.clip(out.astype(np.float32) + random.uniform(-30, 30), 0, 255).astype(np.uint8)
    # 高斯模糊模拟运动/离焦。
    if random.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    # 高斯噪声。
    if random.random() < 0.3:
        noise = np.random.normal(0, 8, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


class DigitDataset(Dataset):
    def __init__(self, dataset_dir: str, train: bool):
        self.samples = []  # (tensor [1,28,20], label 0..8)
        for cls, name in enumerate(CLASS_DIRS):
            files = sorted(glob.glob(os.path.join(dataset_dir, name, "*.*")))
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                self.samples.append((f, cls))
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, cls = self.samples[idx]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if self.train:
            img = augment(img)
        feat = preprocess(img)
        return torch.from_numpy(feat).unsqueeze(0), cls


class DigitCNN(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16x14x10
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 32x7x5
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),  # 64x7x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 5, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    correct = [0] * len(CLASS_DIRS)
    total = [0] * len(CLASS_DIRS)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            y = y.numpy()
            for p, t in zip(pred, y):
                total[t] += 1
                if p == t:
                    correct[t] += 1
    model.train()
    acc = [c / max(1, t) for c, t in zip(correct, total)]
    overall = sum(correct) / max(1, sum(total))
    return overall, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="datasets/digit_cnn.onnx")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = DigitDataset(args.dataset, train=True)
    # 分层切分训练/验证：逐类 85/15。
    train_files, val_files = [], []
    from collections import defaultdict
    by_cls = defaultdict(list)
    for f, cls in full.samples:
        by_cls[cls].append(f)
    for cls in range(len(CLASS_DIRS)):
        fs = by_cls[cls]
        random.shuffle(fs)
        nval = max(1, int(len(fs) * 0.15))
        val_files += [(f, cls) for f in fs[:nval]]
        train_files += [(f, cls) for f in fs[nval:]]

    train_ds = DigitDataset(args.dataset, train=True)
    train_ds.samples = train_files
    val_ds = DigitDataset(args.dataset, train=False)
    val_ds.samples = val_files
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = DigitCNN().to(device)
    # 类别不平衡（9neg 最多）→ 反频率加权。
    counts = [len(by_cls[c]) for c in range(len(CLASS_DIRS))]
    total = sum(counts)
    weights = torch.tensor([total / (len(CLASS_DIRS) * c) for c in counts],
                           dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()
        overall, acc = evaluate(model, val_loader, device)
        best = max(best, overall)
        print(f"epoch {ep:2d} loss {run_loss / len(train_ds):.4f} "
              f"val_acc {overall:.4f} (best {best:.4f})")
    print("per-class acc:", " ".join(f"{CLASS_DIRS[i]}={acc[i]:.3f}" for i in range(9)))

    # 导出 ONNX（opset 11，OpenCV 4.5.4 兼容）。经典导出器要求模型在 CPU。
    model.eval()
    model = model.cpu()
    dummy = torch.zeros(1, 1, K_HEIGHT, K_WIDTH)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        opset_version=11,
        dynamic_axes=None,
        dynamo=False,  # 走经典导出器（onnx 包），图更简单、OpenCV 4.5.4 dnn 兼容性更好
    )
    print(f"saved ONNX -> {args.out}")

    # 用 OpenCV dnn 复读一遍确认兼容（等价 C++ readNetFromONNX 的路径）。
    net = cv2.dnn.readNetFromONNX(args.out)
    blob = cv2.dnn.blobFromImage(np.zeros((K_HEIGHT, K_WIDTH), np.float32),
                                 1.0, (K_WIDTH, K_HEIGHT), 0.0)
    net.setInput(blob)
    out = net.forward()
    print("cv2.dnn forward ok, output shape:", out.shape)


if __name__ == "__main__":
    main()
