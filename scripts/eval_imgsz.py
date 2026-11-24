"""
分辨率实验：imgsz 对小目标检出的影响

背景：
  VisDrone 原图约 2000x1500，而 YOLO 默认 imgsz=640 —— 缩放后一个行人只剩 4x6 像素。
  之前量到"行人只检出 67%"，有可能是这个默认设置压出来的，不是模型极限。

  VisDrone 的论文和比赛方案基本都用 1500 或切片推理，我们之前用的是默认值。

为什么先测这个：
  它决定要不要上切片推理（SAHI）。
  如果 1280 时行人涨很多 → 是分辨率问题，切片会有更大收益；
  如果只涨几个点 → 模型容量也不够，切片收益有限。

一个必须注意的点：
  提高 imgsz 会增加延迟（像素数平方关系），而阈值 conf 也应该跟着调——
  分辨率变了，置信度分布会变。这里先用统一的 0.25，保持各组可比。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_imgsz.py
"""

import time
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
MODELS = [("yolov8n", "yolov8n_visdrone.pt"), ("yolov8m", "yolov8m_visdrone.pt")]
SIZES = [640, 960, 1280]
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
CONF = 0.25
# 只关注这几类：大目标看有没有变差，中小目标看提升
WATCH = ["car", "motor", "tricycle", "pedestrian", "people"]


def load_truth(imgs):
    truth = {}
    for ip in imgs:
        ap = BASE / "annotations" / (ip.stem + ".txt")
        c = Counter()
        if ap.exists():
            for line in ap.read_text().strip().splitlines():
                p = line.split(",")
                if len(p) >= 6 and p[4].strip() == "1":
                    cat = int(p[5])
                    if 1 <= cat <= 10:
                        c[NAMES[cat - 1]] += 1
        truth[ip.stem] = c
    return truth


def main():
    imgs = sorted((BASE / "images").glob("*.jpg"))
    truth = load_truth(imgs)
    N = len(imgs)
    tot = {n: sum(t[n] for t in truth.values()) for n in NAMES}
    print(f"{N} 张图，conf {CONF}，真值：" +
          "、".join(f"{n} {tot[n]}" for n in WATCH) + "\n", flush=True)

    rows = []
    for tag, wf in MODELS:
        model = YOLO(str(Path(r"D:/visdrone") / wf))
        for sz in SIZES:
            model.predict(str(imgs[0]), verbose=False, device="cpu", imgsz=sz)  # 预热
            t0 = time.perf_counter()
            pred = {n: 0 for n in NAMES}
            for ip in imgs:
                r = model.predict(str(ip), verbose=False, device="cpu",
                                  imgsz=sz, conf=CONF)[0]
                for c in r.boxes.cls.tolist():
                    pred[model.names[int(c)]] += 1
            dt = (time.perf_counter() - t0) / N * 1000
            rec = {n: pred[n] / tot[n] for n in NAMES if tot[n]}
            rows.append((tag, sz, rec, dt))
            print(f"{tag} imgsz={sz}: " +
                  "  ".join(f"{n} {rec[n]:.0%}" for n in WATCH) +
                  f"   延迟 {dt:.0f}ms", flush=True)

    print(f"\n{'模型':<10}{'imgsz':>7}" + "".join(f"{n:>13}" for n in WATCH) + f"{'延迟':>9}")
    print("  " + "-" * (17 + 13 * len(WATCH) + 9))
    for tag, sz, rec, dt in rows:
        print(f"  {tag:<8}{sz:>7}" + "".join(f"{rec[n]:>12.0%}" for n in WATCH)
              + f"{dt:>8.0f}ms")


if __name__ == "__main__":
    main()
