"""
计数问答的量化：检测器数得准不准

这是"检测 + Agent 计数问答"这条路的量化关卡。
问答系统的答案是"图里有 23 辆车"，所以真正要评估的是：
  · 检测器数出来的数，和标注差多少
  · 差得少的比例有多高（用户能不能接受）

三个指标：
  MAE        平均绝对误差（|预测-真值| 的均值），越小越好
  完全准确    预测 == 真值的比例
  误差≤2      差距不超过 2 个的比例（人的容忍带）

一个必要对照：
  如果所有图都拿"中位数量"去猜，MAE 是多少？
  检测器必须明显好于这个常数猜测，否则它没有价值。

数据：VisDrone2019-DET-val，548 张（无人机视角，10 类）
标注格式：x,y,w,h,score,category,truncation,occlusion（score=1 才算有效）

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_count.py
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")

# 标注里的 category 是 1~10，模型输出 0~9，差一个
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]

CONF = 0.25


def main():
    imgs = sorted((BASE / "images").glob("*.jpg"))
    print(f"图片 {len(imgs)} 张，置信度阈值 {CONF}")

    # ── 标注侧计数 ──
    truth = {}
    for ip in imgs:
        ap = BASE / "annotations" / (ip.stem + ".txt")
        c = Counter()
        if ap.exists():
            for line in ap.read_text().strip().splitlines():
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                if parts[4].strip() != "1":          # 0 = 忽略区域，不算目标
                    continue
                cat = int(parts[5])
                if 1 <= cat <= 10:
                    c[NAMES[cat - 1]] += 1
        truth[ip.stem] = c

    total_obj = sum(sum(c.values()) for c in truth.values())
    print(f"标注目标总数 {total_obj}，平均每张 {total_obj/len(imgs):.0f} 个")

    model = YOLO(str(WEIGHT))
    model.predict(str(imgs[0]), verbose=False, device="cpu")     # 预热

    pred = {}
    for ip in imgs:
        r = model.predict(str(ip), verbose=False, device="cpu", conf=CONF)[0]
        c = Counter(model.names[int(x)] for x in r.boxes.cls.tolist())
        pred[ip.stem] = c

    n = len(imgs)
    print(f"\n{'类别':<16}{'真值总数':>9}{'预测总数':>9}{'MAE':>8}{'完全准确':>10}{'误差≤2':>9}")
    print("  " + "-" * 60)
    for name in NAMES:
        t = np.array([truth[k][name] for k in truth])
        p = np.array([pred[k][name] for k in pred])
        if t.sum() == 0:
            continue
        mae = np.abs(t - p).mean()
        exact = (t == p).mean()
        near = (np.abs(t - p) <= 2).mean()
        print(f"  {name:<14}{t.sum():>9}{p.sum():>9}{mae:>8.1f}{exact:>10.0%}{near:>9.0%}")

    # ── 总量对比 + 常数基线 ──
    T = np.array([sum(truth[k].values()) for k in truth])
    P = np.array([sum(pred[k].values()) for k in pred])
    base = np.full_like(T, int(np.median(T)))
    print(f"\n{'每张图目标总数':<16}{'MAE':>8}{'完全准确':>10}{'误差≤2':>9}")
    print("  " + "-" * 44)
    print(f"  {'检测器':<14}{np.abs(T-P).mean():>8.1f}"
          f"{(T == P).mean():>10.0%}{(np.abs(T - P) <= 2).mean():>9.0%}")
    print(f"  {'常数猜测(中位)':<14}{np.abs(T-base).mean():>8.1f}"
          f"{(T == base).mean():>10.0%}{(np.abs(T - base) <= 2).mean():>9.0%}")

    # ── 阈值影响：检测器数量受 conf 影响，值得看一眼 ──
    print("\n换个置信度阈值（只看 car，全 548 张会慢，抽 100 张）：")
    sub = imgs[:100]
    for conf in (0.15, 0.25, 0.4, 0.6):
        errs = []
        for ip in sub:
            r = model.predict(str(ip), verbose=False, device="cpu", conf=conf)[0]
            c = Counter(model.names[int(x)] for x in r.boxes.cls.tolist())
            errs.append(abs(c.get("car", 0) - truth[ip.stem]["car"]))
        print(f"  conf {conf}: car 的 MAE = {np.mean(errs):.1f}")


if __name__ == "__main__":
    main()
