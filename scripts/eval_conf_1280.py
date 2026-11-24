"""
1280 分辨率下的阈值扫描与真实误差

为什么这么写：
  直接跑 5 个阈值要推理 5 遍（548×5 次）。改成只跑一遍（conf=0.05 全收），
  把每个框的类别和置信度存下来，阈值在离线算 —— 一遍推理换整条曲线。

要回答两件事：
  1. car 的检出率在 1280 下超过 100%，是不是阈值不合适？哪个阈值能回到 100% 附近
  2. 用户真正感受到的是"每张图差几个数"（MAE），不是总检出率

报四组指标（都按每张图算）：
  检出率 = 预测总数 / 真值总数    （看系统性偏差）
  MAE    = |预测 - 真值| 的均值     （看单张误差）
  完全准确 / 误差≤2                （看用户容忍度）

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_conf_1280.py
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
IMGSZ = 1280
LOW_CONF = 0.05
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65]


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
    model = YOLO(str(WEIGHT))
    model.predict(str(imgs[0]), verbose=False, device="cpu", imgsz=IMGSZ, conf=LOW_CONF)

    # 跑一遍，把 (类别, 置信度) 全存下来
    boxes = defaultdict(list)
    import time
    t0 = time.perf_counter()
    for ip in imgs:
        r = model.predict(str(ip), verbose=False, device="cpu",
                          imgsz=IMGSZ, conf=LOW_CONF)[0]
        names = model.names
        for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            boxes[ip.stem].append((names[int(cls)], float(conf)))
    per_img_ms = (time.perf_counter() - t0) / N * 1000
    print(f"{N} 张图，imgsz={IMGSZ}，单张 {per_img_ms:.0f}ms"
          f"（含推理+取框，共 {(time.perf_counter()-t0):.0f}s）")
    print(f"低阈值 {LOW_CONF} 下共收集 {sum(len(v) for v in boxes.values())} 个框\n")

    T_all = np.array([sum(truth[k].values()) for k in truth])
    T_car = np.array([truth[k]["car"] for k in truth])
    T_ped = np.array([truth[k]["pedestrian"] for k in truth])

    print(f"{'阈值':>6}{'car检出':>9}{'car MAE':>9}{'行人检出':>10}{'行人 MAE':>10}"
          f"{'总MAE':>8}{'总误差≤2':>10}")
    print("  " + "-" * 64)
    best = None
    for th in THRESHOLDS:
        c_all = np.zeros(N); c_car = np.zeros(N); c_ped = np.zeros(N)
        for i, ip in enumerate(imgs):
            st = ip.stem
            for name, conf in boxes[st]:
                if conf >= th:
                    c_all[i] += 1
                    if name == "car":
                        c_car[i] += 1
                    elif name == "pedestrian":
                        c_ped[i] += 1
        row = (th,
               c_car.sum() / T_car.sum(), np.abs(c_car - T_car).mean(),
               c_ped.sum() / T_ped.sum(), np.abs(c_ped - T_ped).mean(),
               np.abs(c_all - T_all).mean(), (np.abs(c_all - T_all) <= 2).mean())
        print(f"{th:>6.2f}{row[1]:>8.0%}{row[2]:>9.1f}{row[3]:>9.0%}{row[4]:>10.1f}"
              f"{row[5]:>8.1f}{row[6]:>10.0%}")
        if row[5] == min(r[5] for r in [
                (th, 0, 0, 0, 0, np.abs(c_all - T_all).mean(), 0)]):
            pass
        if best is None or row[5] < best[5]:
            best = row

    print(f"\n按总 MAE 最优的阈值：{best[0]:.2f}"
          f"（car 检出 {best[1]:.0%}、行人检出 {best[3]:.0%}、总 MAE {best[5]:.1f}）")
    print("对照：640 分辨率下 yolov8m 的总 MAE 是 20.9，yolov8n 是 27.1")


if __name__ == "__main__":
    main()
