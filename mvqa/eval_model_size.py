"""
模型大小的取舍：n / m / l 三档在"计数准确率"和"CPU 延迟"上的对比

要回答的问题：
  现在是 yolov8n，行人只检出约一半。换 m 或 l 能补多少？
  而延迟从 39ms 涨到多少？换算成"用户要等多久"值不值。

报三组数：
  1. 每个类别的检出率 = 预测总数 / 真值总数（比 MAE 更直观地暴露"漏检"）
  2. 每张图总量的 MAE（用户实际感受到的误差）
  3. 单张推理延迟（决定能不能做交互式问答）

为什么单列"检出率"：
  MAE 会把"漏检"和"多检"混在一起。行人漏一半时 MAE 是 9，
  但真正的问题不是"差 9 个"，而是"系统性少报 50%"，那在问答里是严重错误。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_model_size.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHTS = [("yolov8n", "yolov8n_visdrone.pt"),
           ("yolov8m", "yolov8m_visdrone.pt"),
           ("yolov8l", "yolov8l_visdrone.pt")]
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
CONF = 0.25


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
    print(f"图片 {len(imgs)} 张，conf {CONF}\n")

    results = {}
    for tag, wf in WEIGHTS:
        wp = Path(r"D:/visdrone") / wf
        if not wp.exists():
            print(f"跳过 {tag}（缺权重）")
            continue
        model = YOLO(str(wp))
        model.predict(str(imgs[0]), verbose=False, device="cpu")    # 预热

        import time
        pred, lat = {}, []
        for ip in imgs:
            t0 = time.perf_counter()
            r = model.predict(str(ip), verbose=False, device="cpu", conf=CONF)[0]
            lat.append((time.perf_counter() - t0) * 1000)
            pred[ip.stem] = Counter(model.names[int(x)] for x in r.boxes.cls.tolist())
        results[tag] = (pred, np.median(lat))
        print(f"{tag} 完成，延迟中位 {np.median(lat):.0f}ms", flush=True)

    # ── 检出率 ──
    print(f"\n{'类别':<15}" + "".join(f"{t:>12}" for t, _ in WEIGHTS if t in results))
    print("  " + "-" * (15 + 12 * len(results)))
    for name in NAMES:
        t_sum = sum(truth[k][name] for k in truth)
        if t_sum == 0:
            continue
        row = f"  {name:<13}"
        for tag in results:
            p_sum = sum(results[tag][0][k][name] for k in results[tag][0])
            row += f"{p_sum/t_sum:>11.0%}"
        print(row + f"   (真值 {t_sum})")

    # ── 总量误差与延迟 ──
    T = np.array([sum(truth[k].values()) for k in truth])
    print(f"\n{'模型':<10}{'总量 MAE':>10}{'误差≤2':>9}{'延迟中位':>10}{'548 张总耗时':>14}")
    print("  " + "-" * 54)
    for tag in results:
        P = np.array([sum(results[tag][0][k].values()) for k in results[tag][0]])
        lat = results[tag][1]
        print(f"  {tag:<9}{np.abs(T-P).mean():>10.1f}"
              f"{(np.abs(T-P) <= 2).mean():>9.0%}{lat:>9.0f}ms"
              f"{lat*len(imgs)/1000:>12.0f}s")


if __name__ == "__main__":
    main()
