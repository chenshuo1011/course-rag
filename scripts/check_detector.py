"""
检测器可用性验证：VisDrone 预训练权重在 CPU 上能不能用

要回答三件事：
  1. 权重能不能加载、类别对不对
  2. CPU 上一张图推理多久（决定能不能做交互式问答）
  3. 检测结果看着对不对（数量、框的位置）

这一步不做精度评测——没有标注，先看"能不能用"。
如果这一关过不了，后面的"检测+Agent 计数问答"整条路都不用谈了。

用法：
    "D:\anaconda3\envs\yolo\python.exe" check_detector.py
"""

import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
IMGS = Path(r"D:/visdrone/imgs")
OUT = Path(r"D:/visdrone/rendered")

# VisDrone 的 10 类（和 COCO 完全不同，这是为什么必须用领域微调权重）
VISDRONE_NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
                  5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus",
                  9: "motor"}


def main():
    print(f"加载权重 {WEIGHT.name} ...")
    model = YOLO(str(WEIGHT))
    names = model.names
    print(f"  模型自带类别（{len(names)} 类）：{list(names.values())[:12]}")

    # 过滤掉下载失败的小文件（hf-mirror 限流时会返回 300 多字节的提示页，
    # 后缀还是 .jpg，cv2.imread 会返回 None 然后崩在 im.ndim 上）
    files = [f for f in sorted(IMGS.glob("*.jpg")) if f.stat().st_size > 20_000]
    if len(files) < len(list(IMGS.glob("*.jpg"))):
        print(f"  跳过 {len(list(IMGS.glob('*.jpg'))) - len(files)} 个无效文件（下载被限流）")
    if not files:
        print("没有图片")
        return
    print(f"\n测试 {len(files)} 张图")

    # 预热：第一次推理包含模型图优化，不能算进平均
    model.predict(str(files[0]), verbose=False, device="cpu")

    times, counts = [], []
    for f in files:
        t0 = time.perf_counter()
        r = model.predict(str(f), verbose=False, device="cpu", conf=0.25)[0]
        times.append((time.perf_counter() - t0) * 1000)
        cls = r.boxes.cls.tolist() if r.boxes is not None else []
        c = Counter(names[int(x)] for x in cls)
        counts.append(sum(c.values()))

    times.sort()
    print(f"\n{'推理耗时':<10}{'毫秒':>8}")
    print(f"  {'中位数':<8}{times[len(times)//2]:>8.0f}")
    print(f"  {'最快':<8}{times[0]:>8.0f}")
    print(f"  {'最慢':<8}{times[-1]:>8.0f}")
    print(f"  {'单张均值':<8}{sum(times)/len(times):>8.0f}")

    print(f"\n每张图检出的目标数：最少 {min(counts)}，中位 "
          f"{sorted(counts)[len(counts)//2]}，最多 {max(counts)}")

    # 拿目标最多的一张渲染出来，人眼确认框准不准
    OUT.mkdir(parents=True, exist_ok=True)
    idx = counts.index(max(counts))
    r = model.predict(str(files[idx]), verbose=False, device="cpu", conf=0.25)[0]
    print(f"\n渲染目标最多的一张（{files[idx].name}，检出 {counts[idx]} 个）：")
    c = Counter(names[int(x)] for x in r.boxes.cls.tolist())
    for k, v in c.most_common():
        print(f"    {k}: {v}")

    plotted = r.plot()
    out_path = OUT / f"det_{files[idx].stem}.jpg"
    cv2.imwrite(str(out_path), plotted)
    print(f"\n带框的图已存：{out_path}")
    print("（用看图工具打开，确认框是不是压在车/人上，而不是乱标）")


if __name__ == "__main__":
    main()
