"""
带忽略区域处理的正确评测（VisDrone 官方协议）

为什么必须改：
  上一版只按 score==1 过滤标注行，把 score==0 的行直接丢了。
  但 VisDrone 里 score==0 表示「忽略区域」——模糊、严重遮挡、边界不清的地方，
  规范要求：**检测框如果落进忽略区域（IoU>0.5），不计入误检**。
  不处理的话，模型在忽略区域里的正确检出会被算成"多检"，
  于是出现 car 检出率 124%、行人 115% 这种不可能的数字。

两个协议细节：
1. 忽略区域是类别无关的：一个框不管被识别成什么类，只要跟忽略区域重叠超阈值就丢
2. 计数用 score==1 的有效标注

对比三组配置，全部用同一套协议：
  n@640  / n@1280  / m@640

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_count_v2.py
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
CONFIGS = [("n@640", "yolov8n_visdrone.pt", 640),
           ("n@1280", "yolov8n_visdrone.pt", 1280),
           ("m@640", "yolov8m_visdrone.pt", 640)]
LOW_CONF = 0.05
IGNORE_IOU = 0.5
THRESHOLDS = [0.20, 0.25, 0.30, 0.40]


def iou_one(box, others):
    """一个框与一组框的最大 IoU。box=(x1,y1,x2,y2)，others 同格式的数组。"""
    if len(others) == 0:
        return 0.0
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = a + b - inter
    return float((inter / np.maximum(union, 1e-9)).max())


def load_annots(imgs):
    """返回 (每图有效标注计数, 每图的忽略区域数组)。"""
    valid, ignore = {}, {}
    for ip in imgs:
        ap = BASE / "annotations" / (ip.stem + ".txt")
        c, ign = Counter(), []
        if ap.exists():
            for line in ap.read_text().strip().splitlines():
                p = line.split(",")
                if len(p) < 6:
                    continue
                x, y, w, h = (float(v) for v in p[:4])
                box = (x, y, x + w, y + h)
                if p[4].strip() == "1":
                    cat = int(p[5])
                    if 1 <= cat <= 10:
                        c[NAMES[cat - 1]] += 1
                else:
                    ign.append(box)          # score==0 → 忽略区域
        valid[ip.stem] = c
        ignore[ip.stem] = np.array(ign, dtype=float).reshape(-1, 4)
    return valid, ignore


def main():
    imgs = sorted((BASE / "images").glob("*.jpg"))
    N = len(imgs)
    valid, ignore = load_annots(imgs)
    n_ign = sum(len(v) for v in ignore.values())
    print(f"{N} 张图，有效标注目标 {sum(sum(c.values()) for c in valid.values())} 个，"
          f"忽略区域 {n_ign} 个")

    T_all = np.array([sum(valid[k].values()) for k in valid])
    T_car = np.array([valid[k]["car"] for k in valid])
    T_ped = np.array([valid[k]["pedestrian"] for k in valid])

    print(f"\n{'配置':>10}{'阈值':>7}{'car检出':>9}{'car MAE':>9}"
          f"{'行人检出':>10}{'行人MAE':>9}{'总MAE':>8}{'误差≤2':>8}")
    print("  " + "-" * 72)

    for tag, wf, sz in CONFIGS:
        model = YOLO(str(Path(r"D:/visdrone") / wf))
        model.predict(str(imgs[0]), verbose=False, device="cpu", imgsz=sz, conf=LOW_CONF)

        dets = defaultdict(list)
        for ip in imgs:
            r = model.predict(str(ip), verbose=False, device="cpu",
                              imgsz=sz, conf=LOW_CONF)[0]
            names = model.names
            ig = ignore[ip.stem]
            for b, cls, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                  r.boxes.conf.tolist()):
                # 落进忽略区域的框直接丢（类别无关）
                if len(ig) and iou_one(b, ig) > IGNORE_IOU:
                    continue
                dets[ip.stem].append((names[int(cls)], float(cf)))

        for th in THRESHOLDS:
            ca = np.zeros(N); cc = np.zeros(N); cp = np.zeros(N)
            for i, ip in enumerate(imgs):
                for name, cf in dets[ip.stem]:
                    if cf >= th:
                        ca[i] += 1
                        if name == "car":
                            cc[i] += 1
                        elif name == "pedestrian":
                            cp[i] += 1
            print(f"{tag:>10}{th:>7.2f}{cc.sum()/T_car.sum():>8.0%}"
                  f"{np.abs(cc-T_car).mean():>9.1f}{cp.sum()/T_ped.sum():>9.0%}"
                  f"{np.abs(cp-T_ped).mean():>9.1f}{np.abs(ca-T_all).mean():>8.1f}"
                  f"{(np.abs(ca-T_all)<=2).mean():>8.0%}")
        print()


if __name__ == "__main__":
    main()
