"""
切片推理（SAHI 思路）手写实现：和"提高 imgsz"比，谁更划算

原理：
  原图约 2000x1500。直接喂 imgsz=1280 时图像被缩到 0.64 倍，行人从 12px 变 7.7px。
  改成切成 640x640 的块、每块按 640 推理，块内不缩放（1.0 倍），行人保持 12px。
  → 等效分辨率比 imgsz=1280 还高 56%。

四步，每步都有一个坑：
  1. 切块，重叠 128（重叠太小，跨块目标会被切碎）
  2. 块内推理，坐标要加回偏移（忘加就全挤在左上角）
  3. 跨块做一次 NMS（不做的话同一辆车被数两遍，计数直接虚高）
  4. 丢弃落进忽略区域的框（VisDrone 协议）

指标和前面一致，方便直接对比：
  检出率、每张图 MAE、延迟

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_slice.py
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]

SLICE = 640          # 每块边长
OVERLAP = 128        # 块间重叠
LOW_CONF = 0.05
NMS_IOU = float(__import__("os").environ.get("NMS_IOU", "0.5"))
IGNORE_IOU = 0.5
THRESHOLDS = [0.20, 0.25, 0.30, 0.40]


def load_annots(imgs):
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
                    ign.append(box)
        valid[ip.stem] = c
        ignore[ip.stem] = np.array(ign, dtype=float).reshape(-1, 4)
    return valid, ignore


def max_iou(box, others):
    if len(others) == 0:
        return 0.0
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return float((inter / np.maximum(a + b - inter, 1e-9)).max())


def tile_origins(total, size, step):
    """覆盖整幅图所需的切块起点。最后一个必须是 total-size，避免右下角漏掉。"""
    if total <= size:
        return [0]
    xs = list(range(0, total - size, step))
    if not xs or xs[-1] != total - size:
        xs.append(total - size)
    return xs


def main():
    imgs = sorted((BASE / "images").glob("*.jpg"))
    N = len(imgs)
    valid, ignore = load_annots(imgs)
    T_all = np.array([sum(valid[k].values()) for k in valid])
    T_car = np.array([valid[k]["car"] for k in valid])
    T_ped = np.array([valid[k]["pedestrian"] for k in valid])

    model = YOLO(str(WEIGHT))
    names = model.names
    step = SLICE - OVERLAP

    # 预热 + 看平均块数
    img0 = cv2.imread(str(imgs[0]))
    h0, w0 = img0.shape[:2]
    xs0, ys0 = tile_origins(w0, SLICE, step), tile_origins(h0, SLICE, step)
    print(f"NMS IoU 阈值 {NMS_IOU}")
    print(f"原图 {w0}x{h0} → 切 {len(xs0)}x{len(ys0)} = {len(xs0)*len(ys0)} 块"
          f"（块 {SLICE}，重叠 {OVERLAP}）")

    dets = defaultdict(list)
    t0 = time.perf_counter()
    n_tiles = 0
    for ip in imgs:
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        xs, ys = tile_origins(w, SLICE, step), tile_origins(h, SLICE, step)
        boxes, scores, classes = [], [], []
        for oy in ys:
            for ox in xs:
                tile = img[oy:oy + SLICE, ox:ox + SLICE]
                r = model.predict(tile, verbose=False, device="cpu",
                                  imgsz=SLICE, conf=LOW_CONF)[0]
                n_tiles += 1
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy().copy()
                b[:, [0, 2]] += ox          # 坑 2：坐标加回偏移
                b[:, [1, 3]] += oy
                boxes.append(b)
                scores.append(r.boxes.conf.cpu().numpy())
                classes.append(r.boxes.cls.cpu().numpy())

        if not boxes:
            continue
        B = torch.tensor(np.vstack(boxes), dtype=torch.float32)
        S = torch.tensor(np.concatenate(scores), dtype=torch.float32)
        C = torch.tensor(np.concatenate(classes), dtype=torch.float32)

        # 坑 3：跨块去重，按类别分别做 NMS
        keep = []
        for c in C.unique():
            idx = (C == c).nonzero(as_tuple=True)[0]
            k = nms(B[idx], S[idx], NMS_IOU)
            keep.append(idx[k])
        keep = torch.cat(keep)
        B, S, C = B[keep].numpy(), S[keep].numpy(), C[keep].numpy()

        ig = ignore[ip.stem]
        for b, s, c in zip(B, S, C):
            if len(ig) and max_iou(b, ig) > IGNORE_IOU:      # 坑 4
                continue
            dets[ip.stem].append((names[int(c)], float(s)))

    per_img_ms = (time.perf_counter() - t0) / N * 1000
    print(f"单张平均 {per_img_ms:.0f}ms（共 {n_tiles} 次块推理，"
          f"平均 {n_tiles/N:.1f} 块/张）\n")

    print(f"{'阈值':>6}{'car检出':>9}{'car MAE':>9}{'行人检出':>10}"
          f"{'行人MAE':>9}{'总MAE':>8}{'误差≤2':>8}")
    print("  " + "-" * 58)
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
        print(f"{th:>6.2f}{cc.sum()/T_car.sum():>8.0%}{np.abs(cc-T_car).mean():>9.1f}"
              f"{cp.sum()/T_ped.sum():>9.0%}{np.abs(cp-T_ped).mean():>9.1f}"
              f"{np.abs(ca-T_all).mean():>8.1f}{(np.abs(ca-T_all)<=2).mean():>8.0%}")

    print(f"\n对照（同 548 张、n 模型、同协议）：")
    print("  n@640   总 MAE 27.2（conf 0.25）  延迟 40ms")
    print("  n@1280  总 MAE 13.2（conf 0.20）  延迟 109ms")


if __name__ == "__main__":
    main()
