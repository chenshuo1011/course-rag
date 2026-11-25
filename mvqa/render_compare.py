"""
人工核对：切片检出的"多出来的车"到底是不是车

要判断的事：
  car 检出率 124%，我把多出的 24% 算成误差。但如果那些是真实存在、
  只是标注漏掉的车，那真实误差远小于 MAE 7.2 —— 我的报告会低估系统。

  这个只能人眼看：把标注框（绿）和检测框（红）画在同一张图上。

选图策略：
  找"切片检出 car 数 - 标注 car 数"最大的那张，多检最集中，最容易判断。

用法：
    "D:\anaconda3\envs\yolo\python.exe" render_compare.py
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
OUT = Path(r"D:/visdrone/rendered")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SLICE, OVERLAP = 640, 128
CONF = 0.25
NMS_IOU = 0.5


def tile_origins(total, size, step):
    if total <= size:
        return [0]
    xs = list(range(0, total - size, step))
    if not xs or xs[-1] != total - size:
        xs.append(total - size)
    return xs


def slice_detect(model, img, want="car"):
    h, w = img.shape[:2]
    step = SLICE - OVERLAP
    xs, ys = tile_origins(w, SLICE, step), tile_origins(h, SLICE, step)
    boxes, scores = [], []
    for oy in ys:
        for ox in xs:
            r = model.predict(img[oy:oy+SLICE, ox:ox+SLICE], verbose=False,
                              device="cpu", imgsz=SLICE, conf=CONF)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            keep = [i for i, c in enumerate(r.boxes.cls.tolist())
                    if model.names[int(c)] == want]
            if not keep:
                continue
            b = r.boxes.xyxy.cpu().numpy()[keep].copy()
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            boxes.append(b)
            scores.append(r.boxes.conf.cpu().numpy()[keep])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    B = torch.tensor(np.vstack(boxes), dtype=torch.float32)
    S = torch.tensor(np.concatenate(scores), dtype=torch.float32)
    k = nms(B, S, NMS_IOU)
    return B[k].numpy(), S[k].numpy()


def load_gt(ip, want="car"):
    ap = BASE / "annotations" / (ip.stem + ".txt")
    out = []
    if ap.exists():
        for line in ap.read_text().strip().splitlines():
            p = line.split(",")
            if len(p) < 6 or p[4].strip() != "1" or int(p[5]) != NAMES.index(want) + 1:
                continue
            x, y, w, h = (float(v) for v in p[:4])
            out.append((x, y, x + w, y + h))
    return np.array(out, dtype=float).reshape(-1, 4)


def draw(img, boxes, color, thickness=2, label=None, only_center=True):
    canvas = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if label and only_center:
            cx, cy = (x1 + x2) // 2 - 8, (y1 + y2) // 2
            cv2.putText(canvas, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
    return canvas


def main():
    model = YOLO(str(WEIGHT))
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted((BASE / "images").glob("*.jpg"))

    # 找多检最多的图（只看前 120 张，够找出典型了）
    print("扫描多检最多的图（前 120 张）...")
    best, best_img = -1, None
    for ip in imgs[:120]:
        img = cv2.imread(str(ip))
        B, _ = slice_detect(model, img)
        gt = load_gt(ip)
        diff = len(B) - len(gt)
        if diff > best:
            best, best_img = diff, ip
    print(f"选中 {best_img.name}：检出比标注多 {best} 个\n")

    img = cv2.imread(str(best_img))
    B, S = slice_detect(model, img)
    gt = load_gt(best_img)
    print(f"切片检出 car {len(B)} 个（conf>{CONF}），标注 car {len(gt)} 个")

    # 三张图：只看标注 / 只看检测 / 叠加
    h, w = img.shape[:2]
    a = draw(img, gt, (0, 200, 0), 2)
    b = draw(img, B, (0, 0, 255), 2)
    c = draw(draw(img, gt, (0, 200, 0), 2), B, (0, 0, 255), 2)

    def titled(image, text):
        bar = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, image])

    a = titled(a, f"GT only: {len(gt)} cars (green)")
    b = titled(b, f"Detected: {len(B)} cars (red)")
    c = titled(c, "Overlay: green=GT, red=detected")
    cv2.imwrite(str(OUT / "cmp_gt.jpg"), a)
    cv2.imwrite(str(OUT / "cmp_det.jpg"), b)
    cv2.imwrite(str(OUT / "cmp_overlay.jpg"), c)

    # 放大一块区域看细节（多检通常集中在密集区）
    if len(B):
        cx = int(np.median(B[:, 0]))
        cy = int(np.median(B[:, 1]))
        x0, y0 = max(0, cx - 320), max(0, cy - 240)
        crop = c[40:40 + min(480, h - y0), x0:x0 + min(640, w - x0)]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / "cmp_zoom.jpg"), crop)

    print(f"\n已保存到 {OUT}:")
    for f in ("cmp_gt.jpg", "cmp_det.jpg", "cmp_overlay.jpg", "cmp_zoom.jpg"):
        p = OUT / f
        if p.exists():
            print(f"  {f}  {p.stat().st_size/1024:.0f}KB")


if __name__ == "__main__":
    main()
