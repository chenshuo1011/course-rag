"""
通用模式（COCO 80 类）的检出率评测：用 COCO128 的标注当尺子

为什么需要这个：
  航拍模式有 VisDrone 548 张的实测检出率（car 113%、pedestrian 93%…），
  所以敢说"至少/约"。但通用模式只有 0 依据，只能干巴巴说"模型检出 N 个"——
  用户拿真实照片问，我们连它偏多偏少都不知道。

数据：coco128（126 张 COCO train2017 图 + YOLO 格式标注，6.7MB）。

**⚠️ 这套数字有数据泄漏，只能当上界读。**
  coco128 的图来自 COCO train2017，而 yolov8m 就是在 train2017 上训练的——
  这些图模型见过。所以测出来的召回率是"在训过的图上"的召回，真实场景只会更低。
  正确的做法是换 COCO val2017（模型没见过），但那份数据 1GB+，本机下载只有 26KB/s。
  在没有干净数据之前，**这份结果只用来定位问题（漏检多还是误检多），不用来定措辞**。

  另一个局限：只有 126 张，人物/车辆以外的类别样本极少（多数类不到 20 个框），
  结论只对人、车这类常见目标稍有参考价值。

方法：
  每张图只推理一次（conf 压到 0.05 保留全部候选），再按不同阈值过滤后计数。
  这样一次推理能算出多个阈值的结果，128 张总共约 1 分钟。

指标（做了 IoU 匹配，不用"预测数/标注数"那种粗口径）：
  召回 recall   = 被匹配上的标注数 / 标注总数   ← 漏了多少
  精确 precision = 匹配上的预测数 / 预测总数    ← 报的里面有多少是真的
  逐图 MAE      = 每张图 |预测数 - 标注数| 的平均

  为什么不用总数比：预测 164 个、标注 254 个，比值 65%——
  但这里面可能既有漏检、又有误检，两者混在一起看不出问题在哪。
  VisDrone 那边用的是总数比，因为它只看一眼规模；这里要定"能不能说至少 N 个"，
  必须分清漏检和误检。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_general.py
"""

from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("D:/visdrone/coco128")
WEIGHT = Path("D:/visdrone/yolov8m.pt")
CONF_FLOOR = 0.05
THRESHOLDS = (0.5, 0.35, 0.25)


def load_gt() -> dict[str, list[tuple[int, float, float, float, float]]]:
    """每张图的标注框：[(cls, x1, y1, x2, y2)]，归一化坐标（和 iou() 的约定一致）。"""
    out = {}
    for p in sorted((ROOT / "labels" / "train2017").glob("*.txt")):
        boxes = []
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, w, h = (float(v) for v in parts[1:5])
                boxes.append((int(parts[0]), cx - w / 2, cy - h / 2,
                              cx + w / 2, cy + h / 2))
        out[p.stem] = boxes
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match(gt_boxes, pred_boxes, iou_th=0.5):
    """贪心匹配，同类且 IoU>阈值算命中。返回 (TP, FP, FN)。"""
    used = [False] * len(pred_boxes)
    tp = 0
    for g in gt_boxes:
        best, bi = iou_th, -1
        for i, p in enumerate(pred_boxes):
            if used[i] or p[0] != g[0]:
                continue
            v = iou(g[1:], p[1:])
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used[bi] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    gt = load_gt()
    # 有的标注没有对应图片（coco128 里存在这种情况），先剔掉，免得推理时报文件不存在
    have = {p.stem for p in (ROOT / "images" / "train2017").glob("*.jpg")}
    missing = [s for s in gt if s not in have]
    for s in missing:
        gt.pop(s)
    if missing:
        print(f"跳过 {len(missing)} 个没有图片的标注：{missing[:3]}")
    model = YOLO(str(WEIGHT))
    # 类名直接用模型自带的：coco128 的类别 id 就是 COCO 的 id，和权重里的顺序一致
    names = model.names

    # 每张图推理一次，按图保存每个框的类别与置信度
    preds = {}
    for i, stem in enumerate(sorted(gt), 1):
        img = ROOT / "images" / "train2017" / f"{stem}.jpg"
        r = model.predict(img, verbose=False, device="cpu", imgsz=960, conf=CONF_FLOOR)[0]
        H, W = r.orig_shape
        # 同样统一成归一化 xyxy
        preds[stem] = [
            (int(c), b[0] / W, b[1] / H, b[2] / W, b[3] / H, float(s))
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                               r.boxes.conf.tolist())
        ]
        if i % 32 == 0:
            print(f"  推理中 {i}/{len(gt)}")

    print(f"\n标注 {sum(len(v) for v in gt.values())} 个框，覆盖 {len(gt)} 张图")
    print("⚠️  这些图来自 COCO train2017，yolov8m 训练时见过——召回率只能当上界读，"
          "不能直接拿去定措辞\n")

    for th in THRESHOLDS:
        stat = {}
        for stem, gboxes in gt.items():
            pboxes = [b[:5] for b in preds[stem] if b[5] >= th]
            tp, fp, fn = match(gboxes, pboxes)
            for g in gboxes:
                s = stat.setdefault(g[0], [0, 0, 0, []])
                s[0] += 1
            for p in pboxes:
                stat.setdefault(p[0], [0, 0, 0, []])[1] += 1
            # 逐类再算一次，只为拿每类的 TP/FP/FN
            for cls in {g[0] for g in gboxes} | {p[0] for p in pboxes}:
                g_c = [g for g in gboxes if g[0] == cls]
                p_c = [p for p in pboxes if p[0] == cls]
                t_c, f_p, f_n = match(g_c, p_c)
                s = stat[cls]
                s[2] += t_c
            # 逐图计数误差
            from collections import Counter as _C
            gc, pc = _C(g[0] for g in gboxes), _C(p[0] for p in pboxes)
            for cls in set(gc) | set(pc):
                stat.setdefault(cls, [0, 0, 0, []])[3].append(
                    abs(pc.get(cls, 0) - gc.get(cls, 0)))

        print(f"=== conf >= {th} ===")
        print(f"{'类别':<14}{'标注':>5}{'命中':>5}{'预测':>5}{'召回':>7}{'精确':>7}{'MAE':>6}")
        rows = sorted(stat.items(), key=lambda kv: -kv[1][0])
        for cls, (n_gt, n_pr, n_tp, errs) in rows:
            if n_gt < 10:
                continue
            rec = n_tp / n_gt
            pre = n_tp / n_pr if n_pr else 0
            print(f"{names[cls]:<14}{n_gt:>5}{n_tp:>5}{n_pr:>5}"
                  f"{rec:>6.0%}{pre:>7.0%}{sum(errs)/len(errs):>6.2f}")
        tg = sum(v[0] for v in stat.values())
        tp_all = sum(v[2] for v in stat.values())
        pr_all = sum(v[1] for v in stat.values())
        print(f"{'全部':<14}{tg:>5}{tp_all:>5}{pr_all:>5}"
              f"{tp_all / tg:>6.0%}{(tp_all / pr_all if pr_all else 0):>7.0%}")
        print()


if __name__ == "__main__":
    main()
