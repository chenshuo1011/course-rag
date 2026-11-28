"""
通用模式检出率：COCO val2017（模型没见过的数据）

为什么用 val2017 而不是 coco128：
  coco128 的图来自 train2017，而 yolov8m 就是在 train2017 上训练的——
  测出来的召回是"在见过的图上"的召回。val2017 是官方留出的验证集，
  模型训练时没碰过，这份数字才能拿去定措辞。

只取 60 张：
  完整 val2017 图片包 1GB，本机约 22KB/s，下完要 12 小时。
  所以只下标注包（241MB），再按类别分层挑 60 张，单张下载（约 160KB，可并行）。
  60 张的规模不足以代表 5000 张的分布，但足以判断"漏检为主还是误检为主"。

挑图规则：
  person 最多的 20 张、car 最多的 20 张、两者都适中（各 1~5 个）的 20 张。
  只挑极端的图会高估难度，只挑平均的会漏掉密集场景。

指标：IoU 0.5 匹配，算召回/精确/逐图计数 MAE，三档阈值。
  iscrowd=1 的标注要剔除——那是"人群区域"整体框，不是单个实例，
  留着会让召回率虚高到不可能的数。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_coco_val.py           # 全流程
    "D:\anaconda3\envs\yolo\python.exe" eval_coco_val.py --prepare # 只下数据
"""

import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("D:/visdrone/coco_val")
ANN_ZIP = Path("D:/visdrone/ann.zip")
WEIGHT = Path("D:/visdrone/yolov8m.pt")
IMG_URL = "http://images.cocodataset.org/val2017/{}.jpg"
N_IMG = 60
CONF_FLOOR = 0.05
THRESHOLDS = (0.5, 0.35, 0.25)
IMG_BASE = "http://images.cocodataset.org/val2017/"


def unpack_ann() -> Path:
    """从 annotations_trainval2017.zip 里只取 val 的实例标注。"""
    import zipfile
    dst = ROOT / "instances_val2017.json"
    if dst.exists():
        return dst
    if not ANN_ZIP.exists():
        raise SystemExit(f"缺标注包 {ANN_ZIP}，先下载：\n"
                         f"  curl -C - -o {ANN_ZIP} "
                         f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    with zipfile.ZipFile(ANN_ZIP) as z:
        name = "annotations/instances_val2017.json"
        ROOT.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(name))
    return dst


def pick_ids(data: dict, n: int = N_IMG) -> list[int]:
    """按 person / car / 中等密度分层挑图。"""
    by_img = defaultdict(Counter)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        by_img[a["image_id"]][cats[a["category_id"]]] += 1

    def take(key, k, cond):
        got = [(i, c) for i, c in by_img.items() if cond(c) and i not in picked]
        got.sort(key=lambda kv: -kv[1].get(key, 0))
        for i, _ in got[:k]:
            picked.add(i)

    picked: set[int] = set()
    take("person", n // 3, lambda c: c.get("person", 0) >= 5)
    take("car", n // 3, lambda c: c.get("car", 0) >= 5)
    rest = [(i, c) for i, c in by_img.items()
            if i not in picked and 1 <= c.get("person", 0) <= 5 and 1 <= c.get("car", 0) <= 5]
    rest.sort(key=lambda kv: -(kv[1].get("person", 0) + kv[1].get("car", 0)))
    for i, _ in rest[:n // 3]:
        picked.add(i)
    return sorted(picked)


def download(ids: list[int]) -> dict[int, Path]:
    d = ROOT / "images"
    d.mkdir(parents=True, exist_ok=True)
    out = {}

    def one(i):
        p = d / f"{i:012d}.jpg"
        if p.exists() and p.stat().st_size > 1000:
            return i, p
        url = IMG_URL.format(f"{i:012d}")
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=60) as r:
                    p.write_bytes(r.read())
                if p.stat().st_size > 1000:
                    return i, p
            except Exception:
                time.sleep(2)
        return i, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, p in ex.map(one, ids):
            if p:
                out[i] = p
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gt, pred, th=0.5):
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        best, bi = th, -1
        for i, p in enumerate(pred):
            if used[i] or p[0] != g[0]:
                continue
            v = iou(g[1:], p[1:])
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used[bi] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def main():
    prepare_only = "--prepare" in sys.argv
    ann = unpack_ann()
    data = json.loads(ann.read_text(encoding="utf-8"))
    print(f"val2017：{len(data['images'])} 张图，{len(data['annotations'])} 个标注")

    ids = pick_ids(data)
    print(f"挑出 {len(ids)} 张（person 多 / car 多 / 中等密度 各三分之一）")
    paths = download(ids)
    print(f"下载完成 {len(paths)}/{len(ids)}")
    if prepare_only:
        return

    # GT：只留挑中的图，剔除 iscrowd
    cats = {c["id"]: c["name"] for c in data["categories"]}
    gt = defaultdict(list)
    for a in data["annotations"]:
        if a["image_id"] not in paths or a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((cats[a["category_id"]], x, y, x + w, y + h))

    model = YOLO(str(WEIGHT))
    names = model.names
    preds = {}
    for k, (i, p) in enumerate(sorted(paths.items()), 1):
        img = cv2.imread(str(p))
        r = model.predict(img, verbose=False, device="cpu", imgsz=960, conf=CONF_FLOOR)[0]
        preds[i] = [(names[int(c)], *b, float(s)) for b, c, s in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        if k % 15 == 0:
            print(f"  推理 {k}/{len(paths)}")

    tot_gt = sum(len(v) for v in gt.values())
    print(f"\n参与评测 {len(paths)} 张，{tot_gt} 个标注框（已剔除 iscrowd）\n")

    for th in THRESHOLDS:
        stat = {}
        for i in paths:
            g = gt.get(i, [])
            p = [x[:5] for x in preds[i] if x[5] >= th]
            gc, pc = Counter(x[0] for x in g), Counter(x[0] for x in p)
            for cls in set(gc) | set(pc):
                s = stat.setdefault(cls, [0, 0, 0, []])
                s[0] += gc.get(cls, 0)
                s[1] += pc.get(cls, 0)
                s[3].append(abs(pc.get(cls, 0) - gc.get(cls, 0)))
            for cls in set(gc) | set(pc):
                g_c = [x for x in g if x[0] == cls]
                p_c = [x for x in p if x[0] == cls]
                stat[cls][2] += match(g_c, p_c)[0]

        print(f"=== conf >= {th} ===")
        print(f"{'类别':<14}{'标注':>5}{'命中':>5}{'预测':>5}{'召回':>7}{'精确':>7}{'MAE':>6}")
        for cls, (n_gt, n_pr, n_tp, errs) in sorted(stat.items(), key=lambda kv: -kv[1][0]):
            if n_gt < 10:
                continue
            print(f"{cls:<14}{n_gt:>5}{n_tp:>5}{n_pr:>5}"
                  f"{n_tp / n_gt:>6.0%}{(n_tp / n_pr if n_pr else 0):>7.0%}"
                  f"{sum(errs) / len(errs):>6.2f}")
        tg = sum(v[0] for v in stat.values())
        tp = sum(v[2] for v in stat.values())
        pr = sum(v[1] for v in stat.values())
        print(f"{'全部':<14}{tg:>5}{tp:>5}{pr:>5}{tp / tg:>6.0%}"
              f"{(tp / pr if pr else 0):>7.0%}\n")


if __name__ == "__main__":
    main()
