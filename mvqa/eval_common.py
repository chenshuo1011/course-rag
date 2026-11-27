"""
评测共用的东西：从 548 张里挑出有代表性的 20 张

为什么用标注挑而不是跑检测：
  挑图只需要"这张图里大概有什么"，标注就是现成的答案，读 548 个 txt 不到 1 秒；
  跑检测要 548×110ms ≈ 1 分钟。而且用标注挑不会偏向"模型容易检出的图"。

挑选策略：**按目标密度分层抽样**。
  把 548 张按标注目标总数排序，等距取 20 张（每 27 张取一张）。
  这样得到的密度分布和全集一致，稀疏场景和密集场景都按比例进样本。

  第一版不是这么做的，而是"车最多的 4 张 + 行人最多的 4 张…"，
  结果样本平均 158 个目标/张，而全集只有 71——全是极端密集图，
  会把"能给出数字"的比例估得偏高，也会漏掉"空旷场景乱报"这类问题。
"""

from collections import Counter
from pathlib import Path

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def load_counts() -> dict[str, Counter]:
    """每张图各类别的标注数量（只算 score==1 的有效标注）。"""
    out = {}
    for ap in sorted((BASE / "annotations").glob("*.txt")):
        c = Counter()
        for line in ap.read_text().strip().splitlines():
            p = line.split(",")
            if len(p) >= 6 and p[4].strip() == "1":
                cat = int(p[5])
                if 1 <= cat <= 10:
                    c[NAMES[cat - 1]] += 1
        out[ap.stem] = c
    return out


def pick_images(n: int = 20) -> list[tuple[str, str]]:
    """按目标密度分层抽 n 张。返回 [(图片名, 说明)]。"""
    counts = load_counts()
    items = sorted(counts.items(), key=lambda kv: sum(kv[1].values()))
    if n >= len(items):
        return [(s, f"{sum(c.values())} 个目标") for s, c in items]
    step = len(items) / n
    picked = []
    for i in range(n):
        stem, c = items[min(int(i * step + step / 2), len(items) - 1)]
        top = "、".join(f"{k}{v}" for k, v in c.most_common(2)) or "空"
        picked.append((stem, f"{sum(c.values())} 个（{top}）"))
    return picked
