"""
探针：定"该不该拒答"的阈值

问题背景：
  min-max 归一化出来的"综合分"是相对分——哪怕库里全是无关题，
  最高分也会被拉到 1.0。所以它不能用来判断"这道题到底像不像"。

  能用来判断的是【向量原始相似度】（0~1，绝对值）。

这个脚本量三组数据：
  A 该命中的查询（测试集里命中了的）→ top1 原始相似度分布
  B 不该命中的查询（测试集里没命中的）→ top1 原始相似度分布
  C 完全无关的问题（红烧肉、天气）→ top1 原始相似度

看 A 和 C 能不能被一条线分开。能分开，这条线就是拒答阈值。

运行：python scripts/probe_threshold.py
"""

import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BOOK = ROOT / "评测工作台.xlsx"

OUT_OF_SCOPE = [
    "红烧肉怎么做才好吃",
    "今天天气怎么样",
    "Python 怎么读取一个文件",
    "推荐一部好看的电影",
    "考研数学怎么复习",
    "我最近总是失眠怎么办",
]


def load_cases():
    wb = load_workbook(BOOK)
    ws = wb["1_改写任务"]
    cases = []
    for r in range(2, ws.max_row + 1):
        qid = ws.cell(row=r, column=1).value
        raw = ws.cell(row=r, column=6).value
        if not qid or not raw:
            continue
        for q in str(raw).split("|"):
            q = q.strip()
            if q:
                cases.append((qid, q))
    return cases


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    cases = load_cases()
    queries = [q for _, q in cases]

    qv = embed([QUERY_INSTRUCTION + q for q in queries]).numpy()
    sims = qv @ vecs.T
    top1 = sims.max(axis=1)

    hit, miss = [], []
    for i, (tid, _) in enumerate(cases):
        if int(np.argmax(sims[i])) == id2idx[tid]:
            hit.append(top1[i])
        else:
            miss.append(top1[i])

    oos = embed([QUERY_INSTRUCTION + q for q in OUT_OF_SCOPE]).numpy()
    oos_top1 = (oos @ vecs.T).max(axis=1)

    def stat(name, arr):
        arr = np.array(arr)
        print(f"  {name:<26} n={len(arr):3d}  最小 {arr.min():.3f}  "
              f"中位 {np.median(arr):.3f}  最大 {arr.max():.3f}")

    print("=" * 62)
    print("top-1 的【向量原始相似度】分布")
    print("=" * 62)
    stat("A 检索命中的查询", hit)
    stat("B 检索没命中的查询", miss)
    stat("C 完全无关的问题", oos_top1)
    print()
    print("=" * 62)
    print("C 组逐条（这些理应被拒答）")
    print("=" * 62)
    for q, s in zip(OUT_OF_SCOPE, oos_top1):
        print(f"  {s:.3f}  {q}")
    print()
    print("=" * 62)
    print("阈值参考：如果把阈值设在 T，会拒掉多少")
    print("=" * 62)
    hit_a, miss_a, oos_a = np.array(hit), np.array(miss), np.array(oos_top1)
    for t in (0.35, 0.40, 0.45, 0.50, 0.55):
        print(f"  T={t:.2f}  误拒(正确答案被拒) {np.mean(hit_a < t):5.1%}"
              f"   无关问题被拒 {np.mean(oos_a < t):5.1%}"
              f"   缺失命中被拒 {np.mean(miss_a < t):5.1%}")


if __name__ == "__main__":
    main()
