"""
口语化查询下的检索对照：向量 vs BM25 vs 融合

为什么必须做这个：
  144 条评测集的查询就是教材小节标题，标题原文原样出现在正文里，
  所以 BM25 的字面匹配天然占便宜（92% vs 向量 71%）。
  那不是"BM25 更好"，是"任务被写成了字面匹配"。

  这 20 条是学生口语（"古人觉得世界到底是啥做成的"），
  术语全被换掉了——如果 BM25 在这里掉下来、向量稳住，
  说明两者确实互补；如果 BM25 还领先，那融合就是多余复杂度。

命中判定（这次换了）：
  口语查询的答案可能落在小节任何位置，不是首句，
  所以判「块与该小节区间的重叠 ≥ 200 字」，而不是"含首句"。

  第 9、10 条（第谷、红烧肉）教材里没有，不计命中率，单独看会不会硬答。

用法：python eval_colloquial.py
"""

import json
from pathlib import Path

import numpy as np

import chunk as chunkmod
import embedder
from bm25 import BM25, minmax
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"

# (口语查询, 目标小节起始位置)
TARGETS = [
    ("古人觉得世界到底是啥做成的？", 7457),
    ("马克思说的那套看自然的道理，到底是个啥？", 15033),
    ("书上说世界是好多东西凑一块儿的，这是啥意思？", 18323),
    ("以前把世界当机器看，跟后来那套有啥不一样？", 14903),
    ("马克思他们那套想法是咋一步步弄出来的？", 14903),
    ("书里讲科技那部分，都说了些啥？", 70356),
    ("把世界看成一大堆东西，跟要爱护大自然，有啥不一样？", 22209),
    ("古人觉得世界是啥做的，跟把世界当机器看，差别在哪？", 14014),
    ("为啥读研还得上这门课，学它有啥用？", 5175),
    ("世界上的东西是一直在变，还是老样子？", 7457),
    ("人是从哪儿来的，动物真能变成人吗？", 7457),
    ("古人说世界是水啊火啊做的，这话靠谱吗？", 7457),
    ("以前那种“世界是机器”的说法，为啥后来没人信了？", 14014),
    ("人自己造出来的东西，算不算大自然的一部分？", 20266),
    ("为啥现在老说要爱护环境，跟这门课有啥关系？", 22209),
    ("科学和技术到底有啥不一样？", 30595),
    ("科学家是咋一步步把事情搞明白的？", 36322),
    ("科技越厉害，会不会也带来些麻烦事？", 29897),
]

REFUSE = ["第谷当年拿啥东西看星星来着？", "红烧肉咋做才好吃？"]

KS = (1, 3, 5)
MIN_OVERLAP = 200


def main():
    text = BOOK.read_text(encoding="utf-8")
    spans = chunkmod.chunk_text(text, 500, 100)
    chunks = [text[s:e] for s, e in spans]

    # 每个目标小节的结束位置 = 下一个标题的起点
    heads = sorted({t for _, t in TARGETS} | {21800, 22209, 23047, 24576, 26861,
                                              30595, 32864, 36322, 37000, 70356, 72000})
    # 用更密的标题表算区间：直接取所有标题
    import re
    pat = re.compile(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)")
    cur, all_heads = 0, []
    for line in text.splitlines():
        s = line.strip()
        if 4 <= len(s) <= 45 and pat.match(s) and not re.search(r"[·⋯…]{2,}", s):
            all_heads.append(cur)
        cur += len(line) + 1
    all_heads = sorted(set(all_heads))

    def section_end(start):
        """目标小节的范围：从标题起往后 1000 字。

        不能用"下一个标题"：OCR 把相邻标题挤在一起，有时只隔 12 个字符，
        区间就被压成 12 个字，任何块都凑不够 200 字重叠，导致全部判 0。
        """
        return min(len(text), start + 1000)

    vec_idx = build(embedder.embed(chunks).numpy(), "numpy")
    bm25 = BM25(chunks)

    def overlap(sp, a, b):
        return max(0, min(sp[1], b) - max(sp[0], a))

    print(f"切片 {len(chunks)} 块，口语查询 {len(TARGETS)} 条 + 拒答题 {len(REFUSE)} 条\n")
    print(f"  {'配置':<12}{'命中@1':>8}{'命中@3':>8}{'命中@5':>8}")
    print(f"  {'-'*36}")

    orders = {}
    for name in ("vector", "bm25", "weighted", "rrf"):
        hits = {k: 0 for k in KS}
        for q, tgt in TARGETS:
            qv = embedder.embed_query(q).numpy().reshape(-1)
            qv = qv / np.linalg.norm(qv)
            vs = minmax(vec_idx.v @ qv)
            bs = minmax(bm25.scores(q))

            if name == "vector":
                sc = vs
            elif name == "bm25":
                sc = bs
            elif name == "weighted":
                sc = 0.4 * vs + 0.6 * bs
            else:
                order_v = np.argsort(-vs)[:50]
                order_b = np.argsort(-bs)[:50]
                acc = {}
                for rank, i in enumerate(order_v):
                    acc[i] = acc.get(i, 0) + 1.0 / (60 + rank + 1)
                for rank, i in enumerate(order_b):
                    acc[i] = acc.get(i, 0) + 1.0 / (60 + rank + 1)
                top = sorted(acc, key=lambda i: -acc[i])[:5]
                orders[(name, q)] = top
                for k in KS:
                    if any(overlap(spans[i], tgt, section_end(tgt)) >= MIN_OVERLAP for i in top[:k]):
                        hits[k] += 1
                continue

            top = np.argsort(-sc)[:5].tolist()
            orders[(name, q)] = top
            for k in KS:
                if any(overlap(spans[i], tgt, section_end(tgt)) >= MIN_OVERLAP for i in top[:k]):
                    hits[k] += 1

        n = len(TARGETS)
        print(f"  {name:<12}{hits[1]/n:>8.0%}{hits[3]/n:>8.0%}{hits[5]/n:>8.0%}")

    # 拒答题：看最高分，判断会不会硬答
    print()
    print("  拒答题（教材里没有，看最高分是否明显偏低）：")
    for q in REFUSE:
        qv = embedder.embed_query(q).numpy().reshape(-1)
        qv = qv / np.linalg.norm(qv)
        top = np.argsort(-(vec_idx.v @ qv))[:3]
        print(f"    「{q}」 向量最高分 {float((vec_idx.v @ qv)[top[0]]):.3f}"
              f"  BM25 最高分 {float(bm25.scores(q).max()):.2f}")


if __name__ == "__main__":
    main()
