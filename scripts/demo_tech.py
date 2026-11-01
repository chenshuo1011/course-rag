"""
把项目里用到的技术"拆开给你看"——用真实数字，不用公式吓人

三段：
  【1】文字怎么变成 512 个数字（embedding）
  【2】怎么算两段文字"有多像"（余弦相似度）
  【3】关键词匹配怎么打分（BM25），包括手算一遍

运行（在 D:/course-rag/scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" demo_tech.py
"""

import json
from pathlib import Path

import numpy as np

from bm25 import BM25, minmax, tokenize
from embedder import embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

SENTS = ["系统有啥特点", "系统", "红烧肉的做法"]


def main():
    # ---------------- 1. embedding ----------------
    print("=" * 64)
    print("【1】文字怎么变成数字：一句话 → 512 个小数")
    print("=" * 64)
    vecs = embed(SENTS).numpy()          # 已归一化，长度都是 1
    print(f"三句话，每句变成一个 {vecs.shape[1]} 维向量。下面只截取前 8 维：\n")
    for s, v in zip(SENTS, vecs):
        head = " ".join(f"{x:+.2f}" for x in v[:8])
        print(f"  「{s}」")
        print(f"     [{head}  ...]")
    print()
    print("  注意：这些数字没有直观含义。单个数字不重要，")
    print("  「两句话的 512 个数整体像不像」才重要。")
    print()

    # ---------------- 2. 余弦相似度 ----------------
    print("=" * 64)
    print("【2】怎么算「有多像」：两个向量的夹角余弦")
    print("=" * 64)
    print("  做法就是把两句话的 512 个数分别相乘再相加（点积）。")
    print("  因为向量已经归一化成长度 1，点积就等于余弦相似度。\n")
    for i in range(len(SENTS)):
        for j in range(i + 1, len(SENTS)):
            sim = float(vecs[i] @ vecs[j])
            note = "← 意思接近" if sim > 0.7 else ("← 无关" if sim < 0.3 else "")
            print(f"  {SENTS[i]:<10} ↔ {SENTS[j]:<10}  相似度 {sim:.3f}  {note}")
    print()
    print("  这就是整个检索的地基：意思相近 → 分数高 → 排前面。")
    print("  模型从没读过你这两句话，但它读过海量中文，")
    print("  学会了把「意思相近的话」放到相近的位置。")
    print()

    # ---------------- 3. BM25 ----------------
    print("=" * 64)
    print("【3】关键词匹配：BM25 怎么打分")
    print("=" * 64)
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))

    def doc_text(m):
        parts = [m["stem"]]
        if m.get("options"):
            parts.extend(f"{k}. {v}" for k, v in m["options"].items())
        return " ".join(parts)

    bm = BM25([doc_text(m) for m in meta])
    print(f"  496 道题被切成词，词表共 {len(bm.idf)} 个不同的词")
    print(f"  平均每道题 {bm.avgdl:.0f} 个词\n")

    print("  第一件事：算每个词的「稀有度」（IDF）")
    print("  越少见的词，权重越高——这是 BM25 的核心直觉\n")
    for w in ["系统", "特点", "自然辩证法", "第谷"]:
        if w in bm.idf:
            df = sum(1 for tf in bm.tfs if w in tf)
            print(f"    「{w}」出现在 {df:3d} 道题里 → IDF = {bm.idf[w]:.2f}")
    print()
    print("  「第谷」只出现在少数题里，所以它一出现就能锁定目标；")
    print("  「系统」「特点」满地都是，区分力弱，权重自然低。")
    print()

    print("  第二件事：手算一道题的分数")
    q = "系统 特点"
    scores = bm.scores(q)
    best = int(np.argmax(scores))
    m = meta[best]
    toks = bm.tfs[best]
    print(f"    查询词: {q}")
    print(f"    命中题目: {m['stem'][:34]}")
    print(f"    这道题共 {int(bm.lens[best])} 个词")
    print()
    total = 0.0
    for w in tokenize(q):
        if w not in bm.idf:
            print(f"    「{w}」不在词表里，跳过")
            continue
        f = toks.get(w, 0)
        if f == 0:
            print(f"    「{w}」在这道题里没出现 → 贡献 0")
            continue
        contrib = bm.idf[w] * f * (bm.k1 + 1) / (
            f + bm.k1 * (1 - bm.b + bm.b * bm.lens[best] / bm.avgdl)
        )
        total += contrib
        print(f"    「{w}」出现 {f} 次，IDF {bm.idf[w]:.2f} → 贡献 {contrib:.2f}")
    print(f"\n    合计 BM25 分数 = {total:.2f}（和程序算的 {scores[best]:.2f} 对得上）")
    print()

    # ---------------- 4. 融合 ----------------
    print("=" * 64)
    print("【4】两个分数怎么合成一个")
    print("=" * 64)
    print("  问题：向量相似度在 0~1 之间，BM25 可能是 0~20，量纲不同，")
    print("        直接相加没有意义（BM25 会完全压死向量）。\n")
    print("  做法：各自先归一化到 0~1（min-max），再加权相加：")
    print("        综合分 = 0.5 × 向量分 + 0.5 × BM25 分")
    demo = np.array([2.0, 5.0, 8.0, 20.0])
    print(f"        例：BM25 原始分 {demo.tolist()}")
    print(f"            归一化后   {[round(float(x), 2) for x in minmax(demo)]}")
    print()
    print("  这样两路分数才在同一个尺度上，权重才有意义。")


if __name__ == "__main__":
    main()
