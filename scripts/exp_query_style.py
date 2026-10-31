"""
实验：查询写法对检索命中率的影响

目的：回答一个问题 —— 如果直接拿题库原题干去测，测出来的分数意味着什么？

在同一个索引（496 题）上，用三种"越来越像学生"的查询方式各跑 50 道题：

  风格 A  完整题干           （= 用原题测原题）
  风格 B  题干截断到前 18 字  （= 学生只记得半句）
  风格 C  只留实词关键词      （= 学生只记得几个词，还不会组织句子）

对比 hit@1 / hit@3，就能看出"原题测原题"这个做法虚高多少。

运行：python scripts/exp_query_style.py
"""

import json
import random
from pathlib import Path

import numpy as np

from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# 极简停用词表：只去掉那些"不携带检索信息"的词
STOP = set(
    "的 了 是 在 和 与 或 及 等 这 那 其 之 也 都 就 而 以 于 对 从 为 被 把 "
    "一个 一种 我们 你们 他们 它 他 她 我 你 可以 能够 应该 进行 具有 有 不 "
    "什么 哪些 如何 怎样 下列 关于 按照 通常 理解 实际上 就是 之间 一切 上 下".split()
)


def style_a(q):
    """完整题干"""
    return q["stem"]


def style_b(q):
    """截断到前 18 字，模拟只记得半句"""
    return q["stem"][:18]


def style_c(q):
    """只保留实词，模拟学生只记得几个关键词"""
    import jieba

    words = [w for w in jieba.cut(q["stem"]) if w.strip() and w not in STOP and len(w) >= 2]
    return " ".join(words[:12])


def evaluate(queries, targets, vecs, tag):
    qv = embed([QUERY_INSTRUCTION + s for s in queries])
    sims = (qv @ vecs.T).numpy()          # (N, 496)
    order = np.argsort(-sims, axis=1)     # 按相似度降序

    hit1 = hit3 = 0
    top1_sims = []
    misses = []
    for i, tid in enumerate(targets):
        top1 = order[i][0]
        top1_sims.append(float(sims[i][top1]))
        if top1 == tid:
            hit1 += 1
        if tid in set(order[i][:3].tolist()):
            hit3 += 1
        else:
            misses.append(i)

    n = len(targets)
    print(f"{tag:<26} hit@1 = {hit1 / n:5.1%}   hit@3 = {hit3 / n:5.1%}   "
          f"top1 平均相似度 = {np.mean(top1_sims):.3f}")
    return hit1 / n, hit3 / n, misses, queries


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    rng = random.Random(42)
    sample = rng.sample(meta, 50)
    # 只用单选/多选（判断题题干常常只有一句话，改写空间小）
    sample = [q for q in sample if q["type"] in ("single", "multi")][:40]

    targets = [id2idx[q["id"]] for q in sample]
    print(f"抽样 {len(sample)} 道题，索引共 {len(meta)} 条")
    print(f"随机基线：hit@1 ≈ 1/{len(meta)} = {1 / len(meta):.1%}")
    print()

    qa = [style_a(q) for q in sample]
    qb = [style_b(q) for q in sample]
    qc = [style_c(q) for q in sample]

    print("=== 三种查询写法在同一索引上的命中率 ===")
    evaluate(qa, targets, vecs, "A 完整题干（原题测原题）")
    evaluate(qb, targets, vecs, "B 截断到前 18 字")
    _, _, misses, _ = evaluate(qc, targets, vecs, "C 只留关键词")

    print()
    print("=== 风格 C 的实际查询长什么样（前 6 条）===")
    for i, (q, s) in enumerate(zip(sample, qc)):
        print(f"[{q['id']}]")
        print(f"  原题干 : {q['stem'][:60]}")
        print(f"  实际查询: {s}")
    print()
    print(f"=== 风格 C 有 {len(misses)} 道没进前三，举 5 例 ===")
    for i in misses[:5]:
        q = sample[i]
        qv = embed([QUERY_INSTRUCTION + qc[i]])
        sims = (qv @ vecs.T).numpy()[0]
        top = int(np.argmax(sims))
        print(f"查询  : {qc[i]}")
        print(f"应命中: [{q['id']}] {q['stem'][:40]}")
        print(f"实际  : [{meta[top]['id']}] {meta[top]['stem'][:40]}  ({sims[top]:.3f})")
        print()


if __name__ == "__main__":
    main()
