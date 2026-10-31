"""
对比实验 3：BM25 与向量检索混合

做一个真正有意义的对照：
  A  纯向量（基线）
  F  纯 BM25
  G  0.7·向量 + 0.3·BM25
  H  0.5·向量 + 0.5·BM25
  I  0.3·向量 + 0.7·BM25

两路分数先各自 min-max 归一化到 0~1，再加权求和——
否则向量相似度（约 0.3~0.8）和 BM25 分数（可能 0~20）量纲不同，直接加没有意义。

运行：python scripts/exp_hybrid.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BOOK = ROOT / "评测工作台.xlsx"
TOP_K = 5


def doc_text(m: dict) -> str:
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    return " ".join(parts)


def load_cases():
    wb = load_workbook(BOOK)
    ws1 = wb["1_改写任务"]
    cases = []
    for r in range(2, ws1.max_row + 1):
        qid = ws1.cell(row=r, column=1).value
        raw = ws1.cell(row=r, column=6).value
        if not qid or not raw:
            continue
        for q in str(raw).split("|"):
            q = q.strip()
            if q:
                cases.append({"target": qid, "query": q})
    ws2 = wb["2_等价组核对"]
    equiv = defaultdict(set)
    for r in range(2, ws2.max_row + 1):
        if ws2.cell(row=r, column=6).value == "是":
            equiv[ws2.cell(row=r, column=1).value].add(ws2.cell(row=r, column=3).value)
    return cases, equiv


def evaluate(cases, equiv, id2idx, ranks, tag):
    n = len(cases)
    rows = []
    for i, case in enumerate(cases):
        tid = id2idx[case["target"]]
        allowed = {tid} | {id2idx[e] for e in equiv[case["target"]] if e in id2idx}
        top = ranks[i][:TOP_K]
        rows.append({
            "rank_strict": top.index(tid) + 1 if tid in top else None,
            "rank_group": next((j + 1 for j, x in enumerate(top) if x in allowed), None),
        })

    def rate(key, k):
        return sum(1 for r in rows if r[key] and r[key] <= k) / n

    print(f"{tag:<24}{rate('rank_strict', 1):>9.1%}{rate('rank_strict', 3):>9.1%}"
          f"{rate('rank_group', 1):>10.1%}{rate('rank_group', 3):>10.1%}")
    return rows


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    cases, equiv = load_cases()
    queries = [c["query"] for c in cases]
    print(f"测试用例 {len(cases)} 条")

    print("建 BM25 索引...")
    bm = BM25([doc_text(m) for m in meta])
    print(f"  词表大小 {len(bm.idf)}，平均文档长度 {bm.avgdl:.1f} 词")

    print("编码查询...")
    qv = embed([QUERY_INSTRUCTION + q for q in queries])
    vec_sims = (qv @ vecs.T).numpy()

    print()
    print(f"{'方案':<24}{'hit@1严':>9}{'hit@3严':>9}{'hit@1等':>10}{'hit@3等':>10}")
    print("-" * 64)

    arms = {}

    def run(scores_list, tag):
        ranks = [np.argsort(-s).tolist() for s in scores_list]
        arms[tag] = evaluate(cases, equiv, id2idx, ranks, tag)
        return ranks

    # 纯向量
    run([vec_sims[i] for i in range(len(cases))], "A 纯向量（基线）")

    # 纯 BM25
    bm_scores = [bm.scores(q) for q in queries]
    bm_ranks = run(bm_scores, "F 纯 BM25")

    # 混合
    for w in (0.7, 0.5, 0.3):
        mixed = [
            w * minmax(vec_sims[i]) + (1 - w) * minmax(bm_scores[i])
            for i in range(len(cases))
        ]
        run(mixed, f"向量 {w:.1f} + BM25 {1 - w:.1f}")
    print("-" * 64)

    # 看 BM25 修好了哪些、搞坏了哪些
    a_ranks = [np.argsort(-vec_sims[i]).tolist() for i in range(len(cases))]
    fixed = [i for i in range(len(cases))
             if (a_ranks[i][:1] != [id2idx[cases[i]["target"]]])
             and bm_ranks[i][:1] == [id2idx[cases[i]["target"]]]]
    print(f"\n=== BM25 单独就能命中、而向量命不中的 {len(fixed)} 条（举 6 例）===")
    for i in fixed[:6]:
        print(f"  查询: {queries[i]}")
        print(f"  目标: {cases[i]['target']}")
        print()

    (DATA / "eval_hybrid.json").write_text(json.dumps({
        "cases": len(cases),
        "note": "min-max 归一化后加权求和",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
