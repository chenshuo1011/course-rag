"""
对比实验 4：把「解析」放进检索索引，会不会更好

两种建索引方式：
  A  题干 + 选项          （现状）
  B  题干 + 选项 + 解析    （解析里含同义术语和完整表述）

直觉上 B 应该更好——解析里常常出现题目没写的术语。
但前面已经证明过"往短查询里加词会被质心化"，所以这次也可能变差，必须量。

两路都用同样的混合方式（向量 0.5 + BM25 0.5）评测。

运行：python scripts/exp_explanation.py
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
W_VEC = 0.5


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


def doc_text(m, with_exp=False):
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    if with_exp and m.get("explanation"):
        parts.append(m["explanation"])
    return " ".join(parts)


def evaluate(cases, equiv, id2idx, ranks, tag):
    n = len(cases)
    rows = []
    for i, c in enumerate(cases):
        tid = id2idx[c["target"]]
        allowed = {tid} | {id2idx[e] for e in equiv[c["target"]] if e in id2idx}
        top = ranks[i][:TOP_K]
        rows.append({
            "rs": top.index(tid) + 1 if tid in top else None,
            "rg": next((j + 1 for j, x in enumerate(top) if x in allowed), None),
        })

    def rate(k, kk):
        return sum(1 for r in rows if r[k] and r[k] <= kk) / n

    print(f"{tag:<34}{rate('rs',1):>9.1%}{rate('rs',3):>9.1%}{rate('rg',1):>10.1%}{rate('rg',3):>10.1%}")
    return rate("rs", 1), rate("rs", 3)


def main():
    v2 = json.loads((DATA / "questions_v2.json").read_text(encoding="utf-8"))["questions"]
    id2idx = {q["id"]: i for i, q in enumerate(v2)}
    print(f"含解析题库 {len(v2)} 题")

    cases, equiv = load_cases()
    queries = [c["query"] for c in cases]
    print(f"测试用例 {len(cases)} 条")

    print("\n编码查询...")
    qv = embed([QUERY_INSTRUCTION + q for q in queries]).numpy()

    print(f"{'方案':<34}{'hit@1严':>9}{'hit@3严':>9}{'hit@1等':>10}{'hit@3等':>10}")
    print("-" * 72)

    results = {}
    for tag, with_exp in (("A 索引=题干+选项", False), ("B 索引=题干+选项+解析", True)):
        texts = [doc_text(q, with_exp) for q in v2]
        vpath = DATA / ("vectors_v2.npy" if with_exp else "vectors_v2_noexp.npy")
        if vpath.exists():
            print(f"  [{tag}] 复用已有向量 {vpath.name}")
            vecs = np.load(vpath)
        else:
            print(f"  [{tag}] 编码 {len(texts)} 道题（约 40 秒）...")
            vecs = embed(texts).numpy()
            np.save(vpath, vecs)

        bm = BM25(texts)
        bvec = qv @ vecs.T
        bm25 = np.array([bm.scores(q) for q in queries])
        ranks = [np.argsort(-(W_VEC * minmax(bvec[i]) + (1 - W_VEC) * minmax(bm25[i]))).tolist()
                 for i in range(len(cases))]
        results[tag] = evaluate(cases, equiv, id2idx, ranks, tag)

    print("-" * 72)
    (a1, a3), (b1, b3) = results["A 索引=题干+选项"], results["B 索引=题干+选项+解析"]
    print(f"{'变化':<33}{b1 - a1:>+9.1%}{b3 - a3:>+9.1%}")


if __name__ == "__main__":
    main()
