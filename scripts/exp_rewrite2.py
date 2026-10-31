"""
对比实验 1b：诊断"改写为什么变差"

复用 exp_rewrite.py 生成的改写缓存，不再调用 LLM，零成本多测一个方案：

  A 原查询          只用口语问法
  B 改写            只用改写后的书面问法
  C 原查询+改写      两者拼接（保留口语线索，同时补充书面表述）

如果 C > A，说明改写的"正式化"有一点用，但代价是丢掉了口语里的线索词；
如果 C < A，说明改写引入的术语错位比它带来的收益更大。

运行：python scripts/exp_rewrite2.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BOOK = ROOT / "评测工作台.xlsx"
CACHE = DATA / "rewrite_cache.json"
TOP_K = 5


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


def score(cases, equiv, vecs, id2idx, texts, tag):
    qv = embed([QUERY_INSTRUCTION + t for t in texts])
    sims = (qv @ vecs.T).numpy()
    order = np.argsort(-sims, axis=1)

    rows = []
    for i, case in enumerate(cases):
        tid = id2idx[case["target"]]
        allowed = {tid} | {id2idx[e] for e in equiv[case["target"]] if e in id2idx}
        top = order[i][:TOP_K].tolist()
        rows.append({
            "rank_strict": top.index(tid) + 1 if tid in top else None,
            "rank_group": next((j + 1 for j, x in enumerate(top) if x in allowed), None),
        })
    n = len(rows)

    def rate(key, k):
        return sum(1 for r in rows if r[key] and r[key] <= k) / n

    print(f"{tag:<20}{rate('rank_strict', 1):>9.1%}{rate('rank_strict', 3):>9.1%}"
          f"{rate('rank_group', 1):>10.1%}{rate('rank_group', 3):>10.1%}")
    return rows


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    cases, equiv = load_cases()
    queries = [c["query"] for c in cases]
    rewrites = [cache[q] for q in queries]
    concat = [f"{q} {r}" for q, r in zip(queries, rewrites)]

    print(f"测试用例 {len(cases)} 条")
    print()
    print(f"{'方案':<20}{'hit@1严':>9}{'hit@3严':>9}{'hit@1等':>10}{'hit@3等':>10}")
    print("-" * 60)
    a = score(cases, equiv, vecs, id2idx, queries, "A 原查询")
    b = score(cases, equiv, vecs, id2idx, rewrites, "B 仅改写")
    c = score(cases, equiv, vecs, id2idx, concat, "C 原查询+改写")
    print("-" * 60)

    # 逐条分析：改写相对原查询，检索结果是否变化
    changed = sum(1 for i in range(len(cases))
                  if a[i]["rank_strict"] != b[i]["rank_strict"])
    print(f"\n改写后有 {changed}/{len(cases)} 条检索结果发生变化"
          f"（{changed / len(cases):.0%}）—— 说明改写确实改变了向量，只是改错了方向")

    (DATA / "eval_rewrite_diag.json").write_text(json.dumps({
        "cases": len(cases),
        "changed_rank": changed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
