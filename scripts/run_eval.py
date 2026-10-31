"""
读取「评测工作台.xlsx」，跑第一版基线评测

两套打分口径：
  严格   top-k 里必须有目标题本身
  等价   top-k 里有目标题，或有被标为"同一知识点"的候选题

输出基线数字 + 失败清单，并存成 data/eval_baseline.json 供后续对比。

运行：python scripts/run_eval.py
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
OUT = DATA / "eval_baseline.json"

TOP_K = 5


def load_book():
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
        t = ws2.cell(row=r, column=1).value
        c = ws2.cell(row=r, column=3).value
        flag = ws2.cell(row=r, column=6).value
        if flag == "是":
            equiv[t].add(c)
    return cases, equiv


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    cases, equiv = load_book()
    print(f"测试用例 {len(cases)} 条（来自 {len({c['target'] for c in cases})} 道题）")
    print(f"等价组：{sum(len(v) for v in equiv.values())} 条边，涉及 {len(equiv)} 道题")
    print()

    qv = embed([QUERY_INSTRUCTION + c["query"] for c in cases])
    sims = (qv @ vecs.T).numpy()
    order = np.argsort(-sims, axis=1)

    rows = []
    for i, case in enumerate(cases):
        target = case["target"]
        if target not in id2idx:
            continue
        tid = id2idx[target]
        allowed_strict = {tid}
        allowed_group = {tid} | {id2idx[e] for e in equiv[target] if e in id2idx}

        top = order[i][:TOP_K].tolist()
        rank_strict = top.index(tid) + 1 if tid in top else None
        rank_group = next((j + 1 for j, x in enumerate(top) if x in allowed_group), None)

        rows.append({
            "target": target,
            "query": case["query"],
            "rank_strict": rank_strict,
            "rank_group": rank_group,
            "top1_id": meta[top[0]]["id"],
            "top1_sim": float(sims[i][top[0]]),
            "top1_stem": meta[top[0]]["stem"],
            "target_stem": meta[tid]["stem"],
        })

    n = len(rows)

    def rate(key, k):
        return sum(1 for r in rows if r[key] and r[key] <= k) / n

    print("=" * 62)
    print(f"{'打分口径':<22}{'hit@1':>10}{'hit@3':>10}{'hit@5':>10}")
    print("-" * 62)
    print(f"{'严格（必须原题）':<20}{rate('rank_strict', 1):>10.1%}"
          f"{rate('rank_strict', 3):>10.1%}{rate('rank_strict', 5):>10.1%}")
    print(f"{'等价（同一知识点）':<18}{rate('rank_group', 1):>10.1%}"
          f"{rate('rank_group', 3):>10.1%}{rate('rank_group', 5):>10.1%}")
    print("=" * 62)
    print(f"随机基线 hit@1 ≈ {1 / len(meta):.2%}")
    print()

    print("=== 等价口径下 hit@3 失败的查询（最多列 8 条）===")
    fails = [r for r in rows if not r["rank_group"] or r["rank_group"] > 3]
    for r in fails[:8]:
        print(f"查询  : {r['query']}")
        print(f"应命中: {r['target']}  {r['target_stem'][:44]}")
        print(f"实际  : {r['top1_id']}  {r['top1_stem'][:44]}  ({r['top1_sim']:.3f})")
        print()
    print(f"共 {len(fails)} 条未进前三，占 {len(fails) / n:.1%}")

    OUT.write_text(json.dumps({
        "method": "bge-small-zh-v1.5 纯向量检索",
        "cases": n,
        "hit@1_strict": rate("rank_strict", 1),
        "hit@3_strict": rate("rank_strict", 3),
        "hit@1_group": rate("rank_group", 1),
        "hit@3_group": rate("rank_group", 3),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存: {OUT}")


if __name__ == "__main__":
    main()
