"""
对比实验 1：LLM Query 改写

方案 A（基线）  直接用学生口语问法去检索
方案 B（改写）  先用 DeepSeek 把口语问法改写成书面表述，再去检索

两套打分口径都报：严格（必须原题）+ 等价（同一知识点）

改写结果会缓存到 data/rewrite_cache.json，重跑不重复花钱。

运行：python scripts/exp_rewrite.py
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from embedder import QUERY_INSTRUCTION, embed
from llm import chat

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BOOK = ROOT / "评测工作台.xlsx"
CACHE = DATA / "rewrite_cache.json"
TOP_K = 5

SYSTEM = "你是检索系统的查询改写模块。你只做改写，不回答问题。"

TEMPLATE = """把下面这个学生的口语提问，改写成教材或试卷里的规范书面表述，用于在题库中检索。

要求：
- 只输出改写后的一句话，不加解释、不加前缀
- 保留核心概念，把口语词换成学术用语
- 不要回答问题，不要补充原问题里没有的信息
- 不超过 30 字

学生提问：{q}"""


def clean(s: str) -> str:
    s = s.strip().strip('"“”「」')
    for p in ("改写：", "改写:", "结果：", "结果:"):
        if s.startswith(p):
            s = s[len(p):].strip()
    return s.split("\n")[0].strip()


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


def build_rewrites(queries):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [q for q in queries if q not in cache]
    print(f"需改写 {len(todo)} 条（缓存命中 {len(queries) - len(todo)} 条）")

    if todo:
        def work(q):
            try:
                return q, clean(chat(SYSTEM, TEMPLATE.format(q=q), max_tokens=80))
            except Exception as e:
                print(f"  失败: {q} -> {e}")
                return q, q
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, (q, rw) in enumerate(ex.map(work, todo), 1):
                cache[q] = rw
                if i % 20 == 0:
                    print(f"  完成 {i}/{len(todo)}")
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def score(cases, equiv, vecs, id2idx, meta, texts, tag):
    qv = embed([QUERY_INSTRUCTION + t for t in texts])
    sims = (qv @ vecs.T).numpy()
    order = np.argsort(-sims, axis=1)

    rows = []
    for i, case in enumerate(cases):
        tid = id2idx[case["target"]]
        allowed_group = {tid} | {id2idx[e] for e in equiv[case["target"]] if e in id2idx}
        top = order[i][:TOP_K].tolist()
        rows.append({
            "target": case["target"],
            "query": case["query"],
            "text": texts[i],
            "rank_strict": top.index(tid) + 1 if tid in top else None,
            "rank_group": next((j + 1 for j, x in enumerate(top) if x in allowed_group), None),
            "top1_id": meta[top[0]]["id"],
            "top1_sim": float(sims[i][top[0]]),
        })

    n = len(rows)

    def rate(key, k):
        return sum(1 for r in rows if r[key] and r[key] <= k) / n

    res = {
        "tag": tag,
        "hit@1_strict": rate("rank_strict", 1),
        "hit@3_strict": rate("rank_strict", 3),
        "hit@5_strict": rate("rank_strict", 5),
        "hit@1_group": rate("rank_group", 1),
        "hit@3_group": rate("rank_group", 3),
        "hit@5_group": rate("rank_group", 5),
        "rows": rows,
    }
    print(f"{tag:<18}{res['hit@1_strict']:>9.1%}{res['hit@3_strict']:>9.1%}"
          f"{res['hit@1_group']:>10.1%}{res['hit@3_group']:>10.1%}")
    return res


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    cases, equiv = load_cases()
    queries = [c["query"] for c in cases]
    print(f"测试用例 {len(cases)} 条")
    print()

    cache = build_rewrites(list(dict.fromkeys(queries)))
    rewritten = [cache[q] for q in queries]

    print()
    print(f"{'方案':<18}{'hit@1严':>9}{'hit@3严':>9}{'hit@1等':>10}{'hit@3等':>10}")
    print("-" * 58)
    a = score(cases, equiv, vecs, id2idx, meta, queries, "A 原查询（基线）")
    b = score(cases, equiv, vecs, id2idx, meta, rewritten, "B LLM 改写")
    print("-" * 58)
    print(f"{'变化':<18}{b['hit@1_strict'] - a['hit@1_strict']:>+9.1%}"
          f"{b['hit@3_strict'] - a['hit@3_strict']:>+9.1%}"
          f"{b['hit@1_group'] - a['hit@1_group']:>+10.1%}"
          f"{b['hit@3_group'] - a['hit@3_group']:>+10.1%}")

    print()
    print("=== 改写样例（12 条）===")
    for c, rw in list(zip(cases, rewritten))[:12]:
        print(f"  原 : {c['query']}")
        print(f"  改 : {rw}")
        print()

    fixed = [i for i in range(len(cases))
             if (not a["rows"][i]["rank_strict"] or a["rows"][i]["rank_strict"] > 1)
             and b["rows"][i]["rank_strict"] == 1]
    broken = [i for i in range(len(cases))
              if a["rows"][i]["rank_strict"] == 1
              and (not b["rows"][i]["rank_strict"] or b["rows"][i]["rank_strict"] > 1)]
    print(f"=== 改写救回来的 {len(fixed)} 条（举 5 例）===")
    for i in fixed[:5]:
        print(f"  查询: {cases[i]['query']}")
        print(f"  改写: {rewritten[i]}")
        print()
    print(f"=== 改写搞坏的 {len(broken)} 条（举 5 例）===")
    for i in broken[:5]:
        print(f"  查询: {cases[i]['query']}")
        print(f"  改写: {rewritten[i]}")
        print(f"  改写后 top1: {b['rows'][i]['top1_id']}")
        print()

    (DATA / "eval_rewrite.json").write_text(json.dumps({
        "baseline": {k: v for k, v in a.items() if k != "rows"},
        "rewrite": {k: v for k, v in b.items() if k != "rows"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存: {DATA / 'eval_rewrite.json'}")


if __name__ == "__main__":
    main()
