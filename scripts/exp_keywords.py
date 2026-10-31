"""
对比实验 2：术语扩展（而不是语气改写）

上一轮发现"把口语改写成书面语"没用，甚至更差。
诊断：改写的动作是【语气正式化】（啥→什么），不是【词汇对齐】（最牛→高峰）。
而检索的鸿沟恰恰在词汇上。

这一轮换个做法：让 LLM 把问题拆成【检索关键词 + 教材可能用的同义术语】。

  A 原查询               基线
  D 仅关键词扩展          只用扩展出来的词
  E 原查询+关键词扩展      两者拼接

扩展结果缓存到 data/keyword_cache.json。

运行：python scripts/exp_keywords.py
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
CACHE = DATA / "keyword_cache.json"
TOP_K = 5

SYSTEM = "你是检索系统的查询扩展模块。你只输出关键词，不作解释。"

TEMPLATE = """学生问了一门课（自然辩证法）的问题。请把这个问题转换成一组检索关键词。

要求：
- 提取问题里的核心概念
- 补上教材/试卷里可能用来表达同一意思的术语（同义词、上位词、相关专有名词）
- 只输出关键词，用空格分隔，最多 12 个词
- 不要输出句子，不要解释，不要回答问题

学生问题：{q}"""


def clean(s: str) -> str:
    s = s.strip().strip('"“”「」')
    return " ".join(s.replace("，", " ").replace("、", " ").replace("\n", " ").split())


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


def build(queries):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [q for q in queries if q not in cache]
    print(f"需扩展 {len(todo)} 条（缓存命中 {len(queries) - len(todo)} 条）")

    if todo:
        def work(q):
            try:
                return q, clean(chat(SYSTEM, TEMPLATE.format(q=q), max_tokens=120))
            except Exception as e:
                print(f"  失败: {q} -> {e}")
                return q, q
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, (q, kw) in enumerate(ex.map(work, todo), 1):
                cache[q] = kw
                if i % 30 == 0:
                    print(f"  完成 {i}/{len(todo)}")
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


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

    print(f"{tag:<22}{rate('rank_strict', 1):>9.1%}{rate('rank_strict', 3):>9.1%}"
          f"{rate('rank_group', 1):>10.1%}{rate('rank_group', 3):>10.1%}")
    return rows


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    cases, equiv = load_cases()
    queries = [c["query"] for c in cases]
    cache = build(list(dict.fromkeys(queries)))
    kws = [cache[q] for q in queries]

    print(f"\n测试用例 {len(cases)} 条")
    print()
    print(f"{'方案':<22}{'hit@1严':>9}{'hit@3严':>9}{'hit@1等':>10}{'hit@3等':>10}")
    print("-" * 62)
    score(cases, equiv, vecs, id2idx, queries, "A 原查询")
    score(cases, equiv, vecs, id2idx, kws, "D 仅关键词扩展")
    score(cases, equiv, vecs, id2idx,
          [f"{q} {k}" for q, k in zip(queries, kws)], "E 原查询+关键词扩展")
    print("-" * 62)

    print("\n=== 关键词扩展样例（10 条）===")
    for c, k in list(zip(cases, kws))[:10]:
        print(f"  原 : {c['query']}")
        print(f"  扩 : {k}")
        print()


if __name__ == "__main__":
    main()
