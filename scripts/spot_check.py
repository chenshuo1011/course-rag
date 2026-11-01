"""
抽查：生成质量到底怎么样

之前所有数字都是「检索命中率」（正确答案有没有排进前三），
不等于「学生问一句，它答对了」。

这个脚本抽 10 条用例跑完整流程（检索 + 生成），把检索结果、目标题的
标准答案、模型回答并排列出来，供人工判断。

抽样是分层的：
  5 条检索命中的 —— 看回答对不对
  4 条检索没命中的 —— 看它是拒答还是硬答（硬答最危险）
  1 条题库完全无关的 —— 看拒答兜不兜得住

运行：python scripts/spot_check.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from ask import SYSTEM, TEMPLATE, TOP_K, W_VEC, build_context, doc_text
from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed
from llm import chat

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BOOK = ROOT / "评测工作台.xlsx"
OUT = DATA / "spot_check.json"

OOS = "红烧肉怎么做才好吃"


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
    return cases


def retrieve(q, meta, vecs, bm):
    qv = embed([QUERY_INSTRUCTION + q]).numpy()[0]
    vs = vecs @ qv
    bs = bm.scores(q)
    combined = W_VEC * minmax(vs) + (1 - W_VEC) * minmax(bs)

    import re
    def key(m):
        return re.sub(r"[\s　（）()、，。？,.:：\"'']", "", m["stem"])

    order, seen = [], set()
    for i in np.argsort(-combined):
        m = meta[int(i)]
        k = key(m)
        if k in seen:
            continue
        seen.add(k)
        order.append(int(i))
        if len(order) >= TOP_K:
            break
    return [meta[i] for i in order], [float(vs[i]) for i in order]


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}
    byid = {m["id"]: m for m in meta}

    cases = load_cases()
    bm = BM25([doc_text(m) for m in meta])

    # 先算每条用例的检索结果，分成命中 / 未命中
    qv = embed([QUERY_INSTRUCTION + c["query"] for c in cases]).numpy()
    sims = qv @ vecs.T
    hit, miss = [], []
    for i, c in enumerate(cases):
        top = int(np.argmax(sims[i]))
        (hit if top == id2idx[c["target"]] else miss).append(c)
    print(f"150 条用例中：检索命中 {len(hit)} 条，未命中 {len(miss)} 条")

    rng = random.Random(7)
    picked = ([(c, "hit") for c in rng.sample(hit, 5)]
              + [(c, "miss") for c in rng.sample(miss, 4)])
    picked.append(({"target": None, "query": OOS}, "oos"))

    results = []
    for idx, (c, kind) in enumerate(picked, 1):
        hits, vsim = retrieve(c["query"], meta, vecs, bm)
        ctx = build_context(hits)
        try:
            ans = chat(SYSTEM, TEMPLATE.format(context=ctx, query=c["query"]), max_tokens=300)
        except Exception as e:
            ans = f"<调用失败: {e}>"

        tgt = byid.get(c["target"]) if c["target"] else None
        results.append({
            "no": idx,
            "kind": kind,
            "query": c["query"],
            "target_id": c["target"],
            "target_stem": tgt["stem"] if tgt else "",
            "target_answer": (tgt["answer_text"] or tgt["answer"]) if tgt else "",
            "target_explanation": tgt.get("explanation", "") if tgt else "",
            "top1_id": hits[0]["id"],
            "top1_stem": hits[0]["stem"],
            "top1_answer": hits[0]["answer_text"] or hits[0]["answer"],
            "top1_sim": vsim[0],
            "top3": [{"id": h["id"], "stem": h["stem"][:50]} for h in hits],
            "answer": ans,
        })

        print(f"\n{'=' * 70}")
        print(f"[{idx}] ({kind}) 问：{c['query']}")
        if tgt:
            print(f"    目标题: {tgt['stem'][:56]}")
            print(f"    标准答案: {str(tgt['answer_text'] or tgt['answer'])[:56]}")
            print(f"    解析: {tgt.get('explanation', '')[:70]}")
        print(f"    检索top1: [{hits[0]['id']}] {hits[0]['stem'][:50]}")
        print(f"    回答: {ans}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n\n已保存: {OUT}")


if __name__ == "__main__":
    main()
