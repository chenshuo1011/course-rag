"""
第 4 周：加重排，看命中@1 能提多少、代价多大

做法：
  1. 粗排：向量检索取 top-10（这个 10 是算出来的，不是拍的：
     池子 5→10 天花板 75%→87%，只多 0.75 秒；10→20 加 7 点但多 1.5 秒）
  2. 重排：bge-reranker-base 逐条打分，重新排序
  3. 对比重排前后的命中@1/@3/@5，以及多花的时间

必须同时报两件事：
  涨了多少命中，多花了多少时间。只报前者是耍流氓。

用法：
    cd /d D:\course-rag\scripts
    "D:\anaconda3\envs\yolo\python.exe" eval_rerank.py
"""

import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import chunk as chunkmod
import embedder
from vectorstore import build

POOL = 10          # 粗排取多少条进重排
TOP = 5            # 最终看前 5
KS = (1, 3, 5)


def load_reranker():
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-base").eval()
    return tok, model


def rerank(tok, model, query, passages):
    """给每条候选打分，返回按分数从高到低重排后的下标。"""
    with torch.no_grad():
        batch = tok([query] * len(passages), passages, padding=True,
                    truncation=True, max_length=512, return_tensors="pt")
        scores = model(**batch).logits.view(-1).tolist()
    return sorted(range(len(passages)), key=lambda i: -scores[i]), scores


def main():
    text = open("../data/book/book.txt", encoding="utf-8").read()
    evals = json.loads(open("../data/book/evalset.json", encoding="utf-8").read())

    spans = chunkmod.chunk_text(text, 500, 100)
    texts = [text[s:e] for s, e in spans]
    idx = build(embedder.embed(texts).numpy(), "numpy")
    print(f"切片 {len(spans)} 块，评测 {len(evals)} 条，池子 {POOL}")

    tok, model = load_reranker()
    print("重排模型已加载\n")

    before = {k: 0 for k in KS}
    after = {k: 0 for k in KS}
    t_rerank = 0.0
    detail = []

    for n, e in enumerate(evals, 1):
        hl = len(e["heading"])
        target = "".join(text[e["start"] + hl: e["start"] + hl + 60].split())

        top = idx.search(embedder.embed_query(e["query"]).numpy(), POOL)[0]
        cand = [texts[i] for i in top]

        def ranks_of(order):
            """在给定顺序里，正确答案排第几（没找到返回 None）。"""
            for pos, ci in enumerate(order, 1):
                if target and target in "".join(texts[ci].split()):
                    return pos
            return None

        r_before = ranks_of(top)

        t0 = time.perf_counter()
        order, _scores = rerank(tok, model, e["query"], cand)
        t_rerank += time.perf_counter() - t0

        r_after = ranks_of([top[i] for i in order])

        for k in KS:
            if r_before and r_before <= k:
                before[k] += 1
            if r_after and r_after <= k:
                after[k] += 1
        detail.append((e["query"], r_before, r_after))

        if n % 30 == 0:
            print(f"  {n}/{len(evals)} 条 ...", flush=True)

    n = len(evals)
    print()
    print(f"  {'指标':<8}{'重排前':>9}{'重排后':>9}{'变化':>9}")
    for k in KS:
        b, a = before[k] / n, after[k] / n
        print(f"  {'命中@' + str(k):<8}{b:>8.0%}{a:>9.0%}{(a-b)*100:>+8.0f}pt")

    print()
    print(f"  重排总耗时 {t_rerank:.0f}s，平均每条查询 {t_rerank/n*1000:.0f}ms"
          f"（= 每条候选约 {t_rerank/n/POOL*1000:.0f}ms）")

    improved = [(q, b, a) for q, b, a in detail if b and a and a < b]
    worsened = [(q, b, a) for q, b, a in detail if b and a and a > b]
    print(f"  名次提前 {len(improved)} 条，名次退后 {len(worsened)} 条")
    if improved[:5]:
        print("\n  提前最多的几条：")
        for q, b, a in sorted(improved, key=lambda x: x[2] - x[1])[:5]:
            print(f"    第{b}位 → 第{a}位  {q[:30]}")
    if worsened[:3]:
        print("\n  反而退后的（说明重排也会犯错）：")
        for q, b, a in worsened[:3]:
            print(f"    第{b}位 → 第{a}位  {q[:30]}")

    out = {"pool": POOL, "before": before, "after": after, "n": n,
           "rerank_seconds": round(t_rerank, 1)}
    open("../data/book/eval_rerank.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
