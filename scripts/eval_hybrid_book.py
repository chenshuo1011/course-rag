"""
混合检索：向量 + BM25 在教材上的对比

为什么要试：
  现在瓶颈在召回（命中@5 75%，池子天花板 87%），而重排只能救池子里的。
  向量检索认意思不认字——「朴素唯物主义自然观的观点」和
  「朴素唯物主义自然观的特征」在向量空间几乎同一个点，
  所以「（五）辩证唯物主义自然观的作用」这类邻居节总被排前面。
  BM25 反过来只认字，对这种"字面几乎一致"的标题式查询更准。

  题库链路早就验过这组：纯向量 46.7% / 纯 BM25 53.3% / 融合 58.7%。
  这次看教材上是不是同样的规律。

四种配置：
  vector  —— 只用向量
  bm25    —— 只用 BM25
  weighted—— 两边分数 min-max 归一化后加权（w 是向量的权重）
  rrf     —— 按排名融合，只用名次不用分数（分数量纲不同时更稳）

用法：python eval_hybrid_book.py
"""

import json
from pathlib import Path

import numpy as np

import chunk as chunkmod
import embedder
from bm25 import BM25, minmax
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
EVAL = ROOT / "data" / "book" / "evalset.json"

KS = (1, 3, 5)


def main():
    text = BOOK.read_text(encoding="utf-8")
    evals = json.loads(EVAL.read_text(encoding="utf-8"))
    spans = chunkmod.chunk_text(text, 500, 100)
    chunks = [text[s:e] for s, e in spans]
    print(f"切片 {len(chunks)} 块，评测 {len(evals)} 条")

    vec_idx = build(embedder.embed(chunks).numpy(), "numpy")
    bm25 = BM25(chunks)
    qvecs = [embedder.embed_query(e["query"]).numpy() for e in evals]

    def topk_from(scores, k):
        return np.argsort(-scores)[:k].tolist()

    def hit_at(cfg_top, target):
        for k in KS:
            if any(target and target in "".join(chunks[i].split()) for i in cfg_top[k]):
                pass  # 具体统计在下面统一做

    results = {}

    for name in ("vector", "bm25", "weighted", "rrf"):
        hits = {k: 0 for k in KS}
        for e, qv in zip(evals, qvecs):
            hl = len(e["heading"])
            target = "".join(text[e["start"] + hl: e["start"] + hl + 60].split())

            if name == "vector":
                order = topk_from(vec_idx.v @ (qv.reshape(-1) / np.linalg.norm(qv)), 50)
            elif name == "bm25":
                order = topk_from(bm25.scores(e["query"]), 50)
            elif name == "weighted":
                vs = minmax(vec_idx.v @ (qv.reshape(-1) / np.linalg.norm(qv)))
                bs = minmax(bm25.scores(e["query"]))
                order = topk_from(0.5 * vs + 0.5 * bs, 50)
            else:  # rrf
                rv = topk_from(vec_idx.v @ (qv.reshape(-1) / np.linalg.norm(qv)), 50)
                rb = topk_from(bm25.scores(e["query"]), 50)
                score = {}
                for rank, i in enumerate(rv):
                    score[i] = score.get(i, 0) + 1.0 / (60 + rank + 1)
                for rank, i in enumerate(rb):
                    score[i] = score.get(i, 0) + 1.0 / (60 + rank + 1)
                order = sorted(score, key=lambda i: -score[i])[:50]

            for k in KS:
                if any(target and target in "".join(chunks[i].split()) for i in order[:k]):
                    hits[k] += 1
        results[name] = {k: hits[k] / len(evals) for k in KS}

    print()
    print(f"  {'配置':<12}{'命中@1':>9}{'命中@3':>9}{'命中@5':>9}")
    print(f"  {'-'*40}")
    for name, r in results.items():
        print(f"  {name:<12}{r[1]:>9.0%}{r[3]:>9.0%}{r[5]:>9.0%}")

    # 权重扫描：看向量占多少最合适
    print()
    print("  权重扫描（向量权重 / BM25 权重）")
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        hits5 = 0
        for e, qv in zip(evals, qvecs):
            hl = len(e["heading"])
            target = "".join(text[e["start"] + hl: e["start"] + hl + 60].split())
            vs = minmax(vec_idx.v @ (qv.reshape(-1) / np.linalg.norm(qv)))
            bs = minmax(bm25.scores(e["query"]))
            top = topk_from(w * vs + (1 - w) * bs, 5)
            if any(target and target in "".join(chunks[i].split()) for i in top):
                hits5 += 1
        print(f"    {w:.1f} / {1-w:.1f}   →  命中@5 {hits5/len(evals):>5.0%}")

    out = ROOT / "data" / "book" / "eval_hybrid.json"
    out.write_text(json.dumps({k: {str(kk): vv for kk, vv in v.items()}
                               for k, v in results.items()},
                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
