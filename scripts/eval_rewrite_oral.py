"""
查询改写实验：把学生口语转成教材术语，看检索能不能救回来

要验的假设：
  口语查询与学术正文的字面重合度接近 0，所以向量和 BM25 都失效（11~17%）。
  如果先把口语"翻译"成教材术语，检索就回到它擅长的区间。

  注意这和题库那次改写不是一回事：题库的查询本来就带术语，改写只增加噪声
  （实测 −3.3）；这次是跨语体，差距大得多。

防泄漏：
  改写时只给模型看学生原话，不给它看目标小节——否则等于把答案喂进去。

改写缓存：
  结果存 data/book/rewrites.json，重跑不用再调模型。

用法：python eval_rewrite_oral.py
"""

import json
from pathlib import Path

import numpy as np

import chunk as chunkmod
import embedder
import llm
from bm25 import BM25, minmax
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
CACHE = ROOT / "data" / "book" / "rewrites.json"

from eval_colloquial import TARGETS, MIN_OVERLAP, KS   # 复用同一批判数据

SYSTEM = """你会把学生的口语问题改写成教材检索用的查询。

规则：
1. 只输出改写后的查询本身，一行，不要解释、不要引号。
2. 必须使用这门课的学科术语（例如「朴素唯物主义自然观」「自然界的本原」
   「辩证唯物主义」「系统自然观」「科学技术异化」）。
3. 保留原问题的核心意图，不要添加原问题没有的内容。
4. 不要写成疑问句，写成"主题 + 具体方面"的短语。"""


def main():
    text = BOOK.read_text(encoding="utf-8")
    spans = chunkmod.chunk_text(text, 500, 100)
    chunks = [text[s:e] for s, e in spans]
    vec_idx = build(embedder.embed(chunks).numpy(), "numpy")
    bm25 = BM25(chunks)

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    print("改写：")
    rewritten = []
    for q, tgt in TARGETS:
        if q in cache:
            rw = cache[q]
        else:
            rw = llm.chat(SYSTEM, q, max_tokens=60).strip().strip('"“”')
            cache[q] = rw
        rewritten.append((q, rw, tgt))
        print(f"  {q}")
        print(f"    → {rw}")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def overlap(sp, a, b):
        return max(0, min(sp[1], b) - max(sp[0], a))

    def evaluate(get_query, tag):
        out = {}
        for name in ("vector", "bm25", "weighted"):
            hits = {k: 0 for k in KS}
            for q, rw, tgt in rewritten:
                query = get_query(q, rw)
                qv = embedder.embed_query(query).numpy().reshape(-1)
                qv = qv / np.linalg.norm(qv)
                vs = minmax(vec_idx.v @ qv)
                bs = minmax(bm25.scores(query))
                sc = {"vector": vs, "bm25": bs, "weighted": 0.4 * vs + 0.6 * bs}[name]
                top = np.argsort(-sc)[:5]
                for k in KS:
                    if any(overlap(spans[i], tgt, tgt + 1000) >= MIN_OVERLAP
                           for i in top[:k]):
                        hits[k] += 1
            n = len(rewritten)
            out[name] = {k: hits[k] / n for k in KS}
        return out

    before = evaluate(lambda q, rw: q, "原句")
    after = evaluate(lambda q, rw: rw, "改写")

    print()
    print(f"  {'配置':<12}{'原句@1':>9}{'改写@1':>9}{'原句@5':>9}{'改写@5':>9}")
    print(f"  {'-'*50}")
    for name in ("vector", "bm25", "weighted"):
        b, a = before[name], after[name]
        print(f"  {name:<12}{b[1]:>9.0%}{a[1]:>9.0%}{b[5]:>9.0%}{a[5]:>9.0%}")

    print()
    print("  逐条对比（用 weighted，看命中@5 有没有从 0 变 1）：")
    for q, rw, tgt in rewritten:
        def hit5(query):
            qv = embedder.embed_query(query).numpy().reshape(-1)
            qv = qv / np.linalg.norm(qv)
            vs = minmax(vec_idx.v @ qv)
            bs = minmax(bm25.scores(query))
            top = np.argsort(-(0.4 * vs + 0.6 * bs))[:5]
            return any(overlap(spans[i], tgt, tgt + 1000) >= MIN_OVERLAP for i in top)
        b, a = hit5(q), hit5(rw)
        mark = "救回" if (a and not b) else ("丢了" if (b and not a) else "—")
        print(f"    {mark:<4} 原{b and '✓' or '✗'} 改{a and '✓' or '✗'}  {q[:26]}")


if __name__ == "__main__":
    main()
