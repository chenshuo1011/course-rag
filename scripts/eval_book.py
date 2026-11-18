"""
教材 RAG 的检索评测：不同切片参数下的 hit@k

指标设计（改了三版才对）：
  v1 hit@k = 块与目标区间有重叠 → 块越大越容易沾边，800 字的块几乎什么都覆盖，虚高
  v2 strict@k = 块的起点落在目标区间内 → 块越小越容易命中，这是几何必然，偏向小块
  v3 命中@k = 块里包含该节**首句**（标题后 60 字）→ 与块大小无关，才是"找到了这一节的内容"

配一个"噪声量"一起看：每次检索喂给模型的字数 = k × 平均块长。
块大命中高但噪声多，块小噪声少但容易缺上下文——这才是真取舍。

评测集来自教材小节标题（build_evalset.py），不是模型编的问题，
所以数字不会被"出题时抄了原文"污染。

用法：python eval_book.py
"""

import time
from pathlib import Path

import numpy as np

import chunk as chunkmod
import embedder
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
EVAL = ROOT / "data" / "book" / "evalset.json"

import json

PARAMS = [(200, 0), (300, 0), (300, 100), (500, 100), (800, 100), (1200, 100)]
KS = (1, 3, 5)


def main():
    text = BOOK.read_text(encoding="utf-8")
    evals = json.loads(EVAL.read_text(encoding="utf-8"))
    print(f"教材 {len(text)} 字，评测 {len(evals)} 条\n")

    # 查询只编码一次，所有参数共用
    qvecs = [embedder.embed_query(e["query"]).numpy() for e in evals]

    print(f"  {'切片':<11}{'块数':>4}{'均长':>6}   "
          + "  ".join(f"命中@{k}" for k in KS)
          + "     喂入字数@3   @5")
    print(f"  {'-'*70}")

    rows = []
    for size, ov in PARAMS:
        spans = chunkmod.chunk_text(text, size, ov)
        texts = [text[s:e] for s, e in spans]
        vecs = embedder.embed(texts).numpy()
        idx = build(vecs, "numpy")

        hits = {k: 0 for k in KS}
        for e, qv in zip(evals, qvecs):
            top = idx.search(qv, max(KS))[0]
            # 真值 = 该节标题之后的首句（去掉空白后比较，OCR 可能有零散空格）
            head_len = len(e["heading"])
            target = "".join(text[e["start"] + head_len: e["start"] + head_len + 60].split())
            for k in KS:
                tops = top[:k]
                if any(target and target in "".join(texts[i].split()) for i in tops):
                    hits[k] += 1

        n = len(evals)
        avg = round(np.mean([e2 - s2 for s2, e2 in spans]))
        h = "  ".join(f"{hits[k]/n:>5.0%}" for k in KS)
        print(f"  {f'{size}/{ov}':<11}{len(spans):>4}{avg:>6}   {h}"
              f"      {3*avg:>5}  {5*avg:>5}")
        rows.append({"size": size, "overlap": ov, "chunks": len(spans),
                     "avg": avg, "hit": {k: hits[k]/n for k in KS}})

    out = ROOT / "data" / "book" / "eval_chunk.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已存 {out.name}")


if __name__ == "__main__":
    main()
