"""
扫描 overlap 的取值，看命中率在哪饱和

为什么单独扫这一维：
  之前只测了 0/50/100（对应 300 字的 0%/17%/33%），就下了"100 够用"的结论。
  但重叠的危害是"相邻块高度重复，挤占 top-k 名额"，
  这个危害在大重叠时才显现——不扫到 200/250 就看不到拐点。

同时扫两个 size（300 / 500），因为最优 overlap 可能随块长变化。

用法：python sweep_overlap.py
"""

import json
from pathlib import Path

import numpy as np

import chunk as chunkmod
import embedder
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
EVAL = ROOT / "data" / "book" / "evalset.json"

KS = (1, 3, 5)


def main():
    text = BOOK.read_text(encoding="utf-8")
    evals = json.loads(EVAL.read_text(encoding="utf-8"))
    qvecs = [embedder.embed_query(e["query"]).numpy() for e in evals]
    print(f"教材 {len(text)} 字，评测 {len(evals)} 条\n")

    for size in (300, 500):
        print(f"── size = {size} ──")
        print(f"  {'overlap':>8}{'占比':>7}{'块数':>6}"
              f"{'命中@1':>9}{'命中@3':>9}{'命中@5':>9}{'喂入@5':>9}")
        for ov in (0, 50, 100, 150, 200, 250):
            if ov >= size:
                continue
            spans = chunkmod.chunk_text(text, size, ov)
            texts = [text[s:e] for s, e in spans]
            vecs = embedder.embed(texts).numpy()
            idx = build(vecs, "numpy")

            hits = {k: 0 for k in KS}
            for e, qv in zip(evals, qvecs):
                top = idx.search(qv, max(KS))[0]
                hl = len(e["heading"])
                target = "".join(text[e["start"] + hl: e["start"] + hl + 60].split())
                for k in KS:
                    if any(target and target in "".join(texts[i].split()) for i in top[:k]):
                        hits[k] += 1

            n = len(evals)
            avg = int(np.mean([b - a for a, b in spans]))
            print(f"  {ov:>8}{ov/size:>7.0%}{len(spans):>6}"
                  f"{hits[1]/n:>9.0%}{hits[3]/n:>9.0%}{hits[5]/n:>9.0%}"
                  f"{5*avg:>9}")
        print()


if __name__ == "__main__":
    main()
