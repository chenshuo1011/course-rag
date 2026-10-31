"""
命令行问答演示：随便用大白话问，看系统找到哪道题、答案是什么

用的是评测里表现最好的配置：向量 0.5 + BM25 0.5

运行（在 D:/course-rag/scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" ask.py

输入 exit 退出。
"""

import json
from pathlib import Path

import numpy as np

from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TOP_K = 3
W_VEC = 0.5

TYPE_CN = {"single": "单选题", "multi": "多选题", "judge": "判断题"}


def doc_text(m):
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    return " ".join(parts)


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    print("加载索引与 BM25 ...")
    bm = BM25([doc_text(m) for m in meta])
    print(f"就绪：{len(meta)} 道题\n")

    print("=" * 62)
    print("自然辩证法 题库答疑（随便用大白话问）")
    print("输入 exit 退出")
    print("=" * 62)
    print()

    while True:
        try:
            q = input("你问：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            break

        # 先转成 numpy 再做矩阵乘：
        # 直接把 numpy 数组和 torch 张量相乘，numpy 2.0 会报 DeprecationWarning
        qv = embed([QUERY_INSTRUCTION + q]).numpy()[0]
        vs = vecs @ qv
        bs = bm.scores(q)
        score = W_VEC * minmax(vs) + (1 - W_VEC) * minmax(bs)
        order = np.argsort(-score)[:TOP_K]

        print()
        for rank, i in enumerate(order, 1):
            m = meta[int(i)]
            print(f"  {rank}. [{TYPE_CN.get(m['type'], m['type'])}] {m['stem'][:52]}")
            ans = m["answer_text"] or m["answer"]
            print(f"     答案：{str(ans)[:60]}")
            mark = "★" if rank == 1 and score[i] > 0.55 else " "
            print(f"     {mark} 相似度 {vs[i]:.3f} | BM25 {bs[i]:.2f} | 综合 {score[i]:.3f}")
        print()


if __name__ == "__main__":
    main()
