"""
检索：输入一个问题，返回最相关的 top-k 道题

用法：
    python scripts/query.py                    # 跑内置测试问题
    python scripts/query.py "你的问题"          # 查单个问题

注意这里只做【检索】，还没接大模型生成。
检索是 RAG 的地基 —— 检索不到正确的题，后面生成再好也没用。
"""

import json
import sys
from pathlib import Path

import numpy as np

from embedder import embed_query

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

_meta = None
_vecs = None


def load():
    global _meta, _vecs
    if _meta is None:
        _meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
        _vecs = np.load(DATA / "vectors.npy")
    return _meta, _vecs


def search(query: str, top_k: int = 3):
    """返回 [(分数, 题目dict), ...]"""
    meta, vecs = load()
    qv = embed_query(query).numpy()[0]
    # 向量已 L2 归一化，点积就是余弦相似度
    scores = vecs @ qv
    idx = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), meta[i]) for i in idx]


def show(query: str, top_k: int = 3):
    print("=" * 64)
    print(f"问题：{query}")
    print("=" * 64)
    for rank, (score, q) in enumerate(search(query, top_k), 1):
        print(f"\n  [{rank}] 相似度 {score:.4f}  |  {q['lecture']}  {q['id']}")
        print(f"      题干：{q['stem'][:70]}")
        for k, v in q["options"].items():
            print(f"        {k}. {v}")
        print(f"      答案：{q['answer']}  →  {q['answer_text']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        # 内置测试：故意用"口语化"问法，模拟学生的真实提问
        tests = [
            "自然辩证法到底是什么东西",
            "那个把科学和马克思主义连起来的桥梁是啥",
            "中国古代科技最牛的是哪个朝代",
            "马克思是不是一个人搞出了马克思主义",
        ]
        for t in tests:
            show(t, top_k=2)
