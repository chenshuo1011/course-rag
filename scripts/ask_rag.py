"""
带生成的问答：检索 + 大模型组织答案 + 拒答

和 ask.py 的区别：
  ask.py     检索 → 直接甩出题库里的题目和答案
  ask_rag.py 检索 → 把 top-3 的题目和答案交给 DeepSeek → 让它用口语解释

为什么要加"拒答"：
  检索总会返回 top-3，哪怕库里根本没有相关内容。如果直接把这些题喂给模型，
  它会照着错题编出一段通顺但完全错误的回答——这比答不出来危险得多。

为什么不用相似度阈值做拒答：
  实测过（probe_threshold.py），"检索对了"和"检索错了"的 top-1 相似度分布
  完全重叠（0.5~0.79），分不开。所以改用让模型判断相关性。

运行（在 D:/course-rag/scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" ask_rag.py
"""

import json
from pathlib import Path

import numpy as np

from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed
from llm import chat

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TOP_K = 3
W_VEC = 0.5

SYSTEM = (
    "你是课程答疑助手。你只能依据给定的题目和答案回答，"
    "绝不补充题目里没有的信息。"
)

TEMPLATE = """下面是题库里检索到的几道题：

{context}

学生的问题：{query}

请按以下要求回答：
1. 如果上面有题目能回答这个问题，就基于它的答案，用学生听得懂的口语解释，120 字以内
2. 如果上面的题目都答不了这个问题，只输出这七个字：题库里没有找到
3. 不要说"根据第几题"之类的过渡语，直接给答案
4. 不要编造题目里没有的内容

关于判断题的说明：题型标为"判断题"的题目，**题干本身就是一条知识陈述**，
答案里的"正确/错误"只是判断这条陈述是否成立。
你可以从题干里提取知识来回答问题，但要说明该说法是否成立；
如果几道判断题互相矛盾，以成立的那条为准。"""


def doc_text(m):
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    return " ".join(parts)


def build_context(hits):
    lines = []
    for i, m in enumerate(hits, 1):
        lines.append(f"【题{i}】{m['stem']}")
        if m.get("options"):
            opts = "；".join(f"{k}. {v}" for k, v in m["options"].items())
            lines.append(f"选项：{opts}")
        lines.append(f"答案：{m['answer_text'] or m['answer']}")
        lines.append("")
    return "\n".join(lines)


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    print("加载索引与 BM25 ...")
    bm = BM25([doc_text(m) for m in meta])
    print(f"就绪：{len(meta)} 道题")

    print("=" * 64)
    print("课程答疑（带生成）  输入 exit 退出")
    print("可以问：其他课的问题、生活问题，看它会不会硬答")
    print("=" * 64)
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

        qv = embed([QUERY_INSTRUCTION + q]).numpy()[0]
        vs = vecs @ qv
        bs = bm.scores(q)
        combined = W_VEC * minmax(vs) + (1 - W_VEC) * minmax(bs)
        order = np.argsort(-combined)[:TOP_K]
        hits = [meta[int(i)] for i in order]

        print()
        print("── 检索到的 3 道题 " + "─" * 44)
        for i, (m, idx) in enumerate(zip(hits, order), 1):
            print(f"  {i}. {m['stem'][:46]}")
            print(f"     原始相似度 {vs[idx]:.3f}")

        print()
        print("── 大模型的回答 " + "─" * 46)
        try:
            ans = chat(SYSTEM, TEMPLATE.format(context=build_context(hits), query=q),
                       max_tokens=300)
            print(f"  {ans}")
        except Exception as e:
            print(f"  调用失败：{e}")
        print()


if __name__ == "__main__":
    main()
