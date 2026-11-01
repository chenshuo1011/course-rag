"""
课程题库答疑 —— 统一入口

默认流程：提问 → 混合检索 top-3 → 把题目、答案、解析交给大模型 → 口语回答
加参数 --no-gen 则只做检索，不调大模型（用来观察"检索"这一步本身的效果）

    "D:/anaconda3/envs/yolo/python.exe" ask.py            # 检索 + 生成
    "D:/anaconda3/envs/yolo/python.exe" ask.py --no-gen    # 只检索

为什么要有拒答：
  检索永远会返回 top-3，哪怕库里根本没有相关内容。把这些题直接喂给模型，
  它会照着编一段通顺但完全错误的话——比答不出来危险得多。
  所以要求模型：答不了就输出"题库里没有找到"。

为什么不用相似度阈值做拒答：
  实测（probe_threshold.py）"检索对了"和"检索错了"的 top-1 相似度分布
  完全重叠（0.5~0.79），分不开。只能让模型自己判断相关性。
"""

import json
import sys
from pathlib import Path

import numpy as np

from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed
from llm import chat

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TOP_K = 3
W_VEC = 0.5

WITH_GEN = "--no-gen" not in sys.argv

SYSTEM = (
    "你是课程答疑助手。你只能依据给定的题目、答案和解析回答，"
    "绝不补充材料里没有的信息。"
)

TEMPLATE = """下面是题库里检索到的几道题：

{context}

学生的问题：{query}

请按以下要求回答：
1. 如果材料能直接回答这个问题，就综合答案和解析，用学生听得懂的口语解释，**80 字以内**
2. 如果材料只能回答问题的一部分，就说明材料里确定的那部分，并明确指出"题库里没讲到"的是什么
   （例如：材料讲了阿拉伯人保存古希腊文献，但没专门谈哲学——这种情况也要答，不要拒答）
3. 只有当材料与问题完全没有关系时，才输出这七个字：题库里没有找到
4. **只回答学生问的那一点。** 其他题目里的相关信息不要捎带——除非少了它就说不清楚
5. 不要说"根据第几题"之类的过渡语，直接回答
6. 不要编造材料里没有的内容

关于题目格式的说明：
- 判断题的"答案"是"正确/错误"，真正的知识在**解析**里——解析说明了该说法错在哪、正确表述是什么。
- 选择题的"答案"是选项内容，解析说明了为什么选它、其他为什么不选。
- 如果几道题的说法互相矛盾，以被判定为"正确"的那条、或解析给出的完整表述为准。"""


def doc_text(m):
    """和建索引时保持完全一致，否则向量与 BM25 两路输入不对等"""
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    if m.get("explanation"):
        parts.append(m["explanation"])
    return " ".join(parts)


def build_context(hits):
    lines = []
    for i, m in enumerate(hits, 1):
        lines.append(f"【题{i}】{m['stem']}")
        if m.get("options"):
            lines.append("选项：" + "；".join(f"{k}. {v}" for k, v in m["options"].items()))
        lines.append(f"答案：{m['answer_text'] or m['answer']}")
        if m.get("explanation"):
            lines.append(f"解析：{m['explanation']}")
        lines.append("")
    return "\n".join(lines)


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    print("加载索引与 BM25 ...")
    bm = BM25([doc_text(m) for m in meta])
    print(f"就绪：{len(meta)} 道题 | 模式：{'检索 + 生成' if WITH_GEN else '仅检索'}")

    print("=" * 64)
    print("课程题库答疑    输入 exit 退出")
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

        # 按题干去重：题库里有 6 道题题干完全相同（multi-第四讲-2~7），
        # 不去重的话它们会占满 top-3，把别的候选挤掉。
        # 只比题干，不比选项/解析——它们选项不同但问的是同一件事。
        import re as _re
        def stem_key(m):
            return _re.sub(r"[\s　（）()、，。？,.:：\"'']", "", m["stem"])

        order, seen = [], set()
        for i in np.argsort(-combined):
            m = meta[int(i)]
            key = stem_key(m)
            if key in seen:
                continue
            seen.add(key)
            order.append(int(i))
            if len(order) >= TOP_K:
                break
        order = np.array(order)
        hits = [meta[i] for i in order]

        print()
        print("── 检索到的题 " + "─" * 48)
        for i, (m, idx) in enumerate(zip(hits, order), 1):
            print(f"  {i}. {m['stem'][:46]}")
            print(f"     相似度 {vs[idx]:.3f} | 答案：{str(m['answer_text'] or m['answer'])[:34]}")

        print()
        print("── 回答 " + "─" * 54)
        if WITH_GEN:
            try:
                print("  " + chat(SYSTEM, TEMPLATE.format(context=build_context(hits), query=q),
                                  max_tokens=300))
            except Exception as e:
                print(f"  调用大模型失败：{e}")
                print("  （上面的检索结果仍然可用）")
        else:
            print("  （--no-gen 模式，只显示检索结果）")
        print()


if __name__ == "__main__":
    main()
