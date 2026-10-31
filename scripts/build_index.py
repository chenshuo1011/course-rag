"""
建向量索引：把所有题目编码成向量，存成文件

做一次就够了，之后查询直接读文件，不用重新编码。

运行：python scripts/build_index.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from embedder import embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
QUESTIONS = DATA / "questions.json"
OUT_VEC = DATA / "vectors.npy"
OUT_META = DATA / "index_meta.json"


def build_search_text(q: dict) -> str:
    """
    构造用于检索的文本。

    注意：这里把题干和所有选项都放进去，但不放答案。
    原因是检索时要模拟"学生看到题目"的场景，
    而且答案文本（比如"自然辩证法"）本身可能就是学生提问的关键词，
    放进去会人为降低难度。做对比实验时可以试两种写法。
    """
    parts = [q["stem"]]
    if q["options"]:
        parts.extend(f"{k}. {v}" for k, v in q["options"].items())
    return " ".join(parts)


def main():
    t0 = time.time()

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = data["questions"]
    print(f"载入 {len(questions)} 道题")

    texts = [build_search_text(q) for q in questions]
    print(f"平均文本长度: {sum(len(t) for t in texts) / len(texts):.0f} 字")

    print("正在编码（首次会加载模型）...")
    vecs = embed(texts)
    print(f"编码完成，向量形状: {tuple(vecs.shape)}，耗时 {time.time() - t0:.1f} 秒")

    np.save(OUT_VEC, vecs.numpy())

    meta = [
        {
            "id": q["id"],
            "type": q["type"],
            "lecture": q["lecture"],
            "number": q["number"],
            "stem": q["stem"],
            "options": q["options"],
            "answer": q["answer"],
            "answer_text": q["answer_text"],
        }
        for q in questions
    ]
    OUT_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n已保存:")
    print(f"  向量: {OUT_VEC}  ({OUT_VEC.stat().st_size / 1024:.0f} KB)")
    print(f"  元数据: {OUT_META}")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
