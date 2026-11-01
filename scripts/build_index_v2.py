"""
建最终索引：题干 + 选项 + 解析

和 build_index.py 的区别只有一处——把解析也放进检索文本。

实测（exp_explanation.py，150 条用例）：
  题干+选项         hit@1 58.0%  hit@3 79.3%
  题干+选项+解析     hit@1 58.7%  hit@3 82.0%   ← 采用这个

原因：解析里常出现题干没写的术语（"一般规律""层次性"等），
能把正确题从第 4、5 名拉进前 3。

运行：python scripts/build_index_v2.py
"""

import json
import time
from pathlib import Path

import numpy as np

from embedder import embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SRC = DATA / "questions_v2.json"
OUT_VEC = DATA / "vectors.npy"
OUT_META = DATA / "index_meta.json"


def build_text(q: dict) -> str:
    parts = [q["stem"]]
    if q.get("options"):
        parts.extend(f"{k}. {v}" for k, v in q["options"].items())
    if q.get("explanation"):
        parts.append(q["explanation"])
    return " ".join(parts)


def main():
    t0 = time.time()
    qs = json.loads(SRC.read_text(encoding="utf-8"))["questions"]
    texts = [build_text(q) for q in qs]
    print(f"载入 {len(qs)} 题，平均文本长度 {sum(map(len, texts)) / len(texts):.0f} 字")
    print(f"其中带解析的 {sum(1 for q in qs if q.get('explanation'))} 题")

    print("正在编码...")
    vecs = embed(texts)
    np.save(OUT_VEC, vecs.numpy())
    print(f"编码完成，形状 {tuple(vecs.shape)}，耗时 {time.time() - t0:.1f} 秒")

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
            "explanation": q.get("explanation", ""),
        }
        for q in qs
    ]
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已保存:")
    print(f"  向量: {OUT_VEC}  ({OUT_VEC.stat().st_size / 1024:.0f} KB)")
    print(f"  元数据: {OUT_META}  ({OUT_META.stat().st_size / 1024:.0f} KB)")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
