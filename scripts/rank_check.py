"""
逐条看：正确答案排在第几位？

为什么要做这个：
  「命中@1 38%，命中@5 75%」只告诉你比例，
  但你可能想知道——那 37 个点具体是哪些题？正确答案是排第 2 还是第 5？

  排第 2 和第 5 的意义完全不同：
    排第 2 → 重排只要挪一位就能救
    排第 5 → 重排要挪四位，难度大

要填的是两个空（标了 TODO 的地方），填完就能跑。

用法：
    cd /d D:\course-rag\scripts
    "D:\anaconda3\envs\yolo\python.exe" rank_check.py
"""

import json
from pathlib import Path

import chunk as chunkmod
import embedder
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
EVAL = ROOT / "data" / "book" / "evalset.json"
TOPK = 5          # 只看前 5 名


def rank_of_correct(block_texts, target):
    """正确答案排第几（从 1 数）。没找到返回 None。

    block_texts 是前 TOPK 块的正文，按相似度从高到低排。
    target 是这一节的首句（60 字）。
    """
    tgt = "".join(target.split())      # 去掉所有空白（换行、全角空格）
    for i, blk in enumerate(block_texts, start=1):
        # 两边都去空白再比：OCR 会在中英文之间插空格、换行位置也和原文不同，
        # 不去掉的话明明同一句话也对不上。
        if tgt and tgt in "".join(blk.split()):
            return i
    return None


def main():
    text = BOOK.read_text(encoding="utf-8")
    evals = json.loads(EVAL.read_text(encoding="utf-8"))

    # 用已经验证过的最优参数切块
    spans = chunkmod.chunk_text(text, 500, 100)
    texts = [text[s:e] for s, e in spans]
    idx = build(embedder.embed(texts).numpy(), "numpy")

    rows = []
    for e in evals:
        head_len = len(e["heading"])
        # 真值：这一节标题之后的首句（和 eval_book.py 里一致）
        target = "".join(text[e["start"] + head_len: e["start"] + head_len + 60].split())

        top = idx.search(embedder.embed_query(e["query"]).numpy(), TOPK)[0]
        rank = rank_of_correct([texts[i] for i in top], target)
        rows.append((rank, e["query"]))

    # ── 结果 ──
    # 注意不能写 if r：rows 里是 (rank, query) 元组，元组永远非空，
    # 所以 rank 为 None 的也会被算成"找到了"。必须显式判 r[0] is not None。
    found = [r for r in rows if r[0] is not None]
    print(f"共 {len(rows)} 条，前 {TOPK} 名里找到正确答案的 {len(found)} 条")
    print(f"排第 1 的 {sum(1 for r in found if r[0] == 1)} 条")
    print()

    print("排第 2~5 的（重排能救回来的）：")
    for rank, q in sorted([r for r in found if r[0] > 1]):
        print(f"  第{rank}位  {q[:36]}")

    # 命中@1 = 排第一的条数 / 总条数
    hit1 = sum(1 for r in found if r[0] == 1) / len(rows)
    print()
    print(f"命中@1 = {hit1:.0%}")


if __name__ == "__main__":
    main()
