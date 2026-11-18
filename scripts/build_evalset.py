"""
从教材抽小节标题，构造"标题查询 → 正文位置"的评测集

为什么要这么干：
  换成教材之后没有标注，得自己造 ground truth。
  让模型出题再让模型答会有循环风险（题目可能照着原文措辞写，检索显得虚高）。
  用教材原文的**小节标题**当查询就没有这个问题——标题是书里本来就有的，
  而"这一节的内容在哪"由标题位置客观决定。

要过滤两类噪声：
  1. 目录页的标题（带 ··· 引导线），不是正文
  2. 正文里重复出现的同名标题（目录已经排掉，正文只留第一次之后出现的）

输出 data/book/evalset.json
  [{"qid","query","heading","start","end","chars"}]
查询 = 去掉序号的标题（"一、朴素唯物主义自然观" → "朴素唯物主义自然观"）
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
OUT = ROOT / "data" / "book" / "evalset.json"

HEAD_PAT = re.compile(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+\.\d+\s)")
TOC_MARK = re.compile(r"[·⋯…]{2,}|\.{4,}")

MIN_CHARS, MAX_CHARS = 250, 2500     # 太短没有检索价值，太长一节跨好几个主题


def strip_number(s: str) -> str:
    s = re.sub(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+\.\d+)\s*[:：]?\s*", "", s)
    return s.strip(" :：")


def main():
    text = BOOK.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 逐行算出偏移，才能把"行号"换成"字符位置"
    offsets, cur = [], 0
    for l in lines:
        offsets.append(cur)
        cur += len(l) + 1

    raw = []
    for i, l in enumerate(lines):
        s = l.strip()
        if not (4 <= len(s) <= 45):
            continue
        if TOC_MARK.search(s):          # 目录行
            continue
        if HEAD_PAT.match(s):
            raw.append((i, offsets[i], s))

    # 同一标题在目录和正文各出现一次 → 按标题去重，保留最后一次（正文）
    seen, uniq = set(), []
    for i, off, s in reversed(raw):
        key = strip_number(s)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((i, off, s))
    uniq.reverse()

    items = []
    for k, (i, off, s) in enumerate(uniq):
        end = uniq[k + 1][1] if k + 1 < len(uniq) else len(text)
        # 标题本身也算进这一节
        chars = end - off
        if not (MIN_CHARS <= chars <= MAX_CHARS):
            continue
        items.append({
            "qid": f"sec-{len(items)+1:03d}",
            "query": strip_number(s),
            "heading": s,
            "start": off,
            "end": end,
            "chars": chars,
        })

    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"标题 {len(raw)} 条 → 去重后 {len(uniq)} 条 → 长度合适 {len(items)} 条")
    print(f"正文总量 {len(text)} 字，评测集覆盖 "
          f"{sum(x['chars'] for x in items)} 字（{sum(x['chars'] for x in items)/len(text):.0%}）")
    print()
    for x in items[:8]:
        print(f"  {x['chars']:>4} 字  {x['query'][:34]}")
    print("  ...")
    lens = sorted(x["chars"] for x in items)
    print(f"  长度中位数 {lens[len(lens)//2]} 字")


if __name__ == "__main__":
    main()
