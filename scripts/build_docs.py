"""
把题库拼成"讲次级长文档"

为什么要这一步：
  现在的题库是 496 条结构化记录，每条自带答案和解析——太干净了，
  检索做得好是理所当然的。真实场景里拿到的是连续文本，要先切片。

  所以这里把每讲的题 + 选项 + 答案 + 解析拼成一篇连续长文，
  让"切多大、切在哪"变成真问题。

注意局限（这条要写进实验记录，不能省）：
  这是半合成数据。真实文档里一块文本是连贯论述，这里的块是
  "两道题 + 两段解析"。所以它能练切片/向量库/重排三个技术点，
  但不能说成"做过真实文档 RAG"。

输出：
  data/docs/{讲次}.txt     每讲一篇
  data/docs/_index.json    每篇包含哪些题号（评测时用来判断"检索到没"）
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
META = ROOT / "data" / "index_meta.json"
OUT = ROOT / "data" / "docs"


def render(m: dict) -> str:
    """把一道题渲染成一段文字。

    刻意保留"题号 + 题干 + 选项 + 答案 + 解析"的完整结构，
    但用换行把它们连起来，读起来像连续文档而不是表格。
    """
    lines = [f"{m['number']}. {m['stem']}"]

    opts = m.get("options") or {}
    if opts:
        # 选项排成一行，像讲义里的格式
        lines.append("　".join(f"{k}. {v}" for k, v in opts.items()))

    ans = str(m.get("answer_text") or m.get("answer") or "").strip()
    if ans:
        lines.append(f"答案：{ans}")

    exp = (m.get("explanation") or "").strip()
    if exp:
        lines.append(exp)

    return "\n".join(lines)


def safe_name(lecture: str) -> str:
    """讲次名里带全角空格，文件名里不好用，统一替换掉。"""
    return re.sub(r"[\s　]+", "_", lecture.strip())


def main():
    meta = json.loads(META.read_text(encoding="utf-8"))

    # 按讲次分组，保持题库里的原始顺序（number 就是顺序）
    by_lecture: dict[str, list[dict]] = {}
    for m in meta:
        by_lecture.setdefault(m["lecture"], []).append(m)

    OUT.mkdir(parents=True, exist_ok=True)
    index = {}

    for lecture, items in by_lecture.items():
        items.sort(key=lambda x: (x.get("type", ""), x.get("number", 0)))
        name = safe_name(lecture)

        # 逐条拼，同时记下每道题在整篇里的字符区间 [start, end)
        # 为什么不用"第几题"来识别：判断题、单选、多选各自从 1 开始编号，
        # 数字会重复。字符区间是唯一可靠的对应关系，而且不用往正文里插标记
        # （插标记会让"切在题目中间"这种问题消失，切片实验就失真了）。
        parts, spans, cur = [], [], 0
        for m in items:
            seg = render(m)
            if parts:
                cur += 2                     # 段落之间的空行
            spans.append({"qid": m["id"], "start": cur, "end": cur + len(seg)})
            parts.append(seg)
            cur += len(seg)
        body = "\n\n".join(parts)

        p = OUT / f"{name}.txt"
        p.write_text(body, encoding="utf-8")

        index[name] = {
            "lecture": lecture,
            "file": p.name,
            "qids": [m["id"] for m in items],
            "spans": spans,
            "chars": len(body),
        }
        print(f"  {lecture:<16} {len(items):>3} 题  {len(body):>5} 字  → {p.name}")

    (OUT / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    total = sum(v["chars"] for v in index.values())
    print(f"\n共 {len(index)} 篇，{total} 字")


if __name__ == "__main__":
    main()
