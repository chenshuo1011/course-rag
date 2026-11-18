"""
在教材里定位每道题的"答案出处"

为什么需要这一步：
  题库时代，每条用例都有 target qid，检索到没一目了然。
  换成教材之后没有标注——必须自己造 ground truth，否则 hit@k 无从谈起。

做法：
  题库的"解析"通常就是教材里的原句改写，所以拿解析去书里找。
  OCR 有错字，精确整句匹配会失败，所以取若干 12 字片段分别找，
  多用几个片段投票：多数片段都落在同一区域，就认为定位成功。

为什么片段取 12 字：
  太短（4~6 字）到处都能匹配，噪声极大；
  太长（30+ 字）OCR 错一个字就全废。
  12 字是可用的折中。

输出：data/book/locate.json
  {qid: {"pos": 字符位置, "hits": 找到的片段数, "conf": 命中率}}
只有 conf 达标的题才进入评测。
"""

import json
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"
META = ROOT / "data" / "index_meta.json"
OUT = ROOT / "data" / "book" / "locate.json"

FRAG = 12          # 片段长度
N_FRAG = 8         # 每题取几个片段
MIN_HITS = 3       # 至少几个片段找到才算定位成功


def fragments(text: str, n: int = N_FRAG, size: int = FRAG) -> list[str]:
    """从文本里等距取 n 个定长片段。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < size:
        return []
    span = len(t) - size
    if span == 0:
        return [t]
    step = span // max(1, n - 1) if n > 1 else 0
    return [t[min(i * step, span): min(i * step, span) + size] for i in range(n)]


def main():
    book = re.sub(r"\n", "", BOOK.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    print(f"教材 {len(book)} 字，题目 {len(meta)} 道")

    located, failed = {}, []
    for m in meta:
        src = (m.get("explanation") or "").strip() or m.get("stem", "")
        frags = fragments(src)
        pos = []
        for f in frags:
            i = book.find(f)
            if i >= 0:
                pos.append(i)
        if len(pos) < MIN_HITS:
            failed.append(m["id"])
            continue
        located[m["id"]] = {
            "pos": int(median(pos)),
            "hits": len(pos),
            "of": len(frags),
            "conf": round(len(pos) / max(1, len(frags)), 2),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(located, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"定位成功 {len(located)} 道 / {len(meta)} 道"
          f"（{len(located)/len(meta):.0%}），失败 {len(failed)} 道")
    confs = [v["conf"] for v in located.values()]
    if confs:
        print(f"命中率分布：≥0.75 的 {sum(1 for c in confs if c>=0.75)} 道，"
              f"0.5~0.75 的 {sum(1 for c in confs if 0.5<=c<0.75)} 道")


if __name__ == "__main__":
    main()
