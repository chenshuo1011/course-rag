"""
清洗 OCR 原文：去页眉页脚、把断行并成句子

难点不是"去掉第 12 页这几个数字"，是**页眉会变**：
这本书每章页眉是"第X章 XXX"，翻一章就换一个。硬编码字符串，换本书就废。

所以用统计的办法：**同一个短句在大量页面上重复出现，它就不是正文。**
正文不会每页都出现一模一样的一句。

清洗完输出 data/book/book.txt，供切片使用。
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "book" / "raw_pages.jsonl"
OUT = ROOT / "data" / "book" / "book.txt"

# 句子/段落的结束符。行尾是这些字符时，认为这句话说完了
SENT_END = "。！？；：」』”）)!?:;"
# 明显的标题行：出现在行首就单独成段
HEADING = re.compile(r"^\s*(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+\.\d*)")

# 脚注：OCR 会把脚注并进正文句子中间，例如
#   「都包含经验技①习近平：《…》，人民出版社2022年版，第34页。能，都使用工具机器」
# 这类内容是纯噪声（出版社、页码、书名），检索时只会污染向量。
FOOTNOTE = re.compile(
    r"[①-⑩][^①-⑩\n]{0,150}?(?:第\d+页|年版|出版社)[^①-⑩\n]{0,25}?[。．]")

# 页眉阈值：在超过这么多页上原样出现的短行，判为页眉/页脚
HEADER_MIN_PAGES = 8
HEADER_MAX_LEN = 30


def norm(line: str) -> str:
    """比较用：去掉所有空白。OCR 有时会在中英文之间插空格。"""
    return re.sub(r"\s+", "", line)


def main():
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["page"])
    print(f"原始 {len(rows)} 页")

    # ── 第一遍：找出"跨页重复的短行" ──
    page_sets = [set(norm(x) for x in r["text"].splitlines() if norm(x)) for r in rows]
    counter = Counter()
    for s in page_sets:
        for line in s:
            if 0 < len(line) <= HEADER_MAX_LEN:
                counter[line] += 1

    headers = {line for line, n in counter.items() if n >= HEADER_MIN_PAGES}
    # 纯数字（页码）单独处理，长度不限
    print(f"识别出疑似页眉/页脚 {len(headers)} 条，例如：")
    for h in sorted(headers)[:6]:
        print(f"   ×{counter[h]:>3}  {h}")

    # ── 第二遍：清洗 + 并句 ──
    out_lines, dropped = [], 0
    for r in rows:
        buf = ""
        for raw_line in r["text"].splitlines():
            line = raw_line.strip()
            if not line:
                continue
            n = norm(line)

            if n in headers:                       # 页眉
                dropped += 1
                continue
            if re.fullmatch(r"[\dIVXivx—\-·\s]{1,6}", n):   # 纯页码
                dropped += 1
                continue
            if len(n) <= 2 and not re.search(r"[\u4e00-\u9fff]", n):  # 零星符号
                dropped += 1
                continue

            # 标题行单独成段
            if HEADING.match(line):
                if buf:
                    out_lines.append(buf)
                    buf = ""
                out_lines.append(line)
                continue

            buf += line
            if line and line[-1] in SENT_END:      # 句子结束了，落一行
                out_lines.append(buf)
                buf = ""
        if buf:
            out_lines.append(buf)

    text = "\n".join(out_lines)

    # 清脚注。放在并句之后做：并句会把脚注和前后文粘在一起，
    # 这时候「①…第34页。」才是一整段好匹配的文本。
    before = len(text)
    text, n_foot = FOOTNOTE.subn("", text)
    print(f"清除脚注 {n_foot} 处，减少 {before - len(text)} 字")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")

    print()
    print(f"丢弃行 {dropped} 条")
    print(f"输出 {len(out_lines)} 段，{len(text)} 字 → {OUT.name}")
    print(f"平均每段 {len(text)/max(1,len(out_lines)):.0f} 字")
    print()
    print("--- 开头 300 字 ---")
    print(text[:300])


if __name__ == "__main__":
    main()
