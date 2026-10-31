"""
把题库 docx 解析成结构化 JSON

不依赖第三方库 —— docx 本质是个 zip 包，用标准库直接读里面的 document.xml。

输出字段说明：
    id          唯一标识，如 "single-1-1"
    type        single(单选) / multi(多选) / judge(判断)
    section     所属题型
    lecture     所属讲次
    number      题号（每讲内从 1 开始）
    stem        题干
    options     选项字典 {"A": "...", "B": "..."}，判断题为空
    answer      原始答案（字母或"正确"/"错误"）
    answer_text 把字母还原成选项内容后的答案 —— 这一步很重要

运行：python scripts/parse_questions.py
"""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

SRC = Path("C:/Users/86155/Desktop/自然辩证法题库（含答案）.docx")
OUT = Path(__file__).parent.parent / "data" / "questions.json"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# 匹配用的正则（注意题库里用的是全角句点 ．）
RE_QUESTION = re.compile(r"^(\d+)\s*[．.]\s*(.*)$")
RE_OPTION = re.compile(r"^([A-E])\s*[．.]\s*(.*)$")
RE_ANSWER = re.compile(r"^答案\s*[:：]\s*(.*)$")
RE_LECTURE = re.compile(r"^第[一二三四五六七八九十\d]+讲")
RE_SECTION = re.compile(r"^[一二三四五六七八九十]、\s*(单项选择题|多项选择题|判断题)")

TYPE_MAP = {
    "单项选择题": "single",
    "多项选择题": "multi",
    "判断题": "judge",
}


def read_paragraphs(path: Path) -> list[str]:
    """从 docx 里读出所有段落文本"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(f"{{{W}}}p"):
        text = "".join(t.text or "" for t in p.iter(f"{{{W}}}t"))
        paras.append(text.strip())
    return paras


def build_answer_text(qtype: str, answer: str, options: dict) -> str:
    """把答案字母还原成选项内容"""
    if qtype == "judge":
        return answer
    letters = [c for c in answer if c.isalpha()]
    texts = [options.get(c, f"<缺失选项{c}>") for c in letters]
    return " / ".join(texts) if texts else ""


def parse(paras: list[str]) -> list[dict]:
    questions = []
    section = lecture = None
    cur = None
    started = False

    def flush():
        nonlocal cur
        if cur and cur["stem"]:
            cur["answer_text"] = build_answer_text(
                cur["type"], cur["answer"], cur["options"]
            )
            cur["stem"] = re.sub(r"\s+", " ", cur["stem"]).strip()
            questions.append(cur)
        cur = None

    for raw in paras:
        line = raw.strip()
        if not line:
            continue

        # 题型标题：带"（N 题）"的是目录行，跳过；不带的才是正文标题
        m_section = RE_SECTION.match(line)
        if m_section:
            if "（" in line or "(" in line:
                continue
            started = True
            flush()
            section = m_section.group(1)
            continue

        # 正文从第一个题型标题或"第X讲"开始，之前的内容（封面、目录）全部跳过
        if not started:
            continue

        if RE_LECTURE.match(line):
            flush()
            lecture = line
            continue

        m_answer = RE_ANSWER.match(line)
        if m_answer and cur is not None:
            cur["answer"] = m_answer.group(1).strip()
            flush()
            continue

        m_question = RE_QUESTION.match(line)
        if m_question:
            # 题干里偶尔会以数字开头（比如"1949年..."），靠"当前是否在等答案"来区分
            if cur is not None and not cur["answer"] and not cur["options"]:
                # 上一题还没答案就来了新题号 —— 把新行当题干续接
                cur["stem"] += " " + line
                continue
            flush()
            num, stem = int(m_question.group(1)), m_question.group(2)
            cur = {
                "id": f"{TYPE_MAP.get(section, 'unknown')}-{num}",
                "type": TYPE_MAP.get(section, "unknown"),
                "section": section,
                "lecture": lecture,
                "number": num,
                "stem": stem,
                "options": {},
                "answer": "",
                "answer_text": "",
            }
            continue

        m_option = RE_OPTION.match(line)
        if m_option and cur is not None:
            cur["options"][m_option.group(1)] = m_option.group(2).strip()
            continue

        # 其他情况：当作题干的续行
        if cur is not None:
            cur["stem"] += " " + line

    flush()
    return questions


def main():
    paras = read_paragraphs(SRC)
    questions = parse(paras)

    # 修正 id，避免不同讲次里题号重复
    seen = Counter()
    for q in questions:
        key = f"{q['type']}-{q['lecture']}-{q['number']}"
        seen[key] += 1
        q["id"] = key if seen[key] == 1 else f"{key}-{seen[key]}"
        q["lecture_num"] = q["lecture"] or ""

    result = {
        "source": SRC.name,
        "total": len(questions),
        "questions": questions,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 统计报告 ----
    print("=" * 56)
    print("解析结果")
    print("=" * 56)
    print(f"题目总数: {len(questions)}")
    print()
    by_type = Counter(q["type"] for q in questions)
    name = {"single": "单项选择题", "multi": "多项选择题", "judge": "判断题"}
    for t, c in by_type.most_common():
        print(f"  {name.get(t, t):<8} {c:>4} 题")

    print()
    print("数据质量检查：")
    no_answer = [q for q in questions if not q["answer"]]
    no_opts = [q for q in questions if q["type"] != "judge" and not q["options"]]
    bad_text = [q for q in questions if not q["answer_text"]]
    print(f"  缺答案      : {len(no_answer)}")
    print(f"  缺选项(非判断): {len(no_opts)}")
    print(f"  答案未还原   : {len(bad_text)}")

    total_chars = sum(len(q["stem"]) + sum(len(v) for v in q["options"].values())
                      for q in questions)
    print(f"  正文字符数   : {total_chars:,}")

    print()
    print("各讲分布：")
    by_lec = Counter(q["lecture"] for q in questions)
    for lec, c in by_lec.items():
        print(f"  {lec}: {c} 题")

    print()
    print("=== 样例（单选 / 多选 / 判断 各一）===")
    for t in ["single", "multi", "judge"]:
        q = next((x for x in questions if x["type"] == t), None)
        if q:
            print(f"\n[{q['id']}] {q['lecture']}")
            print(f"  题干: {q['stem'][:60]}")
            if q["options"]:
                for k, v in q["options"].items():
                    print(f"    {k}. {v}")
            print(f"  答案: {q['answer']}  →  {q['answer_text']}")

    print()
    print(f"已写入: {OUT}")


if __name__ == "__main__":
    main()
