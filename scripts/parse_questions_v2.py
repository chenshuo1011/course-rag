"""
解析「含答案解析」版题库 —— 相比旧版多了 explanation 字段

旧版只有"答案是什么"，新版每题还带一段解析（为什么选这个 / 错在哪）。
这对"生成"这一步是关键：模型能拿到解释，而不是只有两个字"错误"。

解析后会和旧版按题干对齐，沿用旧 id，
这样评测工作台里的 150 条用例不用改。

运行：python scripts/parse_questions_v2.py
"""

import json
import re
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree as ET

SRC = Path("C:/Users/86155/Desktop/自然辩证法题库（含答案解析）.docx")
ROOT = Path(__file__).parent.parent
OLD = ROOT / "data" / "questions.json"
OUT = ROOT / "data" / "questions_v2.json"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

RE_QUESTION = re.compile(r"^(\d+)\s*[．.]\s*(.*)$")
RE_OPTION = re.compile(r"^([A-E])\s*[．.]\s*(.*)$")
RE_ANSWER = re.compile(r"^答案\s*[:：]\s*(.*)$")
RE_EXPLAIN = re.compile(r"^解析\s*[:：]\s*(.*)$")
RE_LECTURE = re.compile(r"^第[一二三四五六七八九十\d]+讲")
RE_SECTION = re.compile(r"^[一二三四五六七八九十]、\s*(单项选择题|多项选择题|判断题)")

TYPE_MAP = {"单项选择题": "single", "多项选择题": "multi", "判断题": "judge"}


def read_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return ["".join(t.text or "" for t in p.iter(f"{{{W}}}t")).strip()
            for p in root.iter(f"{{{W}}}p")]


def build_answer_text(qtype: str, answer: str, options: dict) -> str:
    if qtype == "judge":
        return answer
    letters = [c for c in answer if c.isalpha()]
    return " / ".join(options.get(c, f"<缺{c}>") for c in letters)


def parse(paras: list[str]) -> list[dict]:
    questions, section, lecture, cur = [], None, None, None
    started = False
    last = None          # 最近写入的字段名，用于处理换行续写

    def flush():
        nonlocal cur, last
        if cur and cur["stem"]:
            cur["answer_text"] = build_answer_text(cur["type"], cur["answer"], cur["options"])
            cur["stem"] = re.sub(r"\s+", " ", cur["stem"]).strip()
            cur["explanation"] = re.sub(r"\s+", " ", cur["explanation"]).strip()
            questions.append(cur)
        cur, last = None, None

    for raw in paras:
        line = raw.strip()
        if not line:
            continue

        m_sec = RE_SECTION.match(line)
        if m_sec:
            if "（" in line or "(" in line:      # 目录行
                continue
            started = True
            flush()
            section = m_sec.group(1)
            continue

        if not started:
            continue

        if RE_LECTURE.match(line):
            flush()
            lecture = line
            continue

        m_ans = RE_ANSWER.match(line)
        if m_ans and cur is not None:
            cur["answer"] = m_ans.group(1).strip()
            last = "answer"
            continue

        m_exp = RE_EXPLAIN.match(line)
        if m_exp and cur is not None:
            cur["explanation"] = m_exp.group(1).strip()
            last = "explanation"
            flush()                              # 解析是本题最后一段，直接收尾
            continue

        m_q = RE_QUESTION.match(line)
        if m_q and (cur is None or cur["answer"]):
            flush()
            num, stem = int(m_q.group(1)), m_q.group(2)
            cur = {
                "type": TYPE_MAP.get(section, "unknown"),
                "section": section,
                "lecture": lecture,
                "number": num,
                "stem": stem,
                "options": {},
                "answer": "",
                "answer_text": "",
                "explanation": "",
            }
            last = "stem"
            continue

        m_opt = RE_OPTION.match(line)
        if m_opt and cur is not None:
            cur["options"][m_opt.group(1)] = m_opt.group(2).strip()
            last = "stem"
            continue

        # 续行：接到最近写入的字段上
        if cur is not None:
            if last == "explanation":
                cur["explanation"] += " " + line
            elif last in ("stem", None):
                cur["stem"] += " " + line

    flush()
    return questions


def main():
    paras = read_paragraphs(SRC)
    new = parse(paras)
    print(f"新版解析出 {len(new)} 题")

    by_type = Counter(q["type"] for q in new)
    print("题型:", dict(by_type))
    no_exp = [q for q in new if not q["explanation"]]
    print(f"缺解析: {len(no_exp)} 题")
    print(f"解析总字数: {sum(len(q['explanation']) for q in new):,}")

    # ---- 与旧版按题干对齐，沿用旧 id ----
    old = json.loads(OLD.read_text(encoding="utf-8"))["questions"]

    def norm(s):
        return re.sub(r"[\s　（）()、，。？,.．:：\"'']", "", s)

    pool = {}
    for o in old:
        pool.setdefault(norm(o["stem"]), []).append(o)

    matched, unmatched = 0, []
    for q in new:
        key = norm(q["stem"])
        cand = pool.get(key)
        if cand:
            q["id"] = cand.pop(0)["id"]
            matched += 1
        else:
            unmatched.append(q)

    print(f"\n题干精确匹配旧版: {matched} / {len(new)}")

    # 未匹配的用相似度兜底
    if unmatched:
        used = {o["id"] for o in old} - {q.get("id") for q in new}
        remaining = [o for o in old if o["id"] in used]
        for q in unmatched:
            best, score = None, 0.0
            for o in remaining:
                r = SequenceMatcher(None, norm(q["stem"]), norm(o["stem"])).ratio()
                if r > score:
                    best, score = o, r
            if best and score > 0.8:
                q["id"] = best["id"]
                q["match_score"] = round(score, 3)
                remaining.remove(best)
                matched += 1
                print(f"  模糊匹配 {score:.2f}: {q['stem'][:30]}")

    for q in new:
        q.setdefault("id", f"{q['type']}-{q['lecture']}-{q['number']}-x")

    print(f"最终带 id 的题目: {sum(1 for q in new if 'x' not in q['id'])} / {len(new)}")

    OUT.write_text(json.dumps({
        "source": SRC.name,
        "total": len(new),
        "questions": new,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已写入: {OUT}")
    print(f"文件大小: {OUT.stat().st_size / 1024:.0f} KB")

    print("\n=== 样例（三种题型各一）===")
    for t in ("single", "multi", "judge"):
        q = next(x for x in new if x["type"] == t)
        print(f"\n[{q['id']}]")
        print(f"  题干: {q['stem'][:56]}")
        print(f"  答案: {q['answer_text'][:50]}")
        print(f"  解析: {q['explanation'][:70]}")


if __name__ == "__main__":
    main()
