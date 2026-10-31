"""
生成"评测工作台" Excel —— 交给人工填写的那份表

两个 Sheet：
  1_改写任务      50 道题，填"学生会怎么问"（核心任务）
  2_等价组核对    每道题给出最相似的 3 个候选，判断是否同一知识点（修正打分口径）

抽样策略：
  - 去掉重复题干（题库里有几组完全重复的）
  - 按讲次分层，各讲按比例分配名额
  - 固定随机种子，可复现

运行：python scripts/make_worktable.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from embedder import QUERY_INSTRUCTION, embed

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT = ROOT / "评测工作台.xlsx"

TARGET_N = 50
SEED = 2026
TOP_CAND = 3

HEAD_FILL = PatternFill("solid", fgColor="E8E8E4")
TODO_FILL = PatternFill("solid", fgColor="FFF6D9")
HEAD_FONT = Font(bold=True, size=11)
THIN = Side(style="thin", color="C8C8C8")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def norm(s: str) -> str:
    return re.sub(r"[\s（()）　、，。？,.:：\"'']", "", s)


def stratified_sample(questions, n):
    """按讲次分层抽样，名额按各讲题目数比例分配（最大余数法）"""
    by_lec = {}
    for q in questions:
        by_lec.setdefault(q["lecture"], []).append(q)

    total = len(questions)
    raw = {lec: len(qs) / total * n for lec, qs in by_lec.items()}
    alloc = {lec: int(v) for lec, v in raw.items()}
    rest = n - sum(alloc.values())
    for lec, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rest]:
        alloc[lec] += 1

    rng = random.Random(SEED)
    picked = []
    for lec in sorted(by_lec):
        pool = by_lec[lec]
        k = min(alloc[lec], len(pool))
        picked.extend(rng.sample(pool, k))
    return picked


def main():
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
    vecs = np.load(DATA / "vectors.npy")
    id2idx = {m["id"]: i for i, m in enumerate(meta)}

    # 去掉重复题干
    seen, uniq = set(), []
    for q in meta:
        k = norm(q["stem"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(q)

    sample = stratified_sample(uniq, TARGET_N)
    print(f"去重后 {len(uniq)} 道，抽出 {len(sample)} 道")
    print("题型分布:", dict(Counter(q["type"] for q in sample)))
    print("讲次分布:", {k: v for k, v in sorted(Counter(q['lecture'] for q in sample).items())})

    # 预计算等价候选
    idxs = [id2idx[q["id"]] for q in sample]
    qv = embed([QUERY_INSTRUCTION + q["stem"] for q in sample])
    sims = (qv @ vecs.T).numpy()

    wb = Workbook()

    # ---------------- Sheet 0：说明 ----------------
    ws = wb.active
    ws.title = "0_先看这里"
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 100
    lines = [
        ("h", "这份表是干什么的"),
        ("t", "把题库里的书面题干，改写成学生真实会打出来的问法。用它来测检索系统在「真实提问」下的命中率。"),
        ("t", "原题测原题没有意义——那测的是「背题」，不是「提问」。"),
        ("", ""),
        ("h", "Sheet「1_改写任务」怎么填"),
        ("t", "只填 F 列「学生会怎么问」。每行至少 1 条，想多写几条用 | 分隔（半角竖线）。"),
        ("t", "改写的唯一标准：像不像你在微信里问同学的样子。不用完整、不用客气、可以有口语词。"),
        ("", ""),
        ("h", "好的改写 vs 不好的改写（以「古代中国的四大发明是（ ）」为例）"),
        ("ok", "好：四大发明是哪四个来着"),
        ("ok", "好：中国的四大发明都有啥"),
        ("ok", "好：老师说的那个四大发明具体指什么"),
        ("bad", "不好：古代中国四大发明是      ← 只是删字，还是书面语"),
        ("bad", "不好：请问古代中国的四大发明包括哪些？   ← 太正式，学生不会这么问"),
        ("", ""),
        ("h", "Sheet「2_等价组核对」怎么填"),
        ("t", "每题给出 3 个最相似的候选题。判断它们和左边目标题是不是「同一知识点」。"),
        ("t", "判断标准只有一句：学生问这个知识点，答这两道题里的任何一道，学生都算答对了。"),
        ("t", "选「是」或「否」。这道不是必需，但强烈建议做——它决定打分准不准。"),
        ("", ""),
        ("h", "填完怎么办"),
        ("t", "保存这个文件，告诉我一声。我会读取它、跑评测、出一张基线表。"),
        ("t", "不用一次填完，填多少算多少，先填 20 条也能跑。"),
    ]
    r = 1
    for kind, text in lines:
        c = ws.cell(row=r, column=2, value=text)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        if kind == "h":
            c.font = Font(bold=True, size=12)
            ws.row_dimensions[r].height = 24
        elif kind == "ok":
            c.font = Font(color="1D6E56")
        elif kind == "bad":
            c.font = Font(color="A32D2D")
            c.alignment = Alignment(wrap_text=True, vertical="center", indent=3)
        else:
            c.font = Font(size=11)
            ws.row_dimensions[r].height = 30 if text else 8
        r += 1

    # ---------------- Sheet 1：改写任务 ----------------
    ws1 = wb.create_sheet("1_改写任务")
    heads = ["题号", "题型", "讲次", "题干（原题）", "标准答案", "学生会怎么问（在这里填）"]
    widths = [22, 8, 16, 62, 18, 46]
    for i, (h, w) in enumerate(zip(heads, widths), start=1):
        cell = ws1.cell(row=1, column=i, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
        ws1.column_dimensions[get_column_letter(i)].width = w
    ws1.row_dimensions[1].height = 26

    type_cn = {"single": "单选", "multi": "多选", "judge": "判断"}
    for r, q in enumerate(sample, start=2):
        vals = [q["id"], type_cn.get(q["type"], q["type"]), q["lecture"],
                q["stem"], q["answer_text"] or q["answer"], ""]
        for c, v in enumerate(vals, start=1):
            cell = ws1.cell(row=r, column=c, value=v)
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="center",
                                       horizontal="left" if c in (1, 4) else "center")
            if c == 6:
                cell.fill = TODO_FILL
        ws1.row_dimensions[r].height = 34

    ws1.freeze_panes = "A2"

    # ---------------- Sheet 2：等价组核对 ----------------
    ws2 = wb.create_sheet("2_等价组核对")
    heads2 = ["目标题号", "目标题干", "候选题号", "候选题干", "相似度", "是否同一知识点（填 是/否）"]
    widths2 = [20, 44, 22, 56, 9, 22]
    for i, (h, w) in enumerate(zip(heads2, widths2), start=1):
        cell = ws2.cell(row=1, column=i, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
        ws2.column_dimensions[get_column_letter(i)].width = w
    ws2.row_dimensions[1].height = 26

    r = 2
    for i, q in enumerate(sample):
        order = np.argsort(-sims[i])
        cand = [int(j) for j in order if int(j) != id2idx[q["id"]]][:TOP_CAND]
        for j in cand:
            vals = [q["id"], q["stem"][:60], meta[j]["id"], meta[j]["stem"][:80],
                    round(float(sims[i][j]), 3), ""]
            for c, v in enumerate(vals, start=1):
                cell = ws2.cell(row=r, column=c, value=v)
                cell.border = BORDER
                cell.alignment = Alignment(wrap_text=True, vertical="center",
                                           horizontal="center" if c in (5, 6) else "left")
                if c == 6:
                    cell.fill = TODO_FILL
            ws2.row_dimensions[r].height = 30
            r += 1

    dv = DataValidation(type="list", formula1='"是,否"', allow_blank=True)
    ws2.add_data_validation(dv)
    dv.add(f"F2:F{r - 1}")
    ws2.freeze_panes = "A2"

    wb.save(OUT)
    print(f"\n已生成: {OUT}")
    print(f"  1_改写任务: {len(sample)} 行")
    print(f"  2_等价组核对: {r - 2} 行")


if __name__ == "__main__":
    main()
