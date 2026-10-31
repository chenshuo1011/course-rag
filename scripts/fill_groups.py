"""
填写"2_等价组核对"：判定候选題与目标题是否同一知识点

判定规则（只此一条，机械执行，不因看起来顺眼就放宽）：
    候选题的【答案】是否直接回应了目标题所问的那一点？
    - 判断题的答案是"正误 + 该陈述本身"。若该陈述包含目标题所问的内容 → 是
    - 选择题的答案是某个选项内容。若该内容就是目标题要的答案 → 是
    - 只是同一章节、同一话题、同一个词出现 → 否

例：
  「（ ）的出现标志着我国农业实用科学体系的形成」
    vs 判断「《农政全书》的出现标志着我国农业实用科学体系的形成」→ 是（直接给了答案）
  「古代中国的四大发明是（ ）」
    vs 判断「唐宋时期成就主要体现在以四大发明为代表的技术领域」→ 否（没列出四个是什么）

运行：python scripts/fill_groups.py
"""

from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "评测工作台.xlsx"

SAME = {
    ("judge-第一讲-1", "judge-第一讲-2"),
    ("judge-第一讲-1", "judge-第一讲-3"),
    ("judge-第一讲-10", "judge-第一讲-9"),
    ("judge-第一讲-10", "judge-第一讲-13"),
    ("judge-第一讲-10", "judge-第一讲-12"),
    ("single-第一讲-7", "judge-第六讲-26"),
    ("single-第一讲-7", "judge-第六讲-27"),
    ("single-第七讲　工业革命-15", "multi-第七讲　工业革命-15"),
    ("judge-第七讲　工业革命-5", "single-第六讲-13"),
    ("judge-第七讲　工业革命-5", "multi-第六讲-22"),
    ("judge-第七讲　工业革命-2", "multi-第七讲　工业革命-12"),
    ("judge-第七讲　工业革命-2", "judge-第七讲　工业革命-1"),
    ("multi-第七讲　工业革命-10", "single-第七讲　工业革命-5"),
    ("multi-第七讲　工业革命-10", "judge-第七讲　工业革命-10"),
    ("judge-第三讲　古代中国-3", "single-第三讲　古代中国-4"),
    ("judge-第三讲　古代中国-4", "judge-第三讲　古代中国-5"),
    ("judge-第九讲-5", "single-第九讲-1"),
    ("single-第九讲-6", "single-第五讲-21"),
    ("single-第二讲-1", "judge-第二讲-1"),
    ("single-第二讲-1", "single-第二讲-2"),
    ("judge-第二讲-16", "judge-第二讲-14"),
    ("judge-第二讲-16", "judge-第二讲-15"),
    ("judge-第二讲-7", "judge-第二讲-6"),
    ("judge-第二讲-7", "single-第一讲-3"),
    ("judge-第二讲-7", "judge-第二讲-8"),
    ("judge-第二讲-18", "judge-第二讲-19"),
    ("multi-第五讲-4", "single-第五讲-6"),
    ("single-第五讲-17", "judge-第九讲-6"),
    ("judge-第五讲-13", "judge-第五讲-3"),
    ("judge-第八讲-5", "judge-第八讲-3"),
    ("judge-第八讲-5", "judge-第八讲-4"),
    ("judge-第八讲-13", "judge-第八讲-12"),
    ("single-第八讲-5", "multi-第八讲-3"),
    ("judge-第八讲-2", "judge-第八讲-1"),
    ("judge-第八讲-2", "judge-第五讲-16"),
    ("judge-第六讲-27", "judge-第六讲-26"),
    ("judge-第六讲-27", "single-第一讲-7"),
    ("judge-第六讲-27", "single-第一讲-9"),
    ("multi-第六讲-23", "judge-第六讲-25"),
    ("judge-第六讲-17", "judge-第六讲-16"),
    ("multi-第六讲-4", "multi-第六讲-2"),
    ("multi-第六讲-4", "multi-第六讲-3"),
    ("multi-第六讲-26", "judge-第九讲-14"),
    ("single-第四讲-2", "judge-第四讲-20"),
    ("single-第四讲-2", "judge-第四讲-18"),
    ("single-第四讲-2", "judge-第四讲-19"),
    ("judge-第四讲-14", "judge-第四讲-13"),
    ("judge-第四讲-14", "judge-第四讲-12"),
    ("multi-第四讲-18", "multi-第四讲-16"),
}


def main():
    wb = load_workbook(BOOK)
    ws = wb["2_等价组核对"]

    yes = no = 0
    unmatched = set(SAME)
    for r in range(2, ws.max_row + 1):
        t = ws.cell(row=r, column=1).value
        c = ws.cell(row=r, column=3).value
        if (t, c) in SAME:
            ws.cell(row=r, column=6, value="是")
            yes += 1
            unmatched.discard((t, c))
        else:
            ws.cell(row=r, column=6, value="否")
            no += 1

    wb.save(BOOK)
    print(f"已判定 {yes + no} 行：是 {yes} 行，否 {no} 行")
    if unmatched:
        print(f"[警告] {len(unmatched)} 组未在表中找到: {sorted(unmatched)}")


if __name__ == "__main__":
    main()
