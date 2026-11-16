"""
答题模式的验收脚本

检查四件事：
  1. 抽题：只抽能自动判分的题型，给学生的版本里不能有答案
  2. 判分：单选 / 多选 / 判断，答对答错都要判对
  3. 判断题的特殊写法："否" 要当成 "错误"（题库里确有 1 道这么写）
  4. 答错自动记错题本，答对不记

用法（在 scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" check_quiz.py
"""

import random
import shutil
from pathlib import Path

import quiz
import wrongbook

BOOK = wrongbook.BOOK
BACKUP = Path(str(BOOK) + ".bak")
results = []


def check(title, expect, actual):
    ok = expect == actual
    print(f"  [{'OK ' if ok else 'BAD'}] {title}")
    if not ok:
        print(f"        期望: {expect}")
        print(f"        实际: {actual}")
    results.append(ok)


def main():
    if BOOK.exists():
        shutil.copy(BOOK, BACKUP)
        BOOK.unlink()
    try:
        run()
    finally:
        if BACKUP.exists():
            shutil.move(BACKUP, BOOK)
        elif BOOK.exists():
            BOOK.unlink()


def run():
    meta = quiz.load_meta()
    print(f"题库 {len(meta)} 道")

    print("=" * 60)
    print("第 1 步：抽题不给答案")
    items = quiz.pick(meta, lecture="第五讲", n=3, rng=random.Random(7))
    check("抽到 3 道", 3, len(items))
    leaked = [k for k in ("answer", "answer_text", "explanation") for it in items if k in it]
    check("没有泄漏 answer/answer_text/explanation", [], leaked)
    check("每道都有 qid 和题干", True, all(it.get("qid") and it.get("stem") for it in items))
    check("题型都在可自动判分的范围内", True,
          all(it["type"] in ("single", "multi", "judge") for it in items))

    print("=" * 60)
    print("第 2 步：按题型过滤")
    only_judge = quiz.pick(meta, qtype="judge", n=5, rng=random.Random(1))
    check("只抽判断题", True, all(it["type"] == "judge" for it in only_judge))
    check("题量不超过库存", True, len(only_judge) <= 5)

    print("=" * 60)
    print("第 3 步：判分")
    right_cases = [(m, m.get("answer_text") or m.get("answer"))
                   for m in meta if m.get("type") == "single"][:3]
    for m, _ in right_cases:
        r = quiz.grade(meta, m["id"], m["answer"])          # 答对
        check(f"单选答对（{m['id']}）", True, r["correct"])
        check(f"答对不记错题（{m['id']}）", False, r["recorded"])

    wrong_one = right_cases[0][0]
    wrong_letter = next(k for k in wrong_one["options"] if k != wrong_one["answer"])
    r = quiz.grade(meta, wrong_one["id"], wrong_letter)     # 答错
    check(f"单选答错（选 {wrong_letter}）", False, r["correct"])
    check("答错自动记录", True, r["recorded"])
    check("返回里有正确答案", True, bool(r["right"]))
    check("返回里有解析", True, bool(r["explanation"]))

    print("=" * 60)
    print("第 4 步：多选顺序无关")
    multi = next((m for m in meta if m.get("type") == "multi"), None)
    if multi:
        ans = multi["answer"]
        shuffled = "".join(reversed(list(ans)))
        r = quiz.grade(meta, multi["id"], shuffled, record_wrong=False)
        check(f"多选 {ans} 与 {shuffled} 视为相同", True, r["correct"])

    print("=" * 60)
    print("第 5 步：判断题的三种写法")
    judge = [m for m in meta if m.get("type") == "judge"]
    t_ok = next((m for m in judge if str(m.get("answer")).strip() == "正确"), None)
    f_ok = next((m for m in judge if str(m.get("answer")).strip() == "错误"), None)
    f_no = next((m for m in judge if str(m.get("answer")).strip() == "否"), None)
    if t_ok:
        check("答案为正确的题，答'正确'判对",
              True, quiz.grade(meta, t_ok["id"], "正确", record_wrong=False)["correct"])
        check("答案为正确的题，答'错误'判错",
              False, quiz.grade(meta, t_ok["id"], "错误", record_wrong=False)["correct"])
    if f_ok:
        check("答案为错误的题，答'错误'判对",
              True, quiz.grade(meta, f_ok["id"], "错误", record_wrong=False)["correct"])
    if f_no:
        check(f"答案为'否'的题（{f_no['id']}），答'错误'也算对",
              True, quiz.grade(meta, f_no["id"], "错误", record_wrong=False)["correct"])
    else:
        print("  [跳过] 题库里没有答案为'否'的判断题")

    print("=" * 60)
    print("第 6 步：错题本确实写进去了")
    rows = wrongbook.list_by_lecture()
    check("至少记录了一条", True, len(rows) >= 1)
    check("最上面这条是刚才错的", wrong_one["id"], rows[0]["qid"] if rows else None)

    print("=" * 60)
    print("第 7 步：题号不存在时给错误而不是崩")
    r = quiz.grade(meta, "no-such-id", "A")
    check("返回 error 字段", True, "error" in r)

    print("=" * 60)
    print(f"通过 {sum(results)} / {len(results)}")


if __name__ == "__main__":
    main()
