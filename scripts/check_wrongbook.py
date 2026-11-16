"""
错题本验收：记录、折叠、待复习、标记掌握

重点验三件容易被忽略的事：
  1. 折叠正确：错了 → 掌握了 → 又错了，应该回到"待复习"
  2. 文件只增不改：标记掌握不能删掉原来的记录
  3. 口径区分：总条数（事件数）和待复习（按题去重）不是一回事

用法（在 scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" check_wrongbook.py
"""

import json
import shutil
from pathlib import Path

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
    print("=" * 62)
    print("第 1 步：空库不该崩")
    check("pending() 空列表", [], wrongbook.pending())
    st = wrongbook.stats()
    check("stats 待复习为 0", 0, st["待复习"])
    check("stats 已掌握为 0", 0, st["已掌握"])

    print("=" * 62)
    print("第 2 步：记三次错（同一题两次 + 另一题一次）")
    wrongbook.add("single-第五讲-1", "第五讲", "第谷用什么观测", "望远镜", "肉眼")
    wrongbook.add("single-第五讲-1", "第五讲", "第谷用啥观测的", "开普勒", "肉眼")
    wrongbook.add("judge-第三讲　古代中国-13", "第三讲　古代中国",
                  "第谷理论谁传的", "利玛窦", "耶稣会教士")

    st = wrongbook.stats()
    check("总条数（事件数）", 3, st["总条数"])
    check("待复习（按题去重）", 2, st["待复习"])
    rows = wrongbook.pending()
    check("pending 返回 2 条", 2, len(rows))
    check("同一题错两次，times 记为 2", 2,
          next((r["times"] for r in rows if r["qid"] == "single-第五讲-1"), None))
    check("最近的在前", "judge-第三讲　古代中国-13", rows[0]["qid"])

    print("=" * 62)
    print("第 3 步：标记掌握一道")
    wrongbook.master("single-第五讲-1", "第五讲")
    st = wrongbook.stats()
    check("待复习减为 1", 1, st["待复习"])
    check("已掌握 1", 1, st["已掌握"])
    check("总条数不变（记录还在）", 3, st["总条数"])
    check("文件变成 4 行（追加了 mastered）", 4,
          len([l for l in BOOK.read_text(encoding="utf-8").splitlines() if l.strip()]))

    print("=" * 62)
    print("第 4 步：掌握后又错一次，应该回到待复习")
    wrongbook.add("single-第五讲-1", "第五讲", "第三遍还是不会", "又错了", "肉眼")
    rows = wrongbook.pending()
    check("待复习回到 2", 2, len(rows))
    check("这题又在待复习里", True, any(r["qid"] == "single-第五讲-1" for r in rows))
    check("已掌握归零", 0, wrongbook.stats()["已掌握"])

    print("=" * 62)
    print("第 5 步：文件从头到尾只追加，没有丢记录")
    lines = [json.loads(l) for l in BOOK.read_text(encoding="utf-8").splitlines() if l.strip()]
    check("总行数 = 5", 5, len(lines))
    check("第 1 行仍是最早那条 add", "add", lines[0].get("op"))
    check("中间有一行 mastered", True, any(x.get("op") == "mastered" for x in lines))
    check("最后一个事件是 add", "add", lines[-1].get("op"))

    print("=" * 62)
    print("第 6 步：按讲次过滤")
    check("第五讲待复习 1 条", 1, len(wrongbook.pending("第五讲")))
    check("第三讲待复习 1 条", 1, len(wrongbook.pending("第三讲")))

    print("=" * 62)
    print(f"通过 {sum(results)} / {len(results)}")
    if all(results):
        print("错题闭环的数据层通过。下一步：接口和页面。")


if __name__ == "__main__":
    main()
