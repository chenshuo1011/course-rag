"""
错题本：把答错的题记下来，能按讲次查、能统计

为什么要有它：
  学生复习时最想要的不是"再问一遍"，而是"我上次错在哪"。
  服务现在答完就忘了，没有任何记忆。

数据格式：data/wrong_book.jsonl，一行一条
  {"t": "2026-11-16 15:00:00",
   "qid": "single-第五讲-1",
   "lecture": "第五讲",
   "q": "学生原话",
   "wrong": "学生选错的答案",
   "right": "正确答案"}

为什么用 jsonl 而不是一个 json 数组：
  追加记录时不用读全文重写。json 数组要整份覆盖，记录一多、
  并发一高就容易丢数据。
"""

import json
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "wrong_book.jsonl"

# 服务里多个请求可能同时调用 add()。不加锁的话，
# 几次写入会互相插进去，文件里出现半行 JSON
# （server.py 的日志就踩过：362 行坏了 3 行）。
_write_lock = threading.Lock()


# ─────────────────────── 公共逻辑 ───────────────────────

def _read_all() -> list[dict]:
    """读出全部记录。

    单独抽出来，是因为 list_by_lecture 和 stats 都要用。
    同一段读文件的代码复制两遍，以后改格式就要改两处——一定会漏一处。

    三个细节：
      1. 文件不存在 → 返回空列表，不抛异常（新用户第一次查询就是这种情况）
      2. 空行跳过
      3. 坏行跳过：压测见过写坏的行，一条坏的不能让整个功能挂掉
    """
    if not BOOK.exists():
        return []

    rows = []
    with open(BOOK, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write(rec: dict) -> None:
    """把一条记录追加到文件末尾。"""
    BOOK.parent.mkdir(parents=True, exist_ok=True)   # data 目录可能不存在
    line = json.dumps(rec, ensure_ascii=False)      # 中文不转成 \uXXXX
    with _write_lock:
        with open(BOOK, "a", encoding="utf-8") as f:
            f.write(line + "\n")                    # 换行 = 一行一条
            f.flush()


# ─────────────────────── 三个功能 ───────────────────────

def add(qid: str, lecture: str, question: str,
        wrong: str, right: str) -> dict:
    """追加一条错题记录，返回刚写入的这条。

    同一个题号可以记多次（学生可能反复错），这里不去重——
    去重是查询时的事，记录本身要保留每次的痕迹。
    """
    rec = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qid": qid,
        "lecture": lecture,
        "q": question,
        "wrong": wrong,
        "right": right,
    }
    _write(rec)
    return rec


def list_by_lecture(lecture: str = "", limit: int = 20) -> list[dict]:
    """按讲次查错题，最近的在前。

    lecture 为空返回全部；匹配用 `in`，因为讲次名带全角空格
    （如 "第三讲　古代中国"），传 "第三讲" 也要能匹配上。
    """
    rows = _read_all()

    if lecture:
        rows = [r for r in rows if lecture in r.get("lecture", "")]

    # 文件是追加写的，后写的在后面；倒过来就是最近的在前
    # （时间格式 %Y-%m-%d %H:%M:%S 的字符串顺序恰好等于时间顺序）
    return rows[::-1][:limit]


def stats() -> dict:
    """统计各讲错了多少道。

    返回三个键：
      总条数  —— 一共记了多少条（含重复）
      按讲次  —— 每讲多少条
      去重    —— 每讲有多少道**不同的题**（同一题错三次算一道）

    为什么要有去重这个口径：
      学生反复错同一道题，说明这道题是真不会。
      按条数看，它会让这一讲显得错很多，误导复习重点。
    """
    rows = _read_all()

    # Counter 就是"边遍历边计数"，等价于手写 dict.get(k,0)+1
    per_lecture = Counter(r.get("lecture", "") for r in rows)

    # 去重：把 (讲次, 题号) 放进 set，重复的自动合并
    seen = {(r.get("lecture", ""), r.get("qid", "")) for r in rows}
    unique = Counter(lecture for lecture, _qid in seen)

    return {
        "总条数": len(rows),
        "按讲次": dict(per_lecture),
        "去重": dict(unique),
    }


if __name__ == "__main__":
    # 直接跑这个文件时，打印当前统计，方便人工看一眼
    print(json.dumps(stats(), ensure_ascii=False, indent=1))
