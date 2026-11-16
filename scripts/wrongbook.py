"""
错题本：记录错题、复习重做、答对后标记已掌握

数据格式：data/wrong_book.jsonl，一行一个"事件"（只追加，从不修改）

  答错：{"op": "add",      "t": "...", "qid": "...", "lecture": "...",
         "q": "学生原话", "wrong": "选错的答案", "right": "正确答案"}
  掌握：{"op": "mastered", "t": "...", "qid": "...", "lecture": "..."}

为什么用"事件"而不是直接改状态：
  想标记一道题已掌握，最直觉的做法是读出整个文件、改那条记录、再整份写回去。
  问题是——写回的那一刻，别的请求可能正在追加新记录，那些新记录会被覆盖掉。
  （同一道题错了 → 复习对了 → 又错了，这种反复是最常见的情况，不能丢。）

  所以这里改成：文件永远只追加，状态是**读的时候算出来的**（把事件按时间折叠）。
  好处是写入永远是加一条，不会覆盖任何东西；
  代价是读的时候要多做一步折叠——496 道题的规模下，这个代价可以忽略。

  这套做法叫「事件日志 + 状态折叠」，银行流水、订单状态、Git 都是这个思路。
"""

import json
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "wrong_book.jsonl"

# 服务里多个请求可能同时写。不加锁的话，几次写入会互相插进去
# （server.py 的日志踩过：362 行坏了 3 行）。
_write_lock = threading.Lock()


# ─────────────────────── 读写底层 ───────────────────────

def _read_all() -> list[dict]:
    """读出全部事件。文件不存在返回空，坏行跳过。"""
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


def _append(event: dict) -> None:
    """追加一个事件。"""
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with _write_lock:
        with open(BOOK, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


# ─────────────────────── 状态折叠 ───────────────────────

def fold(rows: list[dict]) -> dict[str, dict]:
    """把事件流折叠成"每道题当前是什么状态"。

    规则简单：同一道题，最后一个事件说了算。
      add       → 待复习（pending）
      mastered  → 已掌握（mastered）

    所以"错了 → 对了 → 又错了"会自然变成待复习，不需要特殊处理——
    只要按写入顺序覆盖即可。
    """
    state: dict[str, dict] = {}

    # 用行号当排序依据，而不是时间字符串：
    # 同一秒内可能写入多条，秒级时间戳会撞车，分不出先后。
    # 文件是只追加的，所以「行号大的更新」这件事永远成立。
    for seq, r in enumerate(rows):
        qid = r.get("qid")
        if not qid:
            continue
        op = r.get("op", "add")     # 兼容最早那批没有 op 字段的记录
        prev = state.get(qid)

        if op == "mastered":
            state[qid] = {
                "qid": qid,
                "lecture": r.get("lecture", "") or (prev or {}).get("lecture", ""),
                "status": "mastered",
                "t": r.get("t", ""),
                "seq": seq,
                "times": (prev or {}).get("times", 1),
            }
        else:
            times = (prev or {}).get("times", 0) + 1
            state[qid] = {
                "qid": qid,
                "lecture": r.get("lecture", ""),
                "status": "pending",
                "t": r.get("t", ""),
                "seq": seq,
                "q": r.get("q", ""),
                "wrong": r.get("wrong", ""),
                "right": r.get("right", ""),
                "times": times,
            }
    return state


def _current() -> list[dict]:
    """当前状态（每道题一条），按最近时间倒序。"""
    st = fold(_read_all())
    return sorted(st.values(), key=lambda x: x.get("seq", 0), reverse=True)


# ─────────────────────── 写事件 ───────────────────────

def add(qid: str, lecture: str, question: str,
        wrong: str, right: str) -> dict:
    """记一次答错。同一道题错多次就记多条。"""
    ev = {
        "op": "add",
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qid": qid,
        "lecture": lecture,
        "q": question,
        "wrong": wrong,
        "right": right,
    }
    _append(ev)
    return ev


def master(qid: str, lecture: str = "") -> dict:
    """标记一道题已掌握（复习时答对了）。"""
    ev = {
        "op": "mastered",
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qid": qid,
        "lecture": lecture,
    }
    _append(ev)
    return ev


# ─────────────────────── 查询 ───────────────────────

def pending(lecture: str = "", limit: int = 50) -> list[dict]:
    """待复习（还没掌握的），最近的在前。"""
    rows = [r for r in _current() if r["status"] == "pending"]
    if lecture:
        rows = [r for r in rows if lecture in r.get("lecture", "")]
    return rows[:limit]


def list_by_lecture(lecture: str = "", limit: int = 20) -> list[dict]:
    """全部记录（含已掌握的），最近的在前。保留这个函数是为了兼容旧调用。"""
    rows = _current()
    if lecture:
        rows = [r for r in rows if lecture in r.get("lecture", "")]
    return rows[:limit]


def stats() -> dict:
    """统计：事件数、待复习、已掌握，各带按讲次分布。

    注意两个口径的区别：
      总条数  —— 记了多少次答错（事件数，重复计）
      待复习  —— 现在还有几道题没掌握（按题去重）
    """
    rows = _read_all()
    cur = fold(rows)

    adds = [r for r in rows if r.get("op", "add") == "add"]
    pend = [r for r in cur.values() if r["status"] == "pending"]
    done = [r for r in cur.values() if r["status"] == "mastered"]

    # 去重口径：每讲有多少道**不同的题**（同一题错三次算一道）
    seen = {(r.get("lecture", ""), r.get("qid", "")) for r in adds}

    return {
        "总条数": len(adds),
        "按讲次": dict(Counter(r.get("lecture", "") for r in adds)),
        "去重": dict(Counter(l for l, _q in seen)),
        "待复习": len(pend),
        "待复习按讲次": dict(Counter(r.get("lecture", "") for r in pend)),
        "已掌握": len(done),
        "事件数": len(rows),
    }


if __name__ == "__main__":
    print(json.dumps(stats(), ensure_ascii=False, indent=1))
