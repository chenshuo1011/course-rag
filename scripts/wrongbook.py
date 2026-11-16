"""
错题本：把答错的题记下来，能按讲次查、能统计

为什么要有它：
  学生复习时最想要的不是"再问一遍"，而是"我上次错在哪"。
  服务现在答完就忘了，没有任何记忆。

数据格式：data/wrong_book.jsonl，一行一条（jsonl 比 json 数组好，因为追加不用读全文）

  {"t": "2026-11-16 15:00:00",
   "qid": "single-第五讲-1",
   "lecture": "第五讲",
   "q": "学生原话",
   "wrong": "学生选错的答案",
   "right": "正确答案"}

要你自己写的三个函数，从 add() 开始，写完一个跑一次 check_wrongbook.py。
"""

import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "wrong_book.jsonl"

# 和 server.py 里那个日志 bug 一样：多个请求同时调用 add() 时，
# 如果不加锁，几次写入会互相插进去，文件里出现半行 JSON。
_write_lock = threading.Lock()


def add(qid: str, lecture: str, question: str,
        wrong: str, right: str) -> dict:
    """追加一条错题记录，返回刚写入的这条。

    要求：
      1. 时间用 time.strftime("%Y-%m-%d %H:%M:%S")
      2. 写文件用 "a" 模式（追加），一行一条，行尾换行
      3. data 目录可能不存在，要自己建
      4. 同一个题号可以记多次（学生可能反复错），不去重

    提示：今天上午修过一个 bug——多个线程同时以 "a" 模式写同一文件，
    内容会互相截断。想想这里会不会遇到同样的问题。
    """
    # 第 1 步：把这条记录组装成一个字典（键名要和上面文档里写的一致）
    rec = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qid": qid,
        "lecture": lecture,
        "q": question,
        "wrong": wrong,
        "right": right,
    }

    # 第 2 步：确保 data 目录存在（不存在就建，已存在也不报错）
    BOOK.parent.mkdir(parents=True, exist_ok=True)

    # 第 3 步：字典 → 一行文本
    #   ensure_ascii=False 不能省，否则中文会变成 \u7b2c\u4e94\u8bb2
    line = json.dumps(rec, ensure_ascii=False)

    # 第 4 步：追加写入，行尾加 "\n" 才是"一行一条"
    #   加锁是因为服务里可能有多个请求同时调用它（见文件顶部说明）
    with _write_lock:
        with open(BOOK, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    return rec


def list_by_lecture(lecture: str = "", limit: int = 20) -> list[dict]:
    """按讲次查错题，最近的在前。

    要求：
      1. lecture 为空就返回全部
      2. 用 `lecture in 记录的讲次` 做匹配（讲次名带全角空格，如 "第三讲　古代中国"）
      3. 文件不存在时返回空列表，不要报错
      4. limit 限制返回条数
    """
    # 第 ① 步：文件还不存在（从来没记过错题）就直接返回空
    if not BOOK.exists():
        return []

    # 第 ② 步：读进来，每行变成字典
    #   json.loads 是 add() 里 json.dumps 的反向操作：文本 → 字典
    #   坏行跳过——压测时见过写坏的行，一条坏的不该让整个功能挂掉
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

    # 第 ③ 步：按讲次过滤。lecture 为空就不过滤
    if lecture:
        rows = [r for r in rows if lecture in r.get("lecture", "")]

    # 第 ④ 步：倒序（后写的在后面，倒过来就是最近的在前）+ 截断
    return rows[::-1][:limit]


def stats() -> dict:
    """统计各讲错了多少道，返回 {"总条数": n, "按讲次": {...}}。

    要求：
      1. 文件不存在时返回 {"总条数": 0, "按讲次": {}}
      2. 按讲次名分组计数，讲次名保留原样
      3. 另外给出每讲"错的题号去重后有几道"——同一个题号错三次算一道

    返回示例：
      {"总条数": 5, "按讲次": {"第五讲": 3, "第三讲　古代中国": 2},
       "去重": {"第五讲": 2, "第三讲　古代中国": 2}}
    """
    raise NotImplementedError("stats() 还没写")
