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
from pathlib import Path

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "wrong_book.jsonl"


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
    raise NotImplementedError("add() 还没写")


def list_by_lecture(lecture: str = "", limit: int = 20) -> list[dict]:
    """按讲次查错题，最近的在前。

    要求：
      1. lecture 为空就返回全部
      2. 用 `lecture in 记录的讲次` 做匹配（讲次名带全角空格，如 "第三讲　古代中国"）
      3. 文件不存在时返回空列表，不要报错
      4. limit 限制返回条数
    """
    raise NotImplementedError("list_by_lecture() 还没写")


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
