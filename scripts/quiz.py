"""
答题模式：抽题、判分、答错自动记进错题本

和问答模式（agent.py）的区别：
  问答要模型决定"查什么"，多步工具调用；
  答题不需要模型——题目是给定的，判分是比对，纯数据操作。

所以这两条路不共享流程，只共享题库（meta）和错题本（wrongbook）。

一个设计取舍：
  quiz.py 里的函数都接收 meta 列表，而不是自己去加载题库。
  这样服务端可以把已经加载好的 kb.meta 传进来，不重复加载；
  而且这个模块完全不依赖 torch——答题不需要 embedding，
  为了读一个 json 去加载 100MB 的模型没有道理。
"""

import random

from wrongbook import add as wb_add

# 判断题的答案有三种写法：正确 / 错误 / 否（题库里确实有 1 道写"否"）。
# 和 agent.py 里的 _norm_answer 同义，但故意不 import：
# import agent 会连带加载 torch 和 embedding 模型，为了归一化一个词不值得。
_TRUE = {"正确", "对", "是", "√", "true", "t"}
_FALSE = {"错误", "错", "否", "×", "false", "f"}


def _norm_judge(x) -> str:
    s = str(x or "").strip()
    low = s.lower()
    if s in _TRUE or low in _TRUE:
        return "正确"
    if s in _FALSE or low in _FALSE:
        return "错误"
    return s


def _norm_choice(x) -> str:
    """选择题的答案统一成大写、去空格、排序后的字母串，如 "abd" → "ABD"。"""
    s = "".join(str(x or "").split()).upper()
    return "".join(sorted(s))


def pick(meta: list[dict], lecture: str = "", qtype: str = "",
         n: int = 1, rng: random.Random | None = None) -> list[dict]:
    """抽 n 道题（只抽能自动判分的：单选、多选、判断）。

    返回的是"学生视角"的题：有题干和选项，没有答案和解析。
    """
    pool = [m for m in meta if m.get("type") in ("single", "multi", "judge")]
    if lecture:
        pool = [m for m in pool if lecture in m.get("lecture", "")]
    if qtype:
        pool = [m for m in pool if m.get("type") == qtype]
    if not pool:
        return []

    r = rng or random.Random()
    picked = r.sample(pool, min(n, len(pool)))
    return [_public_view(m) for m in picked]


def _public_view(m: dict) -> dict:
    """给学生的版本：去掉答案、解析、id 之外的一切内部字段。"""
    return {
        "qid": m["id"],
        "lecture": m.get("lecture", ""),
        "type": m.get("type", ""),
        "stem": m.get("stem", ""),
        "options": m.get("options") or {},
    }


def grade(meta: list[dict], qid: str, chosen: str,
          record_wrong: bool = True) -> dict:
    """判一道题，返回结果；答错时自动记进错题本。

    chosen 的写法：
      单选 "B"；多选 "ABD"（顺序无所谓）；判断 "正确" 或 "错误"。
    """
    m = next((x for x in meta if x["id"] == qid), None)
    if m is None:
        return {"error": f"没有这个题号：{qid}"}

    t = m.get("type")
    if t == "judge":
        correct = _norm_judge(chosen) == _norm_judge(m.get("answer"))
    else:
        correct = _norm_choice(chosen) == _norm_choice(m.get("answer"))

    right = str(m.get("answer_text") or m.get("answer") or "")
    result = {
        "qid": qid,
        "correct": correct,
        "chosen": chosen,
        "right": right,
        "explanation": m.get("explanation") or "",
        "recorded": False,
    }

    # 答错就记一条。这里不判断"是不是重复错"——那是统计口径，交给 wrongbook.stats 去重
    if not correct and record_wrong:
        wb_add(
            qid=qid,
            lecture=m.get("lecture", ""),
            question=m.get("stem", ""),
            wrong=str(chosen),
            right=right,
        )
        result["recorded"] = True

    return result


def load_meta(path=None) -> list[dict]:
    """给命令行/验收脚本用：从 json 读题库（服务端请直接传 kb.meta，别走这里）。"""
    import json
    from pathlib import Path

    p = path or (Path(__file__).parent.parent / "data" / "index_meta.json")
    return json.loads(Path(p).read_text(encoding="utf-8"))


if __name__ == "__main__":
    meta = load_meta()
    print(f"题库 {len(meta)} 道，可出题 {sum(1 for m in meta if m.get('type') in ('single','multi','judge'))} 道")
    for q in pick(meta, lecture="第五讲", n=2, rng=random.Random(1)):
        print()
        print(f"【{q['type']}】{q['stem'][:60]}")
        for k, v in q["options"].items():
            print(f"   {k}. {v}")
