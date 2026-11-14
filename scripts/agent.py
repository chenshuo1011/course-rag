"""
课程题库 Agent —— 手写工具调用循环（不用 LangChain）

和 ask.py 的区别：
  ask.py   ：提问 → 检索一次 → 生成一次。单向，模型只能看到这一次的材料。
  agent.py ：模型自己决定「查什么、查几次、什么时候够了」。
             它先看到工具清单，然后每一步输出一个 JSON 动作，
             执行完把结果喂回去，直到它给出最终回答或用完步数。

为什么要手写而不套框架：
  这个循环一共不到 60 行。写成自己的，才能回答「终止条件怎么定」
  「模型死循环了怎么办」这类问题——这类问题套框架的人答不上来。

三个终止条件（缺一不可）：
  1. 模型输出 final —— 正常结束
  2. 步数达到上限   —— 防止无限调用，烧钱
  3. 同一工具+同样参数重复 —— 检测原地打转，立刻干预

第 3 条是这次加的重点。之前做 AI 编程助手实验时见过它反复生成
同样的文件停不下来（实验 1），根因就是没有收敛判断。

用法：
    "D:/anaconda3/envs/yolo/python.exe" agent.py
    "D:/anaconda3/envs/yolo/python.exe" agent.py -q "第三讲有多少道判断题"
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from bm25 import BM25, minmax
from embedder import QUERY_INSTRUCTION, embed
from llm import LAST_USAGE, chat_msgs

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
W_VEC = 0.5
MAX_STEPS = 8

# 调模型的超时与重试（压测后定的）
#
# 原来超时 90 秒、重试 3 次：一旦 DeepSeek 卡住，一条线程被占住 90×3 秒，
# 40 条线程很快被占满，后面所有人排队。
# 实测服务端 p95 是 4.07 秒、最慢 15 秒，所以 15 秒足够覆盖正常情况，
# 超时也只重试 1 次——宁快速失败、走降级，不要拖死整条线程。
LLM_TIMEOUT = 15
LLM_RETRIES = 1

# ─────────────────────────── 题库索引 ───────────────────────────


def doc_text(m):
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    if m.get("explanation"):
        parts.append(m["explanation"])
    return " ".join(parts)


# 题库里有 1 道判断题的答案写成"否"（judge-第四讲-8），其余 170 道用"正确/错误"。
# 直接按字符串过滤会把它漏掉，所以统一归一化。
_ANSWER_MAP = {
    "正确": "正确", "对": "正确", "是": "正确", "√": "正确", "true": "正确", "t": "正确",
    "错误": "错误", "错": "错误", "否": "错误", "×": "错误", "false": "错误", "f": "错误",
}


def _norm_answer(a) -> str:
    if a is None:
        return ""
    s = str(a).strip()
    return _ANSWER_MAP.get(s, _ANSWER_MAP.get(s.lower(), ""))


class KB:
    """题库 + 检索器。工具都挂在这上面。"""

    def __init__(self):
        self.meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))
        self.vecs = np.load(DATA / "vectors.npy")
        self.bm = BM25([doc_text(m) for m in self.meta])
        self.by_id = {m["id"]: m for m in self.meta}

    def search(self, query: str, k: int = 3) -> list[dict]:
        qv = embed([QUERY_INSTRUCTION + query]).numpy()[0]
        vs = self.vecs @ qv
        bs = self.bm.scores(query)
        combined = W_VEC * minmax(vs) + (1 - W_VEC) * minmax(bs)

        order, seen = [], set()
        for i in np.argsort(-combined):
            m = self.meta[int(i)]
            key = re.sub(r"[\s　（）()、，。？,.:：]", "", m["stem"])
            if key in seen:
                continue
            seen.add(key)
            order.append(int(i))
            if len(order) >= k:
                break
        return [
            {
                "qid": self.meta[i]["id"],
                "stem": self.meta[i]["stem"],
                "answer": str(self.meta[i]["answer_text"] or self.meta[i]["answer"]),
                "score": round(float(combined[i]), 3),
            }
            for i in order
        ]

    def detail(self, qid: str) -> dict:
        m = self.by_id.get(qid)
        if not m:
            return {"error": f"没有这个题号：{qid}"}
        return {
            "qid": m["id"],
            "type": m["type"],
            "lecture": m["lecture"],
            "stem": m["stem"],
            "options": m.get("options") or {},
            "answer": str(m["answer_text"] or m["answer"]),
            "explanation": m.get("explanation") or "（题库没有给解析）",
        }

    def stats(self, lecture: str = "", type: str = "", answer: str = "") -> dict:
        rows = self.meta
        if lecture:
            rows = [m for m in rows if lecture in m["lecture"]]
        if type:
            rows = [m for m in rows if m["type"] == type]
        if answer:
            want = _norm_answer(answer)
            if not want:
                return {"error": f"answer 只能是 正确/错误（或 对/错/是/否），收到：{answer}"}
            rows = [m for m in rows if _norm_answer(m.get("answer")) == want]
        by_type, by_lecture, by_answer = {}, {}, {}
        for m in rows:
            by_type[m["type"]] = by_type.get(m["type"], 0) + 1
            by_lecture[m["lecture"]] = by_lecture.get(m["lecture"], 0) + 1
            a = _norm_answer(m.get("answer")) or "其他"
            by_answer[a] = by_answer.get(a, 0) + 1
        return {
            "count": len(rows),
            "by_type": dict(sorted(by_type.items())),
            "by_lecture": dict(sorted(by_lecture.items())),
            "by_answer": by_answer,
            "sample_ids": [m["id"] for m in rows[:5]],
        }

    def grep(self, keyword: str, scope: str = "stem", limit: int = 15) -> dict:
        """全库逐题扫关键词，返回全部命中。

        为什么需要它：
          search 返回的是"语义最相关的 top-k"，不是"含这个词的全部"。
          问"哪几讲提到盖伦"这类需要覆盖全库的问题，用 search 数数量必然出错
          （实测：会漏讲次，也会把天文学相关但没提盖伦的题算进来）。

        为什么要有 scope：
          同一个词有两种"出现"。
            stem —— 题干+解析里讲了它，等于"这道题考它"。问"哪几讲讲到了X"要这个。
            all  —— 连选项一起算。选项里常把名人列成干扰项，比如
                    「天空立法者是（ ）」的选项里有"第谷"，但那道题考的是开普勒。
          实测「第谷」：scope=stem 是 5 道，scope=all 是 8 道。混用就会答错数量。
        """
        kw = (keyword or "").strip()
        if not kw:
            return {"error": "keyword 不能为空"}

        def text_of(m):
            if scope == "all":
                return doc_text(m)
            base = [m["stem"]]
            if m.get("explanation"):
                base.append(m["explanation"])
            return " ".join(base)

        rows = [m for m in self.meta if kw in text_of(m)]
        by_lecture = {}
        for m in rows:
            by_lecture[m["lecture"]] = by_lecture.get(m["lecture"], 0) + 1
        return {
            "keyword": kw,
            "scope": "含选项" if scope == "all" else "题干+解析",
            "total": len(rows),
            "by_lecture": dict(sorted(by_lecture.items())),
            "truncated": len(rows) > limit,
            "hits": [
                {
                    "qid": m["id"],
                    "lecture": m["lecture"],
                    "type": m["type"],
                    "stem": m["stem"][:70],
                }
                for m in rows[:limit]
            ],
        }


# ─────────────────────────── 工具清单 ───────────────────────────

TOOLS = {
    "search": {
        "desc": "按意思检索题目。语义+关键词混合，返回最相关的几条（题号、题干、答案）。",
        "args": {"query": "检索语句，尽量用课程里的术语", "k": "返回条数，1~5，默认 3"},
    },
    "detail": {
        "desc": "按题号取一道题的完整信息：题干、选项、答案、解析。",
        "args": {"qid": "题号，如 single-第一讲-3"},
    },
    "stats": {
        "desc": "统计题库。可按讲次、题型、答案过滤，返回数量、题型/讲次/答案分布。"
                "问「有几道判断题答案是错的」「某讲有多少道错题」必须用这个。",
        "args": {
            "lecture": "讲次，如 第三讲；空表示全部",
            "type": "题型：single/multi/judge；空表示全部",
            "answer": "答案：正确/错误（也接受 对/错/是/否）；空表示全部。判断题的答案就是正确或错误",
        },
    },
    "grep": {
        "desc": "全库逐题扫描关键词，返回全部命中：总数、按讲次分布、每道题的题号与题干。"
                "问「哪几讲提到X」「一共几道提到X」必须用这个，不能用 search 数数量。",
        "args": {
            "keyword": "要查的词，如 盖伦、第谷",
            "scope": "stem=题干+解析（这道题考它）；all=连选项也算（选项里出现过）",
            "limit": "最多返回几道题的明细，默认 15",
        },
    },
}

SYSTEM = """你是课程题库答疑 Agent，可调用工具查库。

工具：
{tools}

每步只输出一个 JSON 对象，不要输出别的内容。
调用工具：{{"thought":"一句话理由","action":"工具名","args":{{...}}}}
给出回答：{{"thought":"够了","action":"final","answer":"口语回答，80 字内"}}

规则：
1. 不确定有没有相关内容，先 search；要看完整解析，用 detail。
2. 问数量、分布、第几讲几道，用 stats；问「几道答案是错的」，用 stats(answer="错误")，
   不要逐题调 detail。
3. 问「哪几讲提到X」或「一共几道提到X」，必须用 grep——search 只返回最相关几条，
   不能当全集。scope=stem 指这道题考它；scope=all 连选项也算（选项常拿名人当干扰项）。
4. 不要重复同样的查询；下结论前逐条核对结果，不能凭几条命中就说「其他都没有」。
5. 库里确实没有时，answer 写「题库里没有找到」，并说明最接近的是什么。
6. 只答学生问的那一点。判断题的答案是正确/错误，知识在解析里。"""


def build_system() -> str:
    lines = []
    for name, t in TOOLS.items():
        args = "；".join(f"{k}: {v}" for k, v in t["args"].items())
        lines.append(f"- {name}({args})\n  {t['desc']}")
    return SYSTEM.format(tools="\n".join(lines))


# ─────────────────────────── 循环 ───────────────────────────


def parse_action(text: str) -> dict:
    """模型偶尔会带 ```json 包裹或前后废话，这里做一次宽容解析。"""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"无法解析成 JSON：{text[:120]}")


def _llm(messages: list[dict], max_tokens: int) -> str:
    """所有调模型的地方都走这里，统一超时与重试次数。"""
    return chat_msgs(messages, max_tokens=max_tokens,
                     retries=LLM_RETRIES, timeout=LLM_TIMEOUT)


_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens",
               "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")


def _add_usage(acc: dict, u: dict) -> None:
    """把一次调用的用量累加到 acc。

    为什么必须累加：
      一个多步问题会调 2~3 次模型，每次都有输入输出。
      只记最后一次的话，3 步的问题成本会被低估 2/3——
      算钱这件事上，漏记比记错更危险。
    """
    for k in _USAGE_KEYS:
        v = u.get(k)
        if v:
            acc[k] = acc.get(k, 0) + int(v)


def _fallback(kb: KB, question: str, called: list, steps: list,
              t0: float, err: Exception, verbose: bool) -> dict:
    """模型调不通时的降级：退回纯检索，把最相关的几道题原样给出去。

    为什么不直接报错：
      用户要的是"这道题的答案在哪"，不是"我们的模型挂了"。
      检索不依赖大模型，它在模型不可用时仍然是可用的——
      降级能救回一部分价值，而报错只能救回一个 500。
    """
    hits = kb.search(question, k=3)
    lines = ["答案生成超时，以下是题库里最相关的题目："]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h['stem'][:40]} → {h['answer'][:30]}")
    if verbose:
        print(f"  降级：{type(err).__name__}: {str(err)[:60]}")
    return {
        "answer": "\n".join(lines),
        "steps": steps,
        "calls": called + ["search"],
        "elapsed": round(time.time() - t0, 1),
        "usage": {},
        "stop": "fallback",
        "fallback_qids": [h["qid"] for h in hits],
    }


def run(kb: KB, question: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": build_system()},
        {"role": "user", "content": f"学生的问题：{question}"},
    ]
    called: list[str] = []
    steps = []
    usage: dict = {}
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        try:
            raw = _llm(messages, 400)
        except Exception as e:
            return _fallback(kb, question, called, steps, t0, e, verbose)
        _add_usage(usage, LAST_USAGE)
        try:
            act = parse_action(raw)
        except ValueError as e:
            # 让它自己修一次，修不好就当作终止
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"上面的输出不是合法 JSON（{e}）。只输出一个 JSON 对象，不要别的内容。"})
            continue

        name = act.get("action", "")
        thought = act.get("thought", "")

        if name == "final":
            steps.append({"step": step, "action": "final", "thought": thought})
            if verbose:
                print(f"  [{step}] final —— {thought}")
            return {
                "answer": act.get("answer", "").strip(),
                "steps": steps,
                "calls": called,
                "elapsed": round(time.time() - t0, 1),
                "usage": usage,
                "stop": "final",
            }

        if name not in TOOLS:
            obs = {"error": f"没有这个工具：{name}。可用：{list(TOOLS)}"}
        else:
            args = act.get("args") or {}
            sig = name + json.dumps(args, ensure_ascii=False, sort_keys=True)
            if sig in called:
                obs = {"error": "这个调用刚才已经做过了，结果在上文。换个思路，或者直接给答案。"}
                if verbose:
                    print(f"  [{step}] {name}({args}) —— 重复调用，已拦下")
            else:
                called.append(sig)
                try:
                    obs = getattr(kb, name)(**args)
                except TypeError as e:
                    obs = {"error": f"参数不对：{e}。这个工具的参数是 {TOOLS[name]['args']}"}
                if verbose:
                    brief = _brief(name, obs)
                    print(f"  [{step}] {name}({_fmt_args(args)}) → {brief}")

        steps.append({"step": step, "action": name, "args": act.get("args"), "thought": thought})
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": "工具返回：\n" + json.dumps(obs, ensure_ascii=False, indent=1)})

    # 用完步数：强制收束，让它基于已有信息作答
    if verbose:
        print(f"  [{MAX_STEPS}] 达到步数上限，强制收束")
    messages.append({"role": "user",
                     "content": "已经用完工具调用次数。现在只能基于上面的结果直接给出 answer，"
                                "输出 {\"action\":\"final\",\"answer\":\"...\"}。信息不足就说明缺什么。"})
    try:
        raw = _llm(messages, 300)
        _add_usage(usage, LAST_USAGE)
    except Exception as e:
        return _fallback(kb, question, called, steps, t0, e, verbose)
    try:
        act = parse_action(raw)
        answer = act.get("answer", raw)
    except ValueError:
        answer = raw
    return {"answer": answer.strip(), "steps": steps, "calls": called,
            "elapsed": round(time.time() - t0, 1), "usage": usage, "stop": "max_steps"}


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _brief(name: str, obs) -> str:
    if isinstance(obs, dict) and "error" in obs:
        return "错误：" + obs["error"][:60]
    if name == "search":
        return f"{len(obs)} 条 | " + " / ".join(o["qid"] for o in obs)
    if name == "detail":
        return f"{obs['qid']} 答案={obs['answer'][:24]}"
    if name == "stats":
        return f"共 {obs['count']} 道，题型 {obs['by_type']}"
    if name == "grep":
        return f"共 {obs['total']} 条，讲次 {obs['by_lecture']}"
    return str(obs)[:60]


def main():
    ONE = "-q" in sys.argv
    if ONE:
        i = sys.argv.index("-q")
        questions = [" ".join(sys.argv[i + 1:])]
    else:
        questions = None

    print("加载索引与 BM25 ...")
    kb = KB()
    print(f"就绪：{len(kb.meta)} 道题 | 步数上限 {MAX_STEPS}")
    print("=" * 64)
    print("课程题库 Agent    输入 exit 退出")
    print("=" * 64)

    while True:
        if questions is not None:
            q = questions[0]
        else:
            try:
                q = input("\n你问：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "退出"):
                break

        print()
        r = run(kb, q)
        print()
        print("── 回答 " + "─" * 54)
        print("  " + r["answer"])
        tk = r["usage"].get("total_tokens")
        print(f"\n  [{r['stop']}] {len(r['calls'])} 次工具调用 · {r['elapsed']}s"
              + (f" · {tk} tokens" if tk else ""))

        if questions is not None:
            break


if __name__ == "__main__":
    main()
