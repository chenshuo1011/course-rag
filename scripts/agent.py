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
MAX_STEPS = 6

# ─────────────────────────── 题库索引 ───────────────────────────


def doc_text(m):
    parts = [m["stem"]]
    if m.get("options"):
        parts.extend(f"{k}. {v}" for k, v in m["options"].items())
    if m.get("explanation"):
        parts.append(m["explanation"])
    return " ".join(parts)


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

    def stats(self, lecture: str = "", type: str = "") -> dict:
        rows = self.meta
        if lecture:
            rows = [m for m in rows if lecture in m["lecture"]]
        if type:
            rows = [m for m in rows if m["type"] == type]
        by_type, by_lecture = {}, {}
        for m in rows:
            by_type[m["type"]] = by_type.get(m["type"], 0) + 1
            by_lecture[m["lecture"]] = by_lecture.get(m["lecture"], 0) + 1
        return {
            "count": len(rows),
            "by_type": dict(sorted(by_type.items())),
            "by_lecture": dict(sorted(by_lecture.items())),
            "sample_ids": [m["id"] for m in rows[:5]],
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
        "desc": "统计题库。可按讲次和题型过滤，返回数量、题型分布、讲次分布。",
        "args": {"lecture": "讲次，如 第三讲；空表示全部", "type": "题型：single/multi/judge；空表示全部"},
    },
}

SYSTEM = """你是课程答疑 Agent，能调用工具查题库。

可用工具：
{tools}

每一步只输出一个 JSON 对象，不要输出别的内容，不要在 JSON 外面加解释。

要用工具时：
{{"thought": "为什么查这个（一句话）", "action": "工具名", "args": {{...}}}}

要回答时：
{{"thought": "材料够了", "action": "final", "answer": "给学生的口语回答，80 字以内"}}

规则：
1. 不确定题库里有没有相关内容时，先 search，不要凭空回答。
2. 需要看某题完整解析时，用 detail。
3. 问数量、分布、第几讲有几道这类问题时，用 stats，不要靠 search 猜。
4. search 结果不足以回答就换一组词再查，但不要重复同样的查询。
5. 题库里确实没有相关内容时，answer 写"题库里没有找到"，并说明查到的最接近的是什么。
6. 回答只讲学生问的那一点，不要捎带无关信息。
7. 判断题的"答案"是正确/错误，真正的知识在解析里。"""


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


def run(kb: KB, question: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": build_system()},
        {"role": "user", "content": f"学生的问题：{question}"},
    ]
    called: list[str] = []
    steps = []
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        raw = chat_msgs(messages, max_tokens=400)
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
                "usage": dict(LAST_USAGE),
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
    raw = chat_msgs(messages, max_tokens=300)
    try:
        act = parse_action(raw)
        answer = act.get("answer", raw)
    except ValueError:
        answer = raw
    return {"answer": answer.strip(), "steps": steps, "calls": called,
            "elapsed": round(time.time() - t0, 1), "usage": dict(LAST_USAGE), "stop": "max_steps"}


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
