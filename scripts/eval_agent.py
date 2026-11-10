"""
Agent 评测 —— 10 条必须多步才能答对的问题

为什么要单独一套评测：
  make_worktable.py 生成的那 150 条，测的是"检索有没有命中那道题"。
  Agent 答的是另一类问题：哪几讲提到某个词、一共几道、哪讲最多。
  用 hit@1 评 Agent，等于用尺子量温度。

标准答案从哪来：
  下面 EXPECT 里的数字，是直接扫 data/index_meta.json 得到的，
  和 agent.py 里的 KB.grep / KB.stats 是两套独立实现。
  启动时会用 raw_scan() 再算一遍，对不上直接报错——
  这样"标准答案"不会因为被测代码改了而悄悄跟着变。

判分方式（全部可自动判，不靠人眼）：
  coverage  答案里必须出现全部讲次 + 正确数量
  stat      答案里必须出现那个数字（可选：还要出现指定词，如"第六讲"）
  qa        答案里必须出现关键答案词
  refuse    答案必须表示"没有/没讲到"

用法：
    "D:/anaconda3/envs/yolo/python.exe" eval_agent.py
    "D:/anaconda3/envs/yolo/python.exe" eval_agent.py -k cov     # 只跑某一类
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

ITEMS = [
    # id, 问题, 类型, 期望
    ("cov-1", "哪几讲提到了盖伦", "coverage",
     {"lectures": ["第二讲", "第四讲"], "count": 3, "kw": "盖伦"}),
    ("cov-2", "哪几讲提到了第谷", "coverage",
     {"lectures": ["第三讲", "第五讲"], "count": 5, "kw": "第谷"}),
    ("cov-3", "哪几讲提到了张衡", "coverage",
     {"lectures": ["第三讲"], "count": 3, "kw": "张衡"}),
    ("cov-4", "哪几讲提到了开普勒", "coverage",
     {"lectures": ["第二讲", "第四讲", "第五讲"], "count": 12, "kw": "开普勒"}),
    ("cnt-1", "第三讲有多少道判断题", "stat", {"n": 18}),
    ("cnt-2", "判断题一共有多少道", "stat", {"n": 171}),
    ("cnt-3", "哪一讲的题目最多，有多少道", "stat", {"n": 72, "must": "第六讲"}),
    ("cmp-1", "多选题比单选题多几道", "stat", {"n": 7}),
    ("qa-1", "第谷接受的是哪种学说", "qa", {"must": "地心说"}),
    ("ref-1", "红烧肉怎么做才好吃", "refuse", {}),

    # ── 以下 5 条是"工具选对也可能答错"的题 ─────────────────────────
    # 前面 10 条只要工具选对就必然对（grep 给总数、stats 给数量）。
    # 这 5 条要模型自己做集合运算或交叉统计，现有 4 个工具都不直接支持。
    # 这题试过三种判法，都不如人意：
    #   ① forbid 其他讲次 → 误杀，正确答案会顺带说"第二讲只提到盖伦"
    #   ② 要求出现"唯一/只有/重合" → 也误杀，模型换了个说法"第二讲只提到盖伦，没有牛顿"
    # 语义等价的表述太多，词表法判不干净。所以只查"有没有说出第四讲"，宁可松一点，
    # 结论靠人抽查兜底——这也说明自动判分不是万能的，有些题必须人工看。
    ("h-1", "盖伦和牛顿有没有出现在同一讲里？是哪一讲", "qa", {"must": "第四讲"}),
    ("h-2", "哪几讲的判断题比多选题多", "set",
     {"must_all": ["第一讲", "第二讲", "第八讲"], "forbid": []}),
    ("h-3", "哪几讲完全没有提到牛顿", "set",
     {"must_all": ["第一讲", "第二讲", "第三讲", "第七讲"], "forbid": []}),
    ("h-4", "第谷和开普勒是什么关系", "qa", {"must": "火星"}),
    ("h-5", "第三讲和第五讲，哪讲的选择题更多", "qa", {"must": "第三讲"}),

    # ── 以下 2 条是实测确认会失败的题 ───────────────────────────────
    # 学生复习时最自然的问法之一，但 stats 只能按讲次/题型过滤，
    # 没有"按答案筛选"这个维度，grep 又只能按关键词扫。
    # 实测它会诚实拒答"工具没法按答案统计"——比硬编强，但没解决问题。
    # 留着这两条，评测才有区分度：全是 15/15 的评测说明不了任何事。
    ("h-6", "判断题里有几道答案是错误的", "stat", {"n": 87}),
    ("h-7", "第三讲有几道判断题答案是错的", "stat", {"n": 11}),
]

CN = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
      6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
      11: "十一", 12: "十二", 15: "十五", 18: "十八", 72: "七十二", 171: "一百七十一"}


def raw_scan():
    """独立于 agent.py 的标准答案来源：直接读 JSON 自己数。"""
    meta = json.loads((DATA / "index_meta.json").read_text(encoding="utf-8"))

    def text(m):
        return m["stem"] + " " + (m.get("explanation") or "")

    out = {}
    for kw in ["盖伦", "第谷", "张衡", "开普勒"]:
        rows = [m for m in meta if kw in text(m)]
        out[kw] = (len(rows), {m["lecture"] for m in rows})
    out["_third_judge"] = sum(
        1 for m in meta if m["type"] == "judge" and "第三讲" in m["lecture"])
    out["_judge"] = sum(1 for m in meta if m["type"] == "judge")
    by = {}
    for m in meta:
        by[m["lecture"]] = by.get(m["lecture"], 0) + 1
    out["_max_lecture"] = max(by.items(), key=lambda x: x[1])
    out["_multi_minus_single"] = (
        sum(1 for m in meta if m["type"] == "multi") -
        sum(1 for m in meta if m["type"] == "single"))

    def lecs(kw):
        return {m["lecture"].replace("\u3000", " ").split()[0]
                for m in meta if kw in text(m)}

    all_lects = {m["lecture"].replace("\u3000", " ").split()[0] for m in meta}
    out["_both_galan_newton"] = lecs("盖伦") & lecs("牛顿")
    out["_no_newton"] = all_lects - lecs("牛顿")
    cross = {}
    for m in meta:
        lec = m["lecture"].replace("\u3000", " ").split()[0]
        cross.setdefault(lec, {"single": 0, "multi": 0, "judge": 0})[m["type"]] += 1
    out["_judge_gt_multi"] = {k for k, v in cross.items() if v["judge"] > v["multi"]}
    out["_third_single"] = cross["第三讲"]["single"]
    out["_fifth_single"] = cross["第五讲"]["single"]
    # 题库有 1 道判断题答案写成"否"（judge-第四讲-8），不是"错误"。
    # 语义上它也算错，所以这里归一化后再统计——直接比字符串会少算 1 道。
    wrong = [m for m in meta if m["type"] == "judge"
             and str(m.get("answer")).strip() in ("错误", "否", "错")]
    out["_judge_wrong"] = len(wrong)
    lw = {}
    for m in wrong:
        lec = m["lecture"].replace("\u3000", " ").split()[0]
        lw[lec] = lw.get(lec, 0) + 1
    out["_judge_wrong_lecture"] = lw
    return out


def check_ground_truth(scan):
    """标准答案写死在 ITEMS 里，这里用独立扫描核对，对不上就停。"""
    bad = []
    for _id, _q, kind, exp in ITEMS:
        if kind != "coverage":
            continue
        n, lecs = scan[exp["kw"]]
        if n != exp["count"]:
            bad.append(f"{exp['kw']} 数量：写死 {exp['count']}，实扫 {n}")
        norm = {l.replace("\u3000", " ").split()[0] for l in lecs}
        if norm != set(exp["lectures"]):
            bad.append(f"{exp['kw']} 讲次：写死 {exp['lectures']}，实扫 {sorted(norm)}")
    if scan["_third_judge"] != 18:
        bad.append(f"第三讲判断题：写死 18，实扫 {scan['_third_judge']}")
    if scan["_judge"] != 171:
        bad.append(f"判断题总数：写死 171，实扫 {scan['_judge']}")
    lec, n = scan["_max_lecture"]
    if n != 72:
        bad.append(f"最多的一讲：写死 72，实扫 {n}")
    if scan["_multi_minus_single"] != 7:
        bad.append(f"多选-单选：写死 7，实扫 {scan['_multi_minus_single']}")

    # 难题那 5 条也要用独立扫描核对，不能只在脑子里算过
    if scan["_both_galan_newton"] != {"第四讲"}:
        bad.append(f"盖伦∩牛顿：写死 ['第四讲']，实扫 {sorted(scan['_both_galan_newton'])}")
    if scan["_judge_gt_multi"] != {"第一讲", "第二讲", "第八讲"}:
        bad.append(f"判断>多选：写死 3 讲，实扫 {sorted(scan['_judge_gt_multi'])}")
    if scan["_no_newton"] != {"第一讲", "第二讲", "第三讲", "第七讲"}:
        bad.append(f"未提牛顿：写死 4 讲，实扫 {sorted(scan['_no_newton'])}")
    if scan["_third_single"] <= scan["_fifth_single"]:
        bad.append(f"第三讲单选题 {scan['_third_single']} 并不多于第五讲 {scan['_fifth_single']}")
    if scan["_judge_wrong"] != 87:
        bad.append(f"判断题答案=错误：写死 87，实扫 {scan['_judge_wrong']}")
    if scan["_judge_wrong_lecture"]["第三讲"] != 11:
        bad.append(f"第三讲判断题答案=错误：写死 11，实扫 {scan['_judge_wrong_lecture']['第三讲']}")
    return bad


def has_number(answer: str, n: int) -> bool:
    if re.search(rf"(?<!\d){n}(?!\d)", answer):
        return True
    return CN.get(n, "\0") in answer


def affirm(answer: str, term: str) -> bool:
    """答案里是否**肯定地**出现了这个词。

    为什么要判否定：实测踩过。
    「第谷接受的是哪种学说」这道题，模型答"地心说（未接受哥白尼日心说）"。
    如果只用 `term in answer` 判"哥白尼"，这句否定句会被判成通过，
    一个错误的标准答案+太宽的断言，会合谋给出假的 10/10。
    """
    for m in re.finditer(re.escape(term), answer):
        head = answer[max(0, m.start() - 4): m.start()]
        if not re.search(r"[未不没非]|不是|并非", head):
            return True
    return False


def judge(kind: str, exp: dict, answer: str) -> tuple[bool, str]:
    if kind == "refuse":
        ok = any(w in answer for w in ("没有找到", "没讲到", "没有相关", "未涉及", "不涉及", "没有讲"))
        return ok, "" if ok else "没拒答"

    if kind == "qa":
        ok = affirm(answer, exp["must"])
        return ok, "" if ok else f"没有肯定地给出「{exp['must']}」"

    if kind == "stat":
        if not has_number(answer, exp["n"]):
            return False, f"缺少数字 {exp['n']}"
        if exp.get("must") and exp["must"] not in answer:
            return False, f"缺少「{exp['must']}」"
        return True, ""

    if kind == "coverage":
        miss = [l for l in exp["lectures"] if l not in answer]
        if miss:
            return False, f"漏讲次 {miss}"
        if not has_number(answer, exp["count"]):
            return False, f"缺少数量 {exp['count']}"
        extra = re.findall(r"第[一二三四五六七八九]讲", answer)
        bad = {x for x in extra if x not in exp["lectures"]}
        if bad:
            return False, f"多讲了 {sorted(bad)}"
        return True, ""

    if kind == "claim":
        miss = [x for x in exp["must_all"] if x not in answer]
        if miss:
            return False, f"漏了 {miss}"
        if not any(w in answer for w in exp["any_of"]):
            return False, f"没有明确断言（缺少 {'/'.join(exp['any_of'])} 之类）"
        return True, ""

    if kind == "set":
        miss = [x for x in exp["must_all"] if x not in answer]
        if miss:
            return False, f"漏了 {miss}"
        forbid = [x for x in exp.get("forbid", []) if x in answer]
        if forbid:
            return False, f"多说了 {forbid}"
        return True, ""

    return False, f"未知类型 {kind}"


def selftest() -> list[str]:
    """先验证判分器本身，否则"10/10"可能只是判分太宽。

    这套自检是必须的：本轮第一次跑出 10/10，其实是 9/10——
    第 9 条模型答"地心说（未接受哥白尼日心说）"，而被判通过，
    因为标准答案被我写成了"哥白尼"，判分又只做了 `in` 判断。
    断言写松了，它会替错误的标准答案打掩护。
    """
    cases = [
        ("refuse", {}, "题库里没有找到相关内容", True),
        ("refuse", {}, "红烧肉要先焯水再炖两小时", False),
        ("qa", {"must": "地心说"}, "第谷接受的是地心说。", True),
        ("qa", {"must": "地心说"}, "第谷并未接受地心说，他接受的是日心说。", False),
        ("qa", {"must": "哥白尼"}, "他没有接受哥白尼的学说。", False),
        ("stat", {"n": 18}, "第三讲共有 18 道判断题。", True),
        ("stat", {"n": 18}, "第三讲共有 18 道判断题。", True),
        ("stat", {"n": 18}, "第三讲共有 180 道判断题。", False),
        ("stat", {"n": 72, "must": "第六讲"}, "第六讲最多，共 72 道。", True),
        ("stat", {"n": 72, "must": "第六讲"}, "第五讲最多，共 72 道。", False),
        ("coverage", {"lectures": ["第二讲", "第四讲"], "count": 3},
         "共 3 道：第二讲 1 道，第四讲 2 道。", True),
        ("coverage", {"lectures": ["第二讲", "第四讲"], "count": 3},
         "共 3 道，都在第二讲。", False),
        ("coverage", {"lectures": ["第二讲", "第四讲"], "count": 3},
         "共 3 道：第二讲、第四讲、第五讲各一道。", False),
        ("set", {"must_all": ["第四讲"], "forbid": ["第二讲"]},
         "只有第四讲同时提到，第二讲只有盖伦。", False),
        ("set", {"must_all": ["第四讲"], "forbid": ["第二讲"]},
         "只有第四讲同时提到这两个人。", True),
        ("set", {"must_all": ["第一讲", "第二讲", "第八讲"]},
         "是第八讲和第二讲，第一讲也是。", True),
        ("set", {"must_all": ["第一讲", "第二讲", "第八讲"]},
         "是第八讲和第二讲。", False),
        # 正确答案里会顺带提到别的讲次（解释为什么是它），此时用 forbid 会误杀
        ("claim", {"must_all": ["第四讲"], "any_of": ["唯一", "重合"]},
         "盖伦在第二讲和第四讲，牛顿在第四、五、六讲，两者唯一重合的是第四讲。", True),
        ("claim", {"must_all": ["第四讲"], "any_of": ["唯一", "重合"]},
         "盖伦在第二讲和第四讲，牛顿在第五讲。", False),
    ]
    bad = []
    for kind, exp, answer, want in cases:
        ok, _why = judge(kind, exp, answer)
        if ok != want:
            bad.append(f"[{kind}] 「{answer}」期望 {'通过' if want else '失败'}，实际 {'通过' if ok else '失败'}")
    return bad


def main():
    from agent import KB, run

    only = None
    if "-k" in sys.argv:
        only = sys.argv[sys.argv.index("-k") + 1]

    items = [it for it in ITEMS if not only or only in it[0]]
    st = selftest()
    if st:
        print("判分器自检失败，先修 judge()：")
        for s in st:
            print("  -", s)
        return 1
    print("判分器自检通过（19 个已知对错的样本全部判对）")
    scan = raw_scan()
    bad = check_ground_truth(scan)
    if bad:
        print("标准答案校验失败，先修 ITEMS：")
        for b in bad:
            print("  -", b)
        return 1
    print(f"标准答案校验通过（{len(ITEMS)} 条全部与独立扫描一致）")
    print("加载索引 ...")
    kb = KB()
    print(f"就绪：{len(kb.meta)} 道题 | 本次跑 {len(items)} 条")
    print("=" * 74)

    rows, passed = [], 0
    for _id, q, kind, exp in items:
        r = run(kb, q, verbose=False)
        ok, why = judge(kind, exp, r["answer"])
        passed += ok
        tools = [s["action"] for s in r["steps"] if s["action"] != "final"]
        rows.append((_id, q, kind, ok, why, tools, r["elapsed"], r["answer"]))
        flag = "通过" if ok else "失败"
        print(f"[{flag}] {_id:<6} {q}")
        if not ok:
            print(f"         原因：{why}")
        print(f"         工具：{tools} · {r['elapsed']}s · {r['stop']}")
        print(f"         答：{r['answer'][:88]}")
        print()

    print("=" * 74)
    print(f"通过 {passed}/{len(items)}  =  {passed / len(items):.0%}")
    by_kind = {}
    for _id, _q, kind, ok, *_ in rows:
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
    for kind, (a, b) in by_kind.items():
        print(f"  {kind:<9} {a}/{b}")
    wrong = [r for r in rows if not r[3]]
    if wrong:
        print("\n失败明细：")
        for _id, q, kind, _ok, why, tools, _t, ans in wrong:
            print(f"  {_id} {q} → {why} (工具 {tools})")

    out = DATA / "agent_eval_result.json"
    out.write_text(json.dumps(
        [{"id": a, "q": b, "kind": c, "pass": d, "why": e,
          "tools": f, "elapsed": g, "answer": h}
         for a, b, c, d, e, f, g, h in rows],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
