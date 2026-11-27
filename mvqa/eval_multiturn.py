"""
多轮评测：10 组追问，判"用上上下文"和"不因上下文而编造"

为什么单独测：
  刚给问答层加了轻量多轮（前端带最近 3 轮）。不测的话，"加了功能但没验证"
  和之前"改缩进没跑评测"是同一类风险。

  两件事要分开判：
    用上上下文   「比刚才多了吗」必须接住上一轮的数字
    不编造       有了历史之后，模型更容易把"历史里没说的"当成"说过"——
                 这条更危险，因为回答会变得自信而错误

每组的判定方式不同，写在用例里：
  number   回答里必须出现上一轮的那个数字
  insist   必须仍调用 count_in，且类别与上一轮一致（继承筛选条件）
  arith    必须算出上一轮数字 + 增量
  deny     必须否认历史里有颜色信息（"没说过颜色"这类）
  nocolor  必须明说看不出颜色，不能给出颜色相关数量

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_multiturn.py
"""

import re

import mvqa

IMG = "0000280_03401_d_0000629"      # 车 32 确认、左侧 1、右侧 10

# (首轮问题, 追问, 判定方式, 说明)
CASES = [
    ("图里有几辆车", "比刚才多了还是少了", "number", "接住数字"),
    ("图里有几辆车", "刚才那个数是多少", "number", "回溯数字"),
    ("图里有几辆车", "那左边呢", "insist", "继承类别→左侧车"),
    ("图里有几辆摩托车", "那右边呢", "insist", "继承类别→右侧摩托"),
    ("图里有多少人", "那右边有多少人", "insist", "继承类别→右侧人"),
    ("图里有几辆车", "再多 5 辆是多少", "arith", "做算术"),
    ("图里有几辆车", "那红色的车呢", "nocolor", "颜色：必须说看不出"),
    ("图里有几辆车", "刚才说过有红色的车吗", "deny", "不许把没说的当成说过"),
    ("图里都有什么", "那红色的车有几辆", "nocolor", "全景后问颜色"),
    ("图里有多少人", "比刚才的车多还是少", "number", "跨轮比较（该说实话）"),
]


def numbers_in(t: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", t)]


def judge(kind: str, prior_num: int, prior_cls: str, calls: list, ans: str):
    """返回 (通过, 原因)。"""
    if kind == "number":
        # 上一轮的数字必须出现在回答里（否则就是没接住上下文）
        used = prior_num in numbers_in(ans)
        return used, "" if used else f"没接住上一轮的数 {prior_num}"

    if kind == "insist":
        zone_calls = [c for c in calls if c["tool"] == "count_in"]
        if not zone_calls:
            return False, "没调 count_in（没继承'那左边/右边'的区域语义）"
        cls_of = mvqa.ALIAS.get(str(zone_calls[-1]["args"].get("cls", "")).strip(),
                                str(zone_calls[-1]["args"].get("cls", "")))
        ok = cls_of == prior_cls
        return ok, "" if ok else f"类别没继承：问 {prior_cls}，实际数了 {cls_of}"

    if kind == "arith":
        ok = (prior_num + 5) in numbers_in(ans)
        return ok, "" if ok else f"算术没做对（应为 {prior_num + 5}）"

    if kind == "deny":
        # 历史里从没提过颜色，必须否认
        deny = any(w in ans for w in ("没说过", "没有说过", "没提", "没有提", "未提"))
        fake = re.search(r"(红色|白色|黑色|蓝色)的?车?\s*(有|是)\s*\d", ans) is not None
        return deny and not fake, ("" if deny and not fake else
                                   "把历史里没说的颜色当成说过了")

    if kind == "nocolor":
        ok = any(w in ans for w in ("看不出", "看不出颜色", "不认颜色", "没有颜色",
                                    "颜色信息", "看不了颜色"))
        fake = re.search(r"(红色|白色|黑色|蓝色)[^0-9]{0,6}\d", ans) is not None
        return ok and not fake, ("" if ok and not fake else "没说清看不出颜色，或给了颜色数量")

    return False, f"未知判定 {kind}"


def main():
    print(f"{'判定':<10}{'首轮 → 追问':<38}结果")
    print("-" * 78)
    rows = []
    for first, follow, kind, note in CASES:
        r1 = mvqa.run(first, IMG, verbose=False)
        prior_nums = numbers_in(r1["answer"])
        # 取首轮答案里最大的数当作"那个数"（工具返回的确认数量通常是最大的）
        prior_num = max(prior_nums) if prior_nums else 0
        prior_cls = ""
        if r1["calls"]:
            c0 = str(r1["calls"][0]["args"].get("cls", "")).strip()
            prior_cls = mvqa.ALIAS.get(c0, c0)

        hist = [{"q": first, "a": r1["answer"]}]
        r2 = mvqa.run(follow, IMG, verbose=False, history=hist)

        ok, why = judge(kind, prior_num, prior_cls, r2["calls"], r2["answer"])
        print(f"{kind:<10}{first[:16] + ' → ' + follow[:16]:<38}{'✓' if ok else '✗'}")
        if not ok:
            print(f"  首轮答：{r1['answer'][:60]}")
            print(f"  追问答：{r2['answer'][:60]}")
            print(f"  ↳ {why}")
        rows.append((kind, ok))

    n = len(rows)
    print(f"\n总判定 {sum(1 for r in rows if r[1])}/{n}")
    by = {}
    for k, ok in rows:
        a = by.setdefault(k, [0, 0])
        a[0] += ok
        a[1] += 1
    for k, (ok, tot) in by.items():
        print(f"  {k:<10} {ok}/{tot}")


if __name__ == "__main__":
    main()
