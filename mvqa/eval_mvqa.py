"""
问答层评测：20 条，四项自动判定

为什么这么设计：
  问答层的能力不在"检测准不准"（那已经量过了），而在于：
    · 会不会选对工具（问数量别去调 locate）
    · 会不会传对参数（问公交别数成所有车）
    · 会不会忠实转述（数字必须来自工具，不能自己编）
    · 会不会按可靠度措辞（低召回的类别说"至少"，不能说确数）

  前两项自动判、第三项自动判、第四项是这次的重点——它是"系统知道自己有多准"的体现。

题型分布（20 条）：
  数量 6 / 全景 2 / 方位 2 / 组合 2 / 措辞 2 / 该拒答 2 / 零目标 2 / 近义类别 2

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_mvqa.py
"""

import re
from pathlib import Path

from mvqa import ALIAS, BASE, RELIABILITY, run

# 用 val 集里真实存在的 4 张（之前凭印象写的 4 个名字有 3 个不存在，
# 结果 20 条里 10 条败在"找不到这张图"——评测选样本前必须先确认文件在）
IMGS = ["0000001_02999_d_0000005", "0000242_00001_d_0000001",
        "0000280_03401_d_0000629", "0000327_02801_d_0000725"]

# (问题, 图片序号, 期望工具, 期望类别关键字, 备注)
CASES = [
    ("这张图里有多少辆车", 0, {"count"}, "car", "数量"),
    ("图里有多少行人", 0, {"count"}, "pedestrian", "数量"),
    ("图里有几辆公交车", 0, {"count"}, "bus", "数量+近义"),
    ("有多少辆卡车", 1, {"count"}, "truck", "数量"),
    ("摩托车有几辆", 2, {"count"}, "motor", "数量"),
    ("三轮车多少", 3, {"count"}, "tricycle", "数量"),
    ("这张图里都有什么", 0, {"overview"}, None, "全景"),
    ("整体上都有哪些目标", 1, {"overview"}, None, "全景"),
    ("车主要停在哪个方位", 0, {"locate"}, "car", "方位"),
    ("人都集中在哪一片", 1, {"locate"}, "pedestrian", "方位"),
    ("车和摩托车各有多少", 2, {"count"}, None, "组合"),
    ("行人和摩托车分别多少", 3, {"count"}, None, "组合"),
    ("图里有多少人", 1, {"count"}, "pedestrian", "措辞"),
    ("有多少行人", 2, {"count"}, "pedestrian", "措辞"),
    ("今天天气怎么样", 0, set(), None, "该拒答"),
    ("这张图能卖多少钱", 1, set(), None, "该拒答"),
    ("图里有帆船吗", 2, {"count", "overview"}, None, "零目标"),
    ("有火车吗", 3, {"count", "overview"}, None, "零目标"),
    ("面包车有几辆", 0, {"count"}, "van", "近义类别"),
    ("自行车有多少", 1, {"count"}, "bicycle", "近义类别"),
]

HEDGE = ("约", "大约", "至少", "左右", "可能", "估计")


def numbers_in(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def main():
    rows = []
    for i, (q, idx, want_tools, want_cls, kind) in enumerate(CASES, 1):
        img = IMGS[idx]
        print(f"\n{'='*64}\n{i}. [{kind}] {q}")
        r = run(q, img, verbose=True)
        used_tools = [c["tool"] for c in r["calls"]]
        used_cls = [str(c["args"].get("cls", "")) for c in r["calls"]]
        ans = r["answer"]
        nums = numbers_in(ans)

        # 判定 1：工具选择
        ok_tool = bool(set(used_tools) & want_tools) if want_tools else not used_tools

        # 判定 2：参数（类别关键字是否出现在传参里）
        ok_cls = True
        if want_cls:
            # 模型传的是中文（"面包车"），要先用 ALIAS 映射成英文类别再比，
            # 否则永远比不上——第一版就是这么错的，20 条里判错 16 条
            resolved = [ALIAS.get(c.strip(), c.strip().lower()) for c in used_cls if c]
            ok_cls = want_cls in resolved

        # 判定 3：数字忠实 —— 答案里的数必须都出现在工具返回里
        # 两个例外：一是拒答题（答案里的"10 类"来自提示词不是编的），
        # 二是答案里明说"看不出/不在识别范围"的（本来就不该有数字）
        tool_nums = []
        for s in r["steps"]:
            tool_nums += numbers_in(str(s["result"]))
        refuse_phrasing = any(w in ans for w in ("看不出", "不在", "识别范围", "只认"))
        ok_num = True if refuse_phrasing else (all(n in tool_nums for n in nums)
                                              if nums else True)

        # 判定 4：措辞 —— 只判"低检出率的类别有没有加模糊词"这一侧。
        # 原来还要求"可靠类别禁止模糊词"，但 car 检出 113%（偏多）时
        # 说"实际可能略少"恰恰是诚实的，禁掉它等于奖励说假话。
        ok_hedge = None
        if want_cls and want_cls in RELIABILITY and nums and not refuse_phrasing:
            ratio = RELIABILITY[want_cls][0]
            if ratio < 0.8:
                ok_hedge = any(h in ans for h in HEDGE)

        print(f"  答：{ans}")
        print(f"  工具 {used_tools} {'✓' if ok_tool else '✗'}  "
              f"参数 {used_cls} {'✓' if ok_cls else '✗'}  "
              f"数字 {'✓' if ok_num else '✗'}  "
              f"措辞 {'✓' if ok_hedge else ('✗' if ok_hedge is False else '—')}")
        rows.append((i, kind, q, ans, ok_tool, ok_cls, ok_num, ok_hedge))

    n = len(rows)
    print(f"\n{'='*64}\n汇总（{n} 条）")
    print(f"  工具选择正确   {sum(1 for r in rows if r[4])}/{n}")
    print(f"  参数正确       {sum(1 for r in rows if r[5])}/{n}")
    print(f"  数字忠实       {sum(1 for r in rows if r[6])}/{n}")
    hedge = [r for r in rows if r[7] is not None]
    if hedge:
        print(f"  措辞合规       {sum(1 for r in hedge if r[7])}/{len(hedge)}"
              f"（只统计有明确数字且已知可靠度的）")
    bad = [(r[0], r[1], r[2], r[3]) for r in rows
           if not (r[4] and r[5] and r[6] and r[7] is not False)]
    if bad:
        print("\n  有问题的：")
        for b in bad:
            print(f"    {b[0]}. [{b[1]}] {b[2]} → {b[3][:50]}")


if __name__ == "__main__":
    main()
