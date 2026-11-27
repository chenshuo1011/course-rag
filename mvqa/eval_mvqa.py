"""
问答层功能评测：20 张图 × 4 类问题 = 80 条

为什么从 4 张扩到 20 张：
  原来 20 条问题挤在 4 张图上，而且那 4 张是我手动挑的"好看"的图——
  结论只能说明机制成立，不能代表整体。现在按目标密度分层抽 20 张
  （见 eval_common.py），样本平均 70 个目标/张，和全集的 71 一致。

每题判四件事：
  工具选择   该调 count 的没去调 locate，该拒答的没硬答
  参数正确   问「公交」传的是 bus，不是"所有车"
  数字忠实   答案里的数字必须出现在工具返回里，不许自己编
  措辞合规   低召回说"至少"（仅统计明确数字、已知可靠度的；详情见 eval_phrasing.py）

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_mvqa.py
"""

import re

from eval_common import pick_images
from mvqa import ALIAS, RELIABILITY, run

# 题型：(问题, 期望工具, 期望类别或 None, 备注)
TEMPLATES = [
    ("这张图里有多少辆车", {"count"}, "car", "单类计数"),
    ("图里有多少人", {"count"}, "pedestrian", "合并计数"),
    ("这张图里都有什么", {"overview"}, None, "全景"),
    ("车主要停在哪一片", {"locate"}, "car", "方位分布"),
]
EXTRA = [
    ("左边那一片有几辆车", {"count_in"}, "car", "区域计数"),
    ("图里有船吗", {"count", "overview"}, None, "不在标签空间"),
]

HEDGE = ("约", "大约", "至少", "左右", "可能", "估计")


def numbers_in(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def build_cases(n_img: int = 20):
    picks = pick_images(n_img)
    cases = []
    for idx, (stem, note) in enumerate(picks):
        for q, tools, cls, kind in TEMPLATES:
            cases.append((stem, note, q, tools, cls, kind))
        # 区域和拒答只在一部分图上问，避免总条数爆炸
        if idx % 4 == 0:
            for q, tools, cls, kind in EXTRA:
                cases.append((stem, note, q, tools, cls, kind))
    return cases


def main():
    cases = build_cases(20)
    n_img = len(set(c[0] for c in cases))
    print(f"共 {len(cases)} 条，覆盖 {n_img} 张图\n")

    rows = []
    for i, (stem, note, q, want_tools, want_cls, kind) in enumerate(cases, 1):
        r = run(q, stem, verbose=False)
        used_tools = [c["tool"] for c in r["calls"]]
        used_cls = [str(c["args"].get("cls", "")) for c in r["calls"]]
        ans = r["answer"]

        ok_tool = bool(set(used_tools) & want_tools) if want_tools else not used_tools

        ok_cls = True
        if want_cls:
            resolved = [ALIAS.get(c.strip(), c.strip().lower()) for c in used_cls if c]
            ok_cls = want_cls in resolved

        tool_nums = []
        for s in r["steps"]:
            tool_nums += numbers_in(str(s["result"]))
        refuse = any(w in ans for w in ("看不出", "不在", "识别范围", "只认"))
        nums = numbers_in(ans)
        ok_num = True if refuse else (all(n in tool_nums for n in nums) if nums else True)

        ok_hedge = None
        if want_cls and want_cls in RELIABILITY and nums and not refuse:
            if RELIABILITY[want_cls][0] < 0.8:
                ok_hedge = any(h in ans for h in HEDGE)

        bad = not (ok_tool and ok_cls and ok_num and ok_hedge is not False)
        if bad:
            print(f"{i:>3}. ✗ [{kind}] {q} @ {stem[:22]}")
            print(f"     答：{ans[:70]}")
            print(f"     工具{used_tools} 参数{used_cls} "
                  f"判定 tool={ok_tool} cls={ok_cls} num={ok_num} hedge={ok_hedge}")
        rows.append((kind, ok_tool, ok_cls, ok_num, ok_hedge))

    n = len(rows)
    print(f"\n{'='*60}")
    print(f"汇总（{n} 条，{n_img} 张图）")
    print(f"  工具选择正确   {sum(1 for r in rows if r[1])}/{n}")
    print(f"  参数正确       {sum(1 for r in rows if r[2])}/{n}")
    print(f"  数字忠实       {sum(1 for r in rows if r[3])}/{n}")
    hedge = [r for r in rows if r[4] is not None]
    if hedge:
        print(f"  措辞合规       {sum(1 for r in hedge if r[4])}/{len(hedge)}"
              f"（只统计低召回且有明确数字的）")

    print("\n按题型：")
    kinds = {}
    for kind, *flags in rows:
        a = kinds.setdefault(kind, [0, 0])
        a[0] += all(f is not False for f in flags)
        a[1] += 1
    for k, (ok, tot) in kinds.items():
        print(f"  {k:<10} {ok}/{tot}")


if __name__ == "__main__":
    main()
