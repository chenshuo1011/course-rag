"""
生成质量抽查：10 条，覆盖四类问题

为什么这么选：
  目的不是统计精度，是发现"灾难性错误"——比如硬编、答非所问、引用造假。
  所以刻意把两类"陷阱题"放进去：
    · 容易混的概念（朴素唯物主义 vs 机械唯物主义）
    · 教材里可能根本没有的（第谷用什么观测）→ 看它会不会硬编

输出两个东西：
  1. 屏幕上逐条打印（答案 + 引用 + 耗时）
  2. data/book/gen_check.md —— 一张待填的表

你填什么：每条判一档，加一句为什么
  ✅ 有据、正确、引用对
  ⚠️ 答案对，但引用不准（引错节 / 引的原文没直接支持）
  ❌ 编造 / 无依据 / 答非所问 / 该拒答却硬答

用法：
    cd /d D:\course-rag\scripts
    "D:\anaconda3\envs\yolo\python.exe" check_gen.py
"""

import json
import time
from pathlib import Path

from doc_ask import DocQA

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "book" / "gen_check.md"

# (问题, 类型)
QUESTIONS = [
    ("什么是朴素唯物主义自然观", "单点概念"),
    ("什么是辩证唯物主义自然观", "单点概念"),
    ("系统自然观的核心观点是什么", "单点概念"),
    ("机械唯物主义自然观和辩证唯物主义自然观有什么区别", "需要综合"),
    ("马克思主义自然观是怎么形成的", "需要综合"),
    ("科学技术观主要包括哪些内容", "需要综合"),
    ("系统自然观和生态自然观有什么不同", "容易混淆"),
    ("朴素唯物主义和机械唯物主义的主要差别", "容易混淆"),
    ("第谷用什么工具观测天体", "教材里可能没有"),
    ("红烧肉怎么做", "完全无关"),
]


def main():
    qa = DocQA()

    # 预热：第一次调用包含各种初始化，会把耗时算歪（实测首条 4383ms，之后约 190ms/条）
    print("[预热] ...", flush=True)
    qa.ask("什么是自然辩证法")

    rows = []
    for i, (q, kind) in enumerate(QUESTIONS, 1):
        print(f"\n{'='*66}")
        print(f"{i}. [{kind}] {q}")
        t0 = time.time()
        try:
            r = qa.ask(q)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}")
            rows.append((i, q, kind, "（调用失败）", "—", "", ""))
            continue
        print(f"\n答案：{r['answer']}")
        print("\n引用：")
        for c in r["citations"]:
            print(f"  ［片段{c['n']}］{c['section']}  score={c['score']}")
            print(f"       {c['snippet'][:50]}…")
        t = r["timings"]
        print(f"\n耗时：检索 {t['retrieve_ms']}ms · 重排 {t['rerank_ms']}ms"
              f" · 生成 {t['gen_ms']}ms · 合计 {time.time()-t0:.1f}s")

        rows.append((i, q, kind, r["answer"][:60].replace("\n", " "),
                     r["citations"][0]["section"] if r["citations"] else "—",
                     r["total_ms"], ""))

    # 待填的表
    lines = [
        "# 教材问答 · 生成质量抽查",
        "",
        "填写规则：",
        "",
        "- ✅ 有据、正确、引用对",
        "- ⚠️ 答案对，但引用不准（引错节 / 引的原文没直接支持）",
        "- ❌ 编造 / 无依据 / 答非所问 / 该拒答却硬答",
        "",
        "判断标准是**「能不能在引用的原文里找到依据」**，不是「我觉得对不对」。",
        "",
        "| # | 类型 | 问题 | 答案摘要 | 引用首节 | 判断 | 为什么 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, q, kind, ans, sec, ms, mark in rows:
        lines.append(f"| {i} | {kind} | {q} | {ans} | {sec} | {mark} | |")
    lines += ["", "## 统计", "", "- 有据率（✅ / 10）：", "- 编造率（❌ / 10）：", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'='*66}")
    print(f"待填的表已写入 {OUT}")
    print("打开它，逐条填「判断」和「为什么」两列。")


if __name__ == "__main__":
    main()
