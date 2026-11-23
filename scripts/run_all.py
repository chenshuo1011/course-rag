"""
一条命令重建整条链路

为什么需要它：
  31 个脚本，顺序还不自由——比如清洗改了 book.txt 之后，
  评测集的字符偏移全部失效，必须先重建 evalset 才能评测。
  这种隐藏依赖只存在脑子里，换个人（或者三个月后的自己）就会踩。

用法：
    python run_all.py            # 跳过耗时的重排评测（约 5 分钟）
    python run_all.py --full     # 全跑，含重排评测
    python run_all.py --ocr      # 连 OCR 一起跑（约 10 分钟，需要桌面有 PDF）

每一步都会打印耗时和产物路径。中途失败会停下并告诉你是哪一步。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
PY = sys.executable


def run_step(name, args, est=""):
    print()
    print("=" * 68)
    print(f"▶ {name}" + (f"  （约 {est}）" if est else ""))
    print("=" * 68)
    t0 = time.time()
    r = subprocess.run([PY, *args], cwd=SCRIPTS, capture_output=False)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n✗ {name} 失败（退出码 {r.returncode}），后面的步骤没有执行。")
        sys.exit(r.returncode)
    print(f"✓ {name} 完成，{dt:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="连重排评测一起跑")
    ap.add_argument("--ocr", action="store_true", help="连 OCR 一起跑（10 分钟）")
    args = ap.parse_args()

    print("=" * 68)
    print("course-rag 全链路重建")
    print(f"Python: {PY}")
    print("=" * 68)

    # ── 第 0 步：OCR（可选）──
    raw = ROOT / "data" / "book" / "raw_pages.jsonl"
    if args.ocr:
        run_step("OCR 扫描版 PDF（230 页）", ["ocr_book.py"], "10 分钟")
    elif not raw.exists():
        print("\n⚠ 没有 data/book/raw_pages.jsonl，教材链路会跳过。")
        print("  要重建的话加 --ocr（需要桌面上有那份 PDF）。")

    # ── 第 1 步：题库 → 讲次级长文 ──
    run_step("题库拼成长文档", ["build_docs.py"], "10 秒")

    # ── 第 2 步：教材清洗 ──
    if raw.exists():
        run_step("清洗 OCR 文本（去页眉、脚注、圈号）", ["clean_book.py"], "10 秒")
    else:
        print("\n跳过清洗：没有 OCR 原始数据。")

    # ── 第 3 步：切片 ──
    run_step("题库切片（500/100，句子边界对齐）",
             ["chunk.py", "--size", "500", "--overlap", "100"], "20 秒")

    # ── 第 4 步：教材评测集 ──
    # 这一步必须在清洗之后：evalset 存的是字符偏移，
    # book.txt 一变，偏移全部失效，指标会假崩（实测从 37% 掉到 11%）。
    if raw.exists():
        run_step("重建教材评测集（依赖上一步的 book.txt）",
                 ["build_evalset.py"], "5 秒")

        # ── 第 5 步：检索评测 ──
        run_step("教材检索评测（6 组切片参数）", ["eval_book.py"], "3 分钟")

        # ── 第 6 步：重排评测（可选）──
        if args.full:
            run_step("重排评测（144 条 × 10 候选）", ["eval_rerank.py"], "5 分钟")
        else:
            print("\n跳过重排评测（加 --full 才跑，约 5 分钟）。")

    # ── 产物清单 ──
    print()
    print("=" * 68)
    print("产物")
    print("=" * 68)
    outputs = [
        ("data/docs/_index.json", "题库讲次级长文 + 每题的字符区间"),
        ("data/chunks.json", "题库切片结果"),
        ("data/book/raw_pages.jsonl", "OCR 原始页（断点续跑用）"),
        ("data/book/book.txt", "清洗后的教材正文"),
        ("data/book/evalset.json", "教材评测集（144 条）"),
        ("data/book/eval_chunk.json", "切片参数评测结果"),
        ("data/book/eval_rerank.json", "重排前后对比"),
        ("data/book/locate.json", "（已废弃）题库解析在书里的定位尝试"),
    ]
    for rel, desc in outputs:
        p = ROOT / rel
        size = f"{p.stat().st_size/1024:.0f}KB" if p.exists() else "—"
        mark = "✓" if p.exists() else "·"
        print(f"  {mark} {rel:<34} {size:>7}  {desc}")

    print()
    print("下一步：")
    print("  run_server.bat   启动四个页面（/ 答题 · /review 复习 · /chat 问答 · /doc 教材）")


if __name__ == "__main__":
    main()
