"""
把扫描版 PDF 逐页 OCR 成文字

为什么要单独存"原始 OCR 结果"：
  清洗规则一定会调好几轮（页眉怎么去、断行怎么并）。
  如果清洗写在 OCR 循环里，每改一次都要重新 OCR 一遍 230 页，一轮十几分钟。
  所以这里只做 OCR，一页一行存 jsonl；清洗是另一个脚本的事。

用法：
    python ocr_book.py                # 从头跑到尾，已做的页自动跳过
    python ocr_book.py --start 10 --end 20
"""

import argparse
import json
import time
from pathlib import Path

import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).parent.parent
PDF = Path(r"C:\Users\86155\Desktop\自然辩证法概论2025版pdf电子书.pdf")
OUT = ROOT / "data" / "book" / "raw_pages.jsonl"
SCALE = 2.0        # 渲染倍率：1.0 时小字识别错得多，2.0 明显更准


def load_done() -> set[int]:
    """已经 OCR 过的页码，用来断点续跑。"""
    if not OUT.exists():
        return set()
    done = set()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["page"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="起始页（从 0 数）")
    ap.add_argument("--end", type=int, default=-1, help="结束页（不含），-1 表示到最后一页")
    args = ap.parse_args()

    pdf = pdfium.PdfDocument(str(PDF))
    n = len(pdf)
    end = n if args.end < 0 else min(args.end, n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    todo = [i for i in range(args.start, end) if i not in done]
    print(f"共 {n} 页，本次要处理 {len(todo)} 页（已完成 {len(done)} 页）")
    if not todo:
        print("没有要处理的页。")
        return

    ocr = RapidOCR()
    t_all = time.time()

    with open(OUT, "a", encoding="utf-8") as f:
        for k, pno in enumerate(todo, 1):
            t0 = time.time()
            try:
                img = pdf[pno].render(scale=SCALE).to_numpy()
                res, _ = ocr(img)
                text = "\n".join(x[1] for x in res) if res else ""
            except Exception as e:
                text = ""
                print(f"  第 {pno+1} 页失败：{e}")

            f.write(json.dumps({"page": pno, "text": text}, ensure_ascii=False) + "\n")
            f.flush()

            if k % 10 == 0 or k == len(todo):
                used = time.time() - t_all
                eta = used / k * (len(todo) - k)
                print(f"  {k}/{len(todo)} 页 · {time.time()-t0:.1f}s/页 · "
                      f"已用 {used/60:.1f} 分 · 预计还需 {eta/60:.1f} 分", flush=True)

    print(f"\n完成，共 {len(todo)} 页，用时 {(time.time()-t_all)/60:.1f} 分")


if __name__ == "__main__":
    main()
