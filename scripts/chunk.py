"""
把长文档切成检索用的块

核心问题只有一个：**切多大、切在哪。**

回答这个问题的指标有两个层次：

  1. 结构指标（不用跑模型，立刻能算）
       - 块数、平均块字数
       - **被切断的题数**：一道题的区间跨了两块 → 检索到它时答案不完整
       - 空块数：切出来全是选项、没有实质内容

  2. 检索指标（要跑 embedding，慢）
       - hit@k：150 条用例里，含正确答案的块排进前 k 的比例

  先用第 1 层选候选参数，再用第 2 层定夺。不要一上来就 embedding 跑 9 组，
  那是浪费——大部分参数组合在结构指标上就能被排除。

用法：
    python chunk.py                      # 默认 300 / 50
    python chunk.py --size 500 --overlap 0
    python chunk.py --sweep              # 一次跑完参数网格，出对照表
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "data" / "docs"
OUT = ROOT / "data" / "chunks.json"


def load_docs() -> dict:
    """读文档和每道题的字符区间。"""
    return json.loads((DOCS / "_index.json").read_text(encoding="utf-8"))


# 句末标点。换行也算——教材里一个自然段结束时会有换行
SENT_END = "。！？；\n»”」』）"

# 块首要跳过的字符：脚注圈号 + 收尾引号/括号。
# 正文里常见「…加以扬弃。②，“企图用…」，对齐到句号后会留下孤零零的「②，」，
# 引用显示出来既不像一句话的开头，也容易被误当成正文。
LEADING_SKIP = "①②③④⑤⑥⑦⑧⑨⑩»”」』）。，、；：\n "


def chunk_text(text: str, size: int, overlap: int,
               snap: bool = True, tolerance: float = 0.3) -> list[tuple[int, int]]:
    """按字符数切块，返回 [(start, end), ...]。

    overlap 是相邻两块的重叠字符数——怕把一句切断，所以让后一块回退一点，
    重复上一块的结尾。

    关于句子边界（snap）：
      纯粹按字数切，一定会切在句子中间。实测后果很具体——
      「辩证唯物主义自然观的定义」那句被切成两半，前半块和后半块都不像定义，
      重排模型给这两块都打了低分（5.12 / 5.15），而给一段通顺的无关段落打了 7.01。
      所以切到目标位置附近时，往后找一个句末标点再断。

    tolerance 是允许偏离多少：块长 500、tolerance=0.3，
    就允许在 350~650 之间找句号。太大块长会乱跳，太小常常找不到句号只能硬切。
    """
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size，否则永远前进不了")

    spans = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)

        if snap and end < n:
            # 优先往后找：宁可块长一点，也要在句号后断开
            lo = max(start + 1, int(end - size * tolerance))
            hi = min(n, int(end + size * tolerance))
            cut = None
            for i in range(end, hi):
                if text[i] in SENT_END:
                    cut = i + 1
                    break
            if cut is None:                      # 后面没有，就往回找
                for i in range(end, lo, -1):
                    if text[i - 1] in SENT_END:
                        cut = i
                        break
            if cut:
                end = cut

        spans.append((start, end))
        if end >= n:
            break

        # 下一块的起点：先回退 overlap，再往后对齐到句号之后。
        #
        # 为什么要对齐起点：之前只对齐了结尾，起点是「结尾往前 100 字」，
        # 必然落在句子中间——每块开头都是半句话。而块开头恰恰是用户
        # 在引用里看到的第一眼（截图里那种「发展模式及动力 学推向前进“①」）。
        # 对齐之后 overlap 的实际长度会略小于设定值（60~100），可以接受。
        nxt = max(0, end - overlap)
        if snap and 0 < nxt < n:
            hi = min(n, nxt + max(20, int(size * tolerance)))
            for i in range(nxt, hi):
                if text[i - 1] in SENT_END:
                    nxt = i
                    break
            # 跳过紧跟句号的脚注标记和收尾引号。
            # 正文里常见「…加以扬弃。②，“企图用…」，对齐到句号后
            # 会留下孤零零的「②，」，引用显示出来很难看，也不像一句话的开头。
            while nxt < n and text[nxt] in LEADING_SKIP:
                nxt += 1
        start = nxt
    return spans


def qids_in(start: int, end: int, spans: list[dict]) -> list[str]:
    """哪些题落在了 [start, end) 里（有交集就算）。"""
    return [s["qid"] for s in spans if s["end"] > start and s["start"] < end]


def cut_questions(start: int, end: int, spans: list[dict]) -> list[str]:
    """哪些题被块的边界切断了（题的区间跨越了块边界）。"""
    return [s["qid"] for s in spans if s["start"] < end and s["end"] > start
            and not (s["start"] >= start and s["end"] <= end)]


def build(size: int, overlap: int, docs: dict | None = None,
          verbose: bool = True) -> tuple[list[dict], dict]:
    """切块并统计结构指标。"""
    docs = docs or load_docs()
    chunks = []
    stats = {"blocks": 0, "empty": 0, "chars": 0}
    cut_seen: set[str] = set()      # 至少被边界碰到过的题
    whole_seen: set[str] = set()    # 至少有一块完整包含它的题

    for name, d in docs.items():
        text = (DOCS / d["file"]).read_text(encoding="utf-8")
        for i, (s, e) in enumerate(chunk_text(text, size, overlap)):
            body = text[s:e]
            qs = qids_in(s, e, d["spans"])
            cuts = cut_questions(s, e, d["spans"])

            # "空块"的定义：一个字都没进题库区间的块（通常是纯格式残留）
            chunks.append({
                "cid": f"{name}#{i}",
                "doc": name,
                "lecture": d["lecture"],
                "start": s,
                "end": e,
                "chars": len(body),
                "qids": qs,
                "text": body,
            })
            stats["blocks"] += 1
            cut_seen.update(cuts)
            # 完整包含：块的区间把这道题的区间整个盖住
            for s2 in d["spans"]:
                if s2["start"] >= s and s2["end"] <= e:
                    whole_seen.add(s2["qid"])
            if not qs:
                stats["empty"] += 1
            stats["chars"] += len(body)

    all_q = {s["qid"] for d in docs.values() for s in d["spans"]}
    stats["cut"] = len(cut_seen)
    stats["whole"] = len(whole_seen)
    stats["lost"] = len(all_q - whole_seen)     # 没有任何一块完整包含它 → 检索到了也读不全
    stats["avg_chars"] = round(stats["chars"] / max(1, stats["blocks"]), 1)
    if verbose:
        print(f"  size={size:<4} overlap={overlap:<4} → "
              f"{stats['blocks']:>3} 块 · 平均 {stats['avg_chars']:>5} 字 · "
              f"完整 {stats['whole']:>3} · 切断 {stats['cut']:>3} · "
              f"**丢了 {stats['lost']:>3}** · 空块 {stats['empty']}")
    return chunks, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--sweep", action="store_true", help="跑参数网格出对照表")
    args = ap.parse_args()

    docs = load_docs()
    print(f"{len(docs)} 篇文档，{sum(v['chars'] for v in docs.values())} 字\n")

    if args.sweep:
        print("参数网格（只看结构指标，还没跑检索）：")
        rows = []
        for size in (200, 300, 400, 500, 800):
            for ov in (0, 50, 100):
                if ov >= size:
                    continue
                _, st = build(size, ov, docs)
                rows.append((size, ov, st))
    else:
        chunks, st = build(args.size, args.overlap, docs)
        OUT.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写 {OUT.name}（{len(chunks)} 块）")
        print(f"  其中含题的块 {sum(1 for c in chunks if c['qids'])} 个，"
              f"不含题的 {st['empty']} 个")


if __name__ == "__main__":
    main()
