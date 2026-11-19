"""
教材问答：切片 → 检索 → 重排 → 生成 + 引用

这是把前四步串起来的最后一段。设计上有三个约束：

1. 只能根据片段回答
   模型凭训练数据也能说个大概，但那就不是 RAG 了。
   所以提示词明确要求"片段里没有就说没有"，并强制标引用编号。

2. 必须能给出处
   教材问答如果没有引用，用户没法核对。
   好在切片时存了字符区间，可以反查它落在哪一小节里。

3. 重排只在 CPU 上跑 pool=10
   实测每条约 150~190ms，10 条 1.9 秒。再多就不可接受了。
   （池子 10 的天花板是 87%，池子 20 是 94% 但要 3 秒，暂时不值得）

用法（命令行）：python doc_ask.py "什么是朴素唯物主义自然观"
"""

import json
import re
import sys
import time
from pathlib import Path

import chunk as chunkmod
import embedder
import llm
from vectorstore import build

ROOT = Path(__file__).parent.parent
BOOK = ROOT / "data" / "book" / "book.txt"

POOL = 10          # 粗排取多少条进重排
TOP_N = 3          # 最终给模型几条

SYSTEM = """你是课程助教，只能用【片段】里的内容回答。

规则：
1. 片段里没有的内容，直接说「教材里没有直接讲到这一点」，不要用你自己的知识补充。
2. 回答末尾标注用到的片段，格式：［片段1］［片段2］
3. 用一两句话直接回答，不要复述片段原文，不要客套。
4. 如果片段只部分回答了问题，说清哪部分有、哪部分没有。"""

# 全书的小节标题（用来把字符位置换成"第几节"）
_HEAD = re.compile(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)")


class DocQA:
    def __init__(self):
        self.text = BOOK.read_text(encoding="utf-8")
        self.spans = chunkmod.chunk_text(self.text, 500, 100)
        self.chunks = [self.text[s:e] for s, e in self.spans]
        t0 = time.time()
        self.index = build(embedder.embed(self.chunks).numpy(), "numpy")
        self._heads = self._load_headings()
        self._tok = self._model = None
        print(f"[DocQA] 索引就绪：{len(self.chunks)} 块，{time.time()-t0:.1f}s", file=sys.stderr)

    # ── 引用：字符位置 → 小节标题 ──
    def _load_headings(self):
        heads, cur = [], 0
        for line in self.text.splitlines():
            s = line.strip()
            if 4 <= len(s) <= 45 and _HEAD.match(s) and not re.search(r"[·⋯…]{2,}", s):
                heads.append((cur, s))
            cur += len(line) + 1
        return heads

    def section_of(self, pos: int) -> str:
        """某字符位置属于哪一小节（取它前面最近的标题）。"""
        name = "开头"
        for off, title in self._heads:
            if off <= pos:
                name = title
            else:
                break
        return name

    def section_range(self, start: int, end: int) -> str:
        """块横跨多节时，把起止两节都标出来。

        为什么不能只用起点：500 字的块经常跨节，起点在上一节末尾、
        正文大部分在下一节。只标起点会说"这段讲的是辩证自然观的作用"，
        而内容其实在讲系统自然观——用户核对时会以为引错了。
        """
        a, b = self.section_of(start), self.section_of(max(start, end - 1))
        return a if a == b else f"{a} → {b}"

    # ── 重排 ──
    def _load_reranker(self):
        if self._model is None:
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-base").eval()
            self._torch = torch
        return self._tok, self._model

    def _rerank(self, query, idxs):
        tok, model = self._load_reranker()
        ps = [self.chunks[i] for i in idxs]
        with self._torch.no_grad():
            batch = tok([query] * len(ps), ps, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
            scores = model(**batch).logits.view(-1).tolist()
        order = sorted(range(len(ps)), key=lambda i: -scores[i])
        return [idxs[i] for i in order], [scores[i] for i in order]

    # ── 主流程 ──
    def ask(self, query: str) -> dict:
        t = {}
        t0 = time.perf_counter()
        cand = self.index.search(embedder.embed_query(query).numpy(), POOL)[0]
        t["retrieve_ms"] = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        order, scores = self._rerank(query, cand)
        t["rerank_ms"] = round((time.perf_counter() - t0) * 1000)

        picked = order[:TOP_N]
        cites = []
        for n, (ci, sc) in enumerate(zip(picked, scores[:TOP_N]), 1):
            cites.append({
                "n": n,
                "section": self.section_range(*self.spans[ci]),
                "snippet": self.chunks[ci][:200],
                "score": round(sc, 3),
                "pos": self.spans[ci][0],
            })

        frag = "\n\n".join(
            f"【片段{c['n']}】（{c['section']}）\n{self.chunks[i]}"
            for c, i in zip(cites, picked))

        t0 = time.perf_counter()
        answer = llm.chat(SYSTEM, f"【片段】\n{frag}\n\n【问题】{query}", max_tokens=300)
        t["gen_ms"] = round((time.perf_counter() - t0) * 1000)

        return {"query": query, "answer": answer.strip(),
                "citations": cites, "timings": t,
                "total_ms": sum(t.values())}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "什么是朴素唯物主义自然观"
    qa = DocQA()
    r = qa.ask(q)
    print(f"\n问题：{r['query']}\n")
    print(r["answer"])
    print("\n引用：")
    for c in r["citations"]:
        print(f"  ［片段{c['n']}］{c['section']}  (score {c['score']})")
        print(f"        {c['snippet'][:60]}…")
    print(f"\n耗时：检索 {r['timings']['retrieve_ms']}ms · 重排 {r['timings']['rerank_ms']}ms"
          f" · 生成 {r['timings']['gen_ms']}ms · 合计 {r['total_ms']/1000:.1f}s")
