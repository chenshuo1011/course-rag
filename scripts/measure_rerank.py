"""
量重排模型的速度，决定候选池能取多大

为什么先量再写：
  重排的成本是 O(候选数)——每一条候选都要单独跑一次模型。
  如果一条要 50ms，50 条就是 2.5 秒，整个方案就得推翻重设计。
  所以"池子开多大"不是拍脑袋定的，是算力定的。

模型：BAAI/bge-reranker-base
  交叉编码器，输入 (query, passage) 拼在一起，输出一个相关性分数。
  和 embedding 不同，它必须成对计算，没法提前把文档编码好。

用法：
    cd /d D:\course-rag\scripts
    "D:\anaconda3\envs\yolo\python.exe" measure_rerank.py
"""

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import chunk as chunkmod

MODEL = "BAAI/bge-reranker-base"
ROOT_TEXT = "../data/book/book.txt"


def main():
    text = open(ROOT_TEXT, encoding="utf-8").read()
    spans = chunkmod.chunk_text(text, 500, 100)
    passages = [text[s:e] for s, e in spans[:50]]
    query = "朴素唯物主义自然观的观点"

    print(f"加载 {MODEL} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    print(f"  加载耗时 {time.time()-t0:.1f}s，参数量 {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    def score_pairs(q, ps):
        """一次算一批 (query, passage) 的分数。"""
        with torch.no_grad():
            batch = tok([q] * len(ps), ps, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
            return model(**batch).logits.view(-1)

    # 预热：第一次会触发 lazy 初始化，不能算进平均
    score_pairs(query, passages[:1])

    print(f"\n{'条数':>6}{'总耗时':>10}{'每条':>9}")
    for n in (1, 5, 10, 20, 50):
        t0 = time.perf_counter()
        score_pairs(query, passages[:n])
        ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>6}{ms:>8.0f}ms{ms/n:>8.1f}ms")

    # 顺便看一眼分数有没有区分度（前 5 条的实际得分）
    with torch.no_grad():
        s = torch.sigmoid(score_pairs(query, passages[:5])).tolist()
    print(f"\n前 5 条的相关性分数：{[round(x, 3) for x in s]}")
    print("（如果 5 个分数几乎一样，说明这个查询在池子里本来就没有强相关，重排也没用）")


if __name__ == "__main__":
    main()
