"""
BM25 检索（自己实现，不依赖 rank_bm25）

为什么要有它：
  向量检索把每个字都揉成 512 个数字，专有名词（第谷、林奈、赫胥黎）的
  区分度会被"平均"掉。BM25 反过来——它只认词有没有出现，稀有词权重高。
  两者错在不同的地方，所以能互补。

公式（Okapi BM25）：
  score(q,d) = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))

  idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
  tf     = 词 t 在文档 d 里出现几次
  |d|    = 文档长度，avgdl 平均文档长度
  k1     = 词频饱和参数（词出现 10 次不会比 3 次强 3 倍）
  b      = 长度惩罚强度（长文档天然容易命中，要罚）
"""

import logging
import re
from collections import Counter

import jieba
import numpy as np

# jieba 首次加载会往 stderr 打四行"Building prefix dict..."，压到 ERROR 级别
jieba.setLogLevel(logging.ERROR)

_STOP = set(
    "的 了 是 在 和 与 或 及 等 这 那 其 之 也 都 就 而 以 于 对 从 为 被 把 有 不 "
    "一个 一种 我们 你们 他们 它 他 她 我 你 什么 哪些 如何 怎样 下列 关于 按照 "
    "a b c d 选项".split()
)

_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）()《》〈〉·—\-\.\,\;\:\?\!]+")


def tokenize(text: str) -> list[str]:
    text = _PUNCT.sub(" ", text)
    out = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w or w in _STOP:
            continue
        if len(w) == 1 and not w.isascii():
            continue          # 丢掉单个汉字，噪声大
        out.append(w.lower())
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.lens = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = self.lens.mean() if len(self.lens) else 1.0
        self.tfs = [Counter(t) for t in self.tokens]

        df = Counter()
        for tf in self.tfs:
            for w in tf:
                df[w] += 1
        self.n = len(docs)
        self.idf = {w: np.log((self.n - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_terms = [w for w in tokenize(query) if w in self.idf]
        if not q_terms:
            return np.zeros(self.n)
        out = np.zeros(self.n)
        for w in q_terms:
            idf = self.idf[w]
            for i, tf in enumerate(self.tfs):
                f=tf.get(w,0)
                if not f:
                    continue
                dl = self.lens[i]
                out[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)
