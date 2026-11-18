"""
两种检索后端：numpy 暴力 vs FAISS

项目现在用的是 numpy 暴力算（矩阵乘 + argpartition），496 条数据下毫秒级。
但真实 RAG 的索引动辄几十万条，暴力算 O(N) 会顶不住——所以要接向量库。

这里把两种实现放在同一个接口后面，方便：
  1. 验证两者结果是否一致（都是内积，应该一模一样）
  2. 量不同数据量下的耗时，找交叉点

为什么用 IndexFlatIP 而不是 IVF/HNSW：
  扁平索引是精确检索，这里是"对照暴力实现"，必须精确；
  IVF/HNSW 是近似检索，会引入额外变量，等准确率基线稳了再谈。
"""

import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    """归一化成单位向量。

    归一化之后，内积 = 余弦相似度。这样两种后端算的是同一个量，
    结果可比——否则一个比内积一个比欧氏距离，对比就没意义了。
    """
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class NumpyIndex:
    """暴力检索：全量算内积再取 top-k。"""

    name = "numpy"

    def __init__(self, vecs: np.ndarray):
        self.v = _l2norm(np.asarray(vecs, dtype=np.float32))

    def search(self, q: np.ndarray, k: int = 5):
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.v @ q                       # (N,)
        k = min(k, len(scores))
        # argpartition 是部分排序，只保证前 k 个最大，比全排序快
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist(), scores[idx].tolist()


class FaissIndex:
    """FAISS 扁平索引：同样是精确内积，但底层是 C++ 且做了 SIMD。"""

    name = "faiss"

    def __init__(self, vecs: np.ndarray):
        import faiss
        v = _l2norm(np.asarray(vecs, dtype=np.float32))
        self.index = faiss.IndexFlatIP(v.shape[1])
        self.index.add(v)
        self.n = v.shape[0]

    def search(self, q: np.ndarray, k: int = 5):
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1.0)
        k = min(k, self.n)
        sim, idx = self.index.search(q, k)
        return idx[0].tolist(), sim[0].tolist()


class FaissIVFIndex:
    """倒排索引：先聚类分桶，查询时只扫最近的几个桶。

    这是第一个"真的不对全量算"的方案——O(N) 变成 O(N/nlist × nprobe)。
    代价是近似：可能漏掉桶外的正确结果。所以必须同时报召回率，
    否则"变快了"可能只是"查得少了"。
    """

    name = "faiss-ivf"

    def __init__(self, vecs: np.ndarray, nlist: int = 64, nprobe: int = 8):
        import faiss
        v = _l2norm(np.asarray(vecs, dtype=np.float32))
        d = v.shape[1]
        # 桶数要小于数据量，否则有些桶分不到点，训练会报错
        self.nlist = max(4, min(nlist, max(4, v.shape[0] // 10)))
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        self.index.train(v)
        self.index.add(v)
        self.index.nprobe = min(nprobe, self.nlist)
        self.n = v.shape[0]

    def search(self, q: np.ndarray, k: int = 5):
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1.0)
        sim, idx = self.index.search(q, min(k, self.n))
        # IVF 没找到时返回 -1，过滤掉
        pairs = [(int(i), float(s)) for i, s in zip(idx[0], sim[0]) if i >= 0]
        return [p[0] for p in pairs], [p[1] for p in pairs]


class FaissHNSWIndex:
    """图索引：每个点连 M 个邻居，查询时沿图走。O(log N)，工业界最常用。"""

    name = "faiss-hnsw"

    def __init__(self, vecs: np.ndarray, m: int = 16, ef_search: int = 32):
        import faiss
        v = _l2norm(np.asarray(vecs, dtype=np.float32))
        self.index = faiss.IndexHNSWFlat(v.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 80
        self.index.add(v)
        self.index.hnsw.efSearch = ef_search
        self.n = v.shape[0]

    def search(self, q: np.ndarray, k: int = 5):
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1.0)
        sim, idx = self.index.search(q, min(k, self.n))
        return idx[0].tolist(), sim[0].tolist()


def build(vecs: np.ndarray, backend: str):
    cls = {"numpy": NumpyIndex, "faiss": FaissIndex,
           "ivf": FaissIVFIndex, "hnsw": FaissHNSWIndex}[backend]
    return cls(vecs)
