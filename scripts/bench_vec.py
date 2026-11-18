"""
向量库对比实验：暴力 vs 扁平 vs 倒排(IVF) vs 图(HNSW)

要回答的问题是：
  1. 扁平索引和 numpy 暴力算，结果一样吗？（应该完全一样，否则不是加速是换算法）
  2. 在我这个数据量（几百块）上，向量库有意义吗？
  3. 如果要撑到十万条，近似索引的**召回率代价**是多少？

第 3 个问题是重点，也是最容易被跳过的一步：
「查询从 18ms 降到 0.15ms」这种话，必须配召回率一起说，
否则等于在说「我少查了很多东西，所以变快了」。

合成数据的两版教训（也写在这里，因为两次都白跑了一轮）：
  v1 用 iid 随机向量 → 高维随机向量近似两两正交、没有聚类，
     任何 ANN 都失效（HNSW 召回 10%）。这不代表 ANN 差，是测试数据错。
  v2 用真实向量加扰动复制 → 每个原向量有几百份近似拷贝，
     top-10 边界上全是平局，任何实现都返回不同子集，召回天生为 0
     （召回算的是索引号集合的交集，平局时无意义）。
  v3 用「聚类中心 + 簇内扰动」→ 接近真实 embedding 的分布。

用法：python bench_vec.py
"""

import time

import numpy as np

import chunk as chunkmod
import embedder
from vectorstore import build


def timeit(fn, repeat=20):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def recall(truth, got) -> float:
    return len(set(truth) & set(got)) / max(1, len(truth))


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def clustered(n: int, dim: int, n_cluster: int, spread: float, seed=0):
    """造有聚类结构的数据：n_cluster 个中心，每个中心周围散布若干点。

    注意 noise 必须先归一化再乘 spread。第一版直接
    ，512 维下 randn 的模长约 sqrt(512)≈22.6，
    乘 0.35 就是 7.9，而中心向量模长只有 1——噪声完全盖住中心，
    点根本没有聚起来，ANN 当然失效（召回 4%）。
    """
    rng = np.random.default_rng(seed)
    centers = _unit(rng.standard_normal((n_cluster, dim)).astype(np.float32))
    idx = rng.integers(0, n_cluster, n)
    noise = _unit(rng.standard_normal((n, dim)).astype(np.float32))
    v = _unit(centers[idx] + noise * spread)
    return v, centers


def main():
    docs = chunkmod.load_docs()

    print("=" * 66)
    print("一、真实数据（500/100 切片）")
    chunks, _ = chunkmod.build(500, 100, docs, verbose=False)
    vecs = embedder.embed([c["text"] for c in chunks]).numpy()
    print(f"  {len(chunks)} 块 · 维度 {vecs.shape[1]}")

    qs = ["第三讲有多少道判断题", "第谷用什么工具观测", "开普勒三定律是什么",
          "林耐对物种的看法", "拉瓦锡的贡献"]
    qvecs = [embedder.embed_query(q).numpy() for q in qs]   # 先编码，只计时检索

    np_idx, fs_idx = build(vecs, "numpy"), build(vecs, "faiss")
    same = sum(1 for qv in qvecs if np_idx.search(qv, 5)[0] == fs_idx.search(qv, 5)[0])
    print(f"  一致性：{same}/{len(qvecs)} 个查询的 top-5 完全相同")
    print(f"  单次检索（不含编码）  numpy {timeit(lambda: np_idx.search(qvecs[0], 5)):.3f}ms"
          f"   faiss {timeit(lambda: fs_idx.search(qvecs[0], 5)):.3f}ms")
    print("  → 这个量级上两者都是几十微秒，换向量库没有收益")

    print()
    print("=" * 66)
    print("二、放大到十万条：精确 vs 近似的取舍")
    dim, n = vecs.shape[1], 100_000
    big, centers = clustered(n, dim, n_cluster=2000, spread=0.35)
    rng = np.random.default_rng(1)
    qidx = rng.integers(0, len(centers), 20)
    # 查询同样要归一化噪声，否则查询点离中心也很远
    queries = [_unit((centers[i] + _unit(rng.standard_normal((1, dim)).astype(np.float32))[0] * 0.3)[None, :])[0]
               for i in qidx]

    print(f"  数据 {n:,} 条 × {dim} 维，{len(queries)} 个查询，top-10")
    print()
    print(f"  {'方案':<20}{'平均耗时':>10}{'召回@10':>10}")
    print(f"  {'-'*40}")

    base = build(big, "numpy")
    truth = [base.search(q, 10)[0] for q in queries]
    t = timeit(lambda: base.search(queries[0], 10), repeat=5)
    print(f"  {'numpy 暴力':<20}{t:>8.2f}ms{100:>9}%")

    idx = build(big, "faiss")
    rs = [recall(truth[i], idx.search(q, 10)[0]) for i, q in enumerate(queries)]
    t = timeit(lambda: idx.search(queries[0], 10), repeat=5)
    print(f"  {'faiss 扁平':<20}{t:>8.2f}ms{np.mean(rs)*100:>9.0f}%")

    for nprobe in (1, 4, 8, 16):
        idx = build(big, "ivf")
        idx.index.nprobe = nprobe
        rs = [recall(truth[i], idx.search(q, 10)[0]) for i, q in enumerate(queries)]
        t = timeit(lambda: idx.search(queries[0], 10), repeat=5)
        print(f"  {'IVF nprobe=' + str(nprobe):<20}{t:>8.2f}ms{np.mean(rs)*100:>9.0f}%")

    for ef in (16, 32, 64, 128):
        idx = build(big, "hnsw")
        idx.index.hnsw.efSearch = ef
        rs = [recall(truth[i], idx.search(q, 10)[0]) for i, q in enumerate(queries)]
        t = timeit(lambda: idx.search(queries[0], 10), repeat=5)
        print(f"  {'HNSW ef=' + str(ef):<20}{t:>8.2f}ms{np.mean(rs)*100:>9.0f}%")


if __name__ == "__main__":
    main()
