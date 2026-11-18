"""
nprobe / efSearch 实验：速度与召回率

这个脚本回答一件事：近似索引的"快"是用什么换来的。

做法：
  1. 用 numpy 暴力检索算 top-10，作为标准答案
  2. 同一个查询，用 IVF / HNSW 在不同参数下各查一次
  3. 比对：花了多少时间，命中了标准答案里的几个

两个细节（第一版都踩过）：
  - 查询必须带噪声、且取多个平均。直接拿簇中心当查询，答案太明显，
    任何索引都能找到，召回全是 100%，看不出任何取舍。
  - 召回要和耗时一起报。只看耗时，nprobe=1 永远最快——但那可能只是"少查了"。

用法：
    cd /d D:/course-rag/scripts
    "D:/anaconda3/envs/yolo/python.exe" try_params.py
"""

import time

import numpy as np

import bench_vec
from vectorstore import build

N, DIM, K = 100_000, 512, 10

print(f"造数据：{N:,} 条 x {DIM} 维 ...")
big, centers = bench_vec.clustered(N, DIM, 2000, 0.35)

# 20 个查询，各自落在某个簇中心的边缘（加 0.3 扰动）
rng = np.random.default_rng(1)
queries = []
for i in rng.integers(0, len(centers), 20):
    noise = rng.standard_normal(DIM).astype(np.float32)
    noise /= np.linalg.norm(noise)
    v = centers[i] + noise * 0.3
    queries.append(v / np.linalg.norm(v))

brute = build(big, "numpy")
t0 = time.perf_counter()
truth = [brute.search(q, K)[0] for q in queries]
t_brute = (time.perf_counter() - t0) / len(queries) * 1000
print(f"\n暴力检索：{t_brute:.2f}ms/次（{len(queries)} 个查询，top-{K}）")


def avg_recall(got):
    return float(np.mean([len(set(t) & set(g)) / K for t, g in zip(truth, got)]))


print(f"\n{'IVF':<6}{'nprobe':>8}{'耗时':>10}{'召回':>8}")
ivf = build(big, "ivf")
for nprobe in (1, 2, 4, 8, 32):
    ivf.index.nprobe = nprobe
    ivf.search(queries[0], K)
    t0 = time.perf_counter()
    got = [ivf.search(q, K)[0] for q in queries]
    ms = (time.perf_counter() - t0) / len(queries) * 1000
    print(f"{'':<6}{nprobe:>8}{ms:>8.2f}ms{avg_recall(got):>8.0%}")

print(f"\n{'HNSW':<6}{'ef':>8}{'耗时':>10}{'召回':>8}")
hnsw = build(big, "hnsw")
for ef in (10, 20, 40, 80, 160):
    hnsw.index.hnsw.efSearch = ef
    hnsw.search(queries[0], K)
    t0 = time.perf_counter()
    got = [hnsw.search(q, K)[0] for q in queries]
    ms = (time.perf_counter() - t0) / len(queries) * 1000
    print(f"{'':<6}{ef:>8}{ms:>8.2f}ms{avg_recall(got):>8.0%}")
