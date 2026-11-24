"""
以文搜图 go/no-go 实验：Sydney-Captions 上的 recall@k

为什么要先做这个：
  B（遥感图像问答）的前提是"文字能搜到图"。如果这个都做不好，
  后面接检测器、做 Agent 都是空中楼阁。
  而且这个实验的标准答案是免费的——描述本来属于哪张图，就是正确答案。

两个对照：
  · 英文原句直接检索   —— 检验 Chinese-CLIP 对英文的容忍度
  · 翻译成中文再检索   —— Chinese-CLIP 的主场
  两者的差距能说明"是否需要多做一步翻译"，那一步是有成本的。

数据来源（数据集已删，要复现必须重下）：
    https://hf-mirror.com/datasets/isaaccorley/Sydney-Captions/resolve/main/Sydney_captions.zip
    339MB，解压到 D:/datasets/sydney（613 张 tif + dataset.json）
    注意这个仓库只有 613 张，完整版应为 4981 张

复现说明：**仅凭缓存复现不了**。trans_zh.json 按描述去重后只剩 235 条，
而「哪句描述属于哪张图」的映射存在 dataset.json 里，删了数据就没了。
图像向量（613×512）还在 data/clip/，可用于聚类/相似图检索，但重算不出 recall。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_clip_sydney.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# torch 和别的库各带一份 OpenMP 运行时，同时加载会直接报 OMP #15 并终止进程。
# 这个开关让进程继续跑（官方说"unsafe"，但这里只做推理，不改权重，风险可接受）。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

import llm

DATA = Path(r"D:/datasets/sydney")
LOCAL = Path(r"D:/models/models/AI-ModelScope--chinese-clip-vit-base-patch16/snapshots/master")
OUT = Path(__file__).parent.parent / "data" / "clip"
MODEL = str(LOCAL) if LOCAL.exists() else "OFA-Sys/chinese-clip-vit-base-patch16"

KS = (1, 5, 10)
BATCH_TRANS = 15


def load_manifest():
    d = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))
    rows = []
    for im in d["images"]:
        rows.append({
            "imgid": im["imgid"],
            "file": im["filename"],
            "en": im["sentences"][0]["raw"].strip(),
        })
    return rows


def translate(texts, cache_path):
    """批量翻译英文描述，结果缓存，重跑不重复调模型。"""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [t for t in texts if t not in cache]
    print(f"  待翻译 {len(todo)} 条（已缓存 {len(cache)} 条）")

    for i in range(0, len(todo), BATCH_TRANS):
        batch = todo[i:i + BATCH_TRANS]
        user = "\n".join(f"{n+1}. {t}" for n, t in enumerate(batch))
        prompt = (
            "把下面的英文遥感图像描述翻译成中文。要求：\n"
            "1. 只输出译文，每行一条，行数必须和输入完全一致\n"
            "2. 不要编号，不要解释\n"
            "3. 保留“住宅区、道路、农田、跑道、河流”这类地名类词汇的直译\n\n"
            + user
        )
        try:
            out = llm.chat("你是遥感图像描述的翻译。", prompt, max_tokens=600).strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # 行数对不上就退化成逐条翻译，别把错位的译文写进缓存
            if len(lines) == len(batch):
                for t, z in zip(batch, lines):
                    cache[t] = z
            else:
                for t in batch:
                    one = llm.chat("你是遥感图像描述的翻译。只输出一句中文译文。",
                                   t, max_tokens=80).strip()
                    cache[t] = one
        except Exception as e:
            print(f"    第 {i} 批失败：{e}")
        print(f"  翻译进度 {min(i+BATCH_TRANS, len(todo))}/{len(todo)}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def main():
    rows = load_manifest()
    print(f"Sydney-Captions：{len(rows)} 张图")

    model = ChineseCLIPModel.from_pretrained(MODEL).eval()
    proc = ChineseCLIPProcessor.from_pretrained(MODEL)

    # ── 图像编码（一次，存盘复用）──
    vec_path = OUT / "sydney_img_vecs.npy"
    if vec_path.exists():
        vecs = np.load(vec_path)
        print(f"复用已编码的图像向量 {vecs.shape}")
    else:
        print("编码 613 张图 ...")
        t0 = time.time()
        imgs = [Image.open(DATA / "images" / r["file"]).convert("RGB") for r in rows]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt")
            v = model.get_image_features(**inp).numpy()
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, v)
        vecs = v
        print(f"  完成，{(time.time()-t0)/60:.1f} 分钟，维度 {v.shape[1]}")

    def text_vec(texts):
        with torch.no_grad():
            inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=52)
            t = model.get_text_features(**inp).numpy()
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    # 同一句描述往往对应好几张图（613 张只有 235 种描述，最多的一种对应 64 张）。
    # 所以"必须命中那张特定图"是错的判定——那样即使模型完美也只能 1/64 概率得分。
    # 正确做法是把同描述的所有图都算正样本（多正样本检索）。
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["en"].strip().lower()].append(i)
    positives = [set(groups[r["en"].strip().lower()]) for r in rows]

    def evaluate(queries, tag):
        tv = text_vec(queries)
        sims = tv @ vecs.T                       # (Q, N)
        exact = {k: 0 for k in KS}
        anny = {k: 0 for k in KS}
        for i, s in enumerate(sims):
            order = np.argsort(-s)
            rank_exact = int(np.where(order == i)[0][0]) + 1
            pos = positives[i]
            rank_any = min(int(np.where(order == p)[0][0]) + 1 for p in pos)
            for k in KS:
                if rank_exact <= k:
                    exact[k] += 1
                if rank_any <= k:
                    anny[k] += 1
        n = len(queries)
        print(f"  {tag}")
        print("    " + "严格(必须那张图)".ljust(18)
              + "".join(f"@{k} {exact[k]/n:>5.0%}" for k in KS))
        print("    " + "宽松(同描述即可)".ljust(18)
              + "".join(f"@{k} {anny[k]/n:>5.0%}" for k in KS))
        return anny

    print(f"\n  随机基线 recall@1 = {1/len(rows):.2%}（613 张里瞎猜一张）")
    evaluate([r["en"] for r in rows], "英文原句")

    print("\n翻译成中文 ...")
    trans = translate([r["en"] for r in rows], OUT / "trans_zh.json")
    zh = [trans[r["en"]] for r in rows]
    evaluate(zh, "中文译文")

    print("\n最近邻示例（中文查询）：")
    tv = text_vec(zh[:3])
    for i, s in enumerate(tv @ vecs.T):
        order = np.argsort(-s)[:3]
        print(f"  「{zh[i]}」")
        for o in order:
            mark = "✓" if o == i else " "
            print(f"    {mark} 第{o+1}张 相似度{s[o]:.3f}  {rows[o]['en'][:50]}")


if __name__ == "__main__":
    main()
