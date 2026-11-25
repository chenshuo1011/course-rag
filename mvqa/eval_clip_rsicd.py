"""
在 RSICD 上重做以文搜图评测：分开"模型弱"和"数据集雷同"这两个病因

为什么换数据集：
  Sydney 那 613 张图只有 235 种描述，最多的一种对应 64 张图——
  那种数据只能测"场景分类"，测不出"能不能精确找到某张图"。
  结果 @5 只有 36%（宽松判定），但无法判断是模型不行还是数据不行。

RSICD：1093 张（test split），描述更细（"机场很大"、"机场旁边是绿草地"），
描述之间的区分度明显更高。

判定两个口径：
  严格 —— 必须命中描述对应的那张图
  宽松 —— 命中任何一张同描述的图（RSICD 描述基本唯一，两者应该接近）

数据来源（parquet 只有 53MB，留在 D:/datasets/ 里，所以能复现）：
    https://hf-mirror.com/datasets/arampacha/rsicd/resolve/main/data/test-00000-of-00001.parquet
    1093 张图 + 英文描述（图以 bytes 存在 parquet 里）
    下载注意：这个镜像偶尔只给部分文件（实测拿到过 3.6MB 的截断版），
    用 curl -C - 续传后再用 pyarrow 校验行数是否为 1093。

复现说明：向量和译文缓存在 data/clip/，重跑不用再编码、不用再调模型翻译。
实测复核：重下数据后重跑，数字与首次完全一致（英文 2/13/24、中文 7/19/30，严格口径）。

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_clip_rsicd.py
"""

import io
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# 两个模块共用的模型封装（scripts/llm.py）不在本目录，
# 把它所在目录加进 sys.path，避免复制一份出来各改各的。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import llm
from eval_clip_sydney import translate          # 复用带缓存的翻译函数

PARQUET = Path(r"D:/datasets/rsicd_test.parquet")
LOCAL = Path(r"D:/models/models/AI-ModelScope--chinese-clip-vit-base-patch16/snapshots/master")
OUT = Path(__file__).parent.parent / "data" / "clip"
MODEL = str(LOCAL) if LOCAL.exists() else "OFA-Sys/chinese-clip-vit-base-patch16"

KS = (1, 5, 10)


def main():
    t = pq.read_table(PARQUET)
    files = [x for x in t.column("filename").to_pylist()]
    caps = [c[0].strip() for c in t.column("captions").to_pylist()]
    raw = [x["bytes"] for x in t.column("image").to_pylist()]
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raw]
    n = len(imgs)
    print(f"RSICD test：{n} 张图")

    uniq = len(set(c.lower() for c in caps))
    print(f"首句描述去重后 {uniq} 条 → 平均每条对应 {n/uniq:.2f} 张图"
          f"（Sydney 是 2.61，越接近 1 越能测出检索能力）")

    model = ChineseCLIPModel.from_pretrained(MODEL).eval()
    proc = ChineseCLIPProcessor.from_pretrained(MODEL)

    vec_path = OUT / "rsicd_img_vecs.npy"
    if vec_path.exists():
        vecs = np.load(vec_path)
        print(f"复用已编码向量 {vecs.shape}")
    else:
        print("编码图像 ...")
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt")
            v = model.get_image_features(**inp).numpy()
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, v)
        vecs = v
        print(f"  完成，维度 {v.shape[1]}")

    def text_vec(texts):
        with torch.no_grad():
            inp = proc(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=52)
            x = model.get_text_features(**inp).numpy()
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    groups = {}
    for i, c in enumerate(caps):
        groups.setdefault(c.lower(), []).append(i)

    def evaluate(queries, tag):
        tv = text_vec(queries)
        sims = tv @ vecs.T
        exact = {k: 0 for k in KS}
        anny = {k: 0 for k in KS}
        ranks = []
        for i, s in enumerate(sims):
            order = np.argsort(-s)
            r_exact = int(np.where(order == i)[0][0]) + 1
            r_any = min(int(np.where(order == p)[0][0]) + 1
                        for p in groups[caps[i].lower()])
            ranks.append(r_exact)
            for k in KS:
                exact[k] += r_exact <= k
                anny[k] += r_any <= k
        print(f"  {tag}")
        print("    " + "严格(必须那张图)".ljust(18)
              + "".join(f"@{k} {exact[k]/n:>5.0%}" for k in KS))
        print("    " + "宽松(同描述即可)".ljust(18)
              + "".join(f"@{k} {anny[k]/n:>5.0%}" for k in KS))
        ranks.sort()
        print(f"    名次中位数 {ranks[n//2]}，前 10% 分位 {ranks[n//10]}")

    print(f"\n  随机基线 recall@1 = {1/n:.3%}")
    evaluate(caps, "英文原句")

    print("\n翻译成中文 ...")
    tr = translate(caps, OUT / "rsicd_trans_zh.json")
    zh = [tr[c] for c in caps]
    print()
    evaluate(zh, "中文译文")

    print("\n最近邻示例：")
    tv = text_vec(zh[:3])
    for i, s in enumerate(tv @ vecs.T):
        order = np.argsort(-s)[:3]
        print(f"  「{zh[i]}」")
        for o in order:
            print(f"    {'✓' if o == i else ' '} 第{o+1}张 {s[o]:.3f}  {caps[o][:50]}")


if __name__ == "__main__":
    main()
