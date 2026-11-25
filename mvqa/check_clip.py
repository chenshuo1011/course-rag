"""
Chinese-CLIP 环境验证：能不能跑、跑多快、检索准不准

为什么先做这个而不是先建库：
  以文搜图整个方案的成本，八成压在"图像编码"上。
  如果 CPU 上一张要 500ms，那 2 万张就是近 3 小时，方案得推翻。
  所以先量这一个数，五分钟就能决定后面怎么排。

三步：
  1. 加载模型（首次会下载约 700MB，走 hf-mirror）
  2. 量编码速度（文本 / 单图 / 批量）
  3. 做一个 5 张图的小检索测试，看排序对不对

用法：
    cd /d D:\course-rag\scripts
    "D:\anaconda3\envs\yolo\python.exe" check_clip.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

LOCAL = Path(r"D:/models/models/AI-ModelScope--chinese-clip-vit-base-patch16/snapshots/master")
MODEL = str(LOCAL) if LOCAL.exists() else "OFA-Sys/chinese-clip-vit-base-patch16"
PICS = Path(r"D:\86155\Pictures")


def main():
    print(f"加载 {MODEL} ...")
    t0 = time.time()
    model = ChineseCLIPModel.from_pretrained(MODEL).eval()
    proc = ChineseCLIPProcessor.from_pretrained(MODEL)
    print(f"  加载耗时 {time.time()-t0:.1f}s，参数量 "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # 拿几张真实图片（屏幕上截的图、照片都行，先验证能不能编码）
    files = sorted([p for p in PICS.glob("*") if p.suffix.lower() in
                    (".png", ".jpg", ".jpeg")])[:5]
    if not files:
        print("Pictures 目录下没有图片，先放几张进来。")
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    print(f"\n测试图片 {len(imgs)} 张：" + "、".join(f.name[:18] for f in files))

    # ── 速度 ──
    def encode_images(batch):
        with torch.no_grad():
            inp = proc(images=batch, return_tensors="pt")
            return model.get_image_features(**inp)

    def encode_texts(texts):
        with torch.no_grad():
            inp = proc(text=texts, return_tensors="pt", padding=True)
            return model.get_text_features(**inp)

    encode_images([imgs[0]])                      # 预热
    encode_texts(["预热"])

    t0 = time.perf_counter(); encode_texts(["靠海的港口，有很多小船"]); t_text = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter(); encode_images([imgs[0]]);                 t_one = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter(); encode_images(imgs);                      t_batch = (time.perf_counter()-t0)*1000

    print(f"\n{'编码速度':<14}{'耗时':>10}")
    print(f"  {'单条文本':<12}{t_text:>8.0f}ms")
    print(f"  {'单张图片':<12}{t_one:>8.0f}ms")
    print(f"  {'{n} 张批量'.format(n=len(imgs)):<12}{t_batch:>8.0f}ms  （每张 {t_batch/len(imgs):.0f}ms）")
    print(f"\n  推算：2000 张约 {t_batch/len(imgs)*2000/1000/60:.1f} 分钟"
          f"，20000 张约 {t_batch/len(imgs)*20000/1000/60:.0f} 分钟")

    # ── 检索准不准（小样本，只验证流程通不通）──
    # 用中文描述这几张图，看能不能把对应的图排到第一
    queries = [
        ("一张软件界面截图", 0),
        ("一张照片或图片", 1),
    ]
    vecs = encode_images(imgs)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    print("\n小检索测试（只验证流程，不代表真实精度）：")
    for q, expect in queries:
        tv = encode_texts([q])
        tv = tv / tv.norm(dim=-1, keepdim=True)
        sims = (tv @ vecs.T)[0]
        order = torch.argsort(-sims).tolist()
        top1 = order[0]
        mark = "✓" if top1 == expect else "✗"
        print(f"  {mark} 「{q}」→ top1 是第 {top1+1} 张"
              f"（相似度 {sims[top1]:.3f}，最高 {sims[order[0]]:.3f}）")


if __name__ == "__main__":
    main()
