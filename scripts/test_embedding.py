"""
测试中文 embedding 模型能否正常下载和使用
模型：BAAI/bge-small-zh-v1.5（约 95MB，CPU 上很快）
"""

import os

# HF 官网在国内不通，走镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print(f"正在加载模型: {MODEL_NAME}")
print("（首次需要下载，约 95MB）\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("模型加载成功")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")


def embed(texts: list[str]) -> torch.Tensor:
    """把一批文本编码成向量（BGE 的标准做法：mean pooling + L2 归一化）"""
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**encoded)
    # 用 attention_mask 做加权平均，屏蔽掉 padding
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    # L2 归一化，这样余弦相似度就等于点积
    return torch.nn.functional.normalize(vec, p=2, dim=1)


sents = [
    "什么是自然辩证法",
    "自然辩证法的研究对象是什么",
    "今天中午吃什么",
]

vecs = embed(sents)
print(f"\n向量维度: {vecs.shape}")

sim = vecs @ vecs.T
print("\n相似度矩阵：")
for i in range(len(sents)):
    for j in range(len(sents)):
        print(f"  {sents[i][:12]:<14} vs {sents[j][:12]:<14} = {sim[i][j]:.4f}")
