"""
统一的 embedding 封装

用 BAAI/bge-small-zh-v1.5：
  - 512 维，约 2400 万参数，CPU 上几十毫秒一条
  - 中文专用，短文本检索效果好

两个注意点：
1. 官网 huggingface.co 在国内不通，走 hf-mirror.com
2. BGE 模型建议给【查询】加一句指令前缀，【文档】不加
   这个细节对效果有影响，后面做对比实验时会验证
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 屏蔽底层 C++ 库的启动日志。
# 这些不是错误，只是"我用了一个加速库、浮点结果可能有极小差异"的提示，
# 但在命令行里会刷掉一屏，掩盖真正的输出。
# USE_TF=0 是关键：transformers 默认会尝试导入 TensorFlow，
# 而本项目只用 PyTorch，根本不需要它——不导入就不会有那些日志。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 官方给中文查询建议的指令前缀
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_tokenizer = None
_model = None


def _lazy_load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed(texts: list[str], batch_size: int = 32) -> torch.Tensor:
    """把一批文本编码成 L2 归一化后的向量，形状 (N, 512)"""
    tok, mdl = _lazy_load()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            out = mdl(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        chunks.append(torch.nn.functional.normalize(vec, p=2, dim=1))
    return torch.cat(chunks, dim=0)


def embed_query(q: str) -> torch.Tensor:
    """查询单独走这里 —— 会加指令前缀，返回 (1, 512)"""
    return embed([QUERY_INSTRUCTION + q])
