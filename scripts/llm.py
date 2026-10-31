"""
极简 LLM 调用封装（只用标准库，不装 openai 包）

用 DeepSeek 的 OpenAI 兼容接口，读 D:/AI-Study/.env 里的密钥。
有重试和退避：网络抖动时不至于整批失败。
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV = Path("D:/AI-Study/.env")


def _load_env():
    cfg = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg.setdefault("BASE_URL", "https://api.deepseek.com")
    cfg.setdefault("MODEL", "deepseek-chat")
    if os.environ.get("DEEPSEEK_API_KEY"):
        cfg["API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    return cfg


_CFG = _load_env()


def chat(system: str, user: str, temperature: float = 0.0,
         max_tokens: int = 120, retries: int = 3) -> str:
    if not _CFG.get("API_KEY"):
        raise RuntimeError("未找到 API_KEY，检查 D:/AI-Study/.env")

    payload = {
        "model": _CFG["MODEL"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = _CFG["BASE_URL"].rstrip("/") + "/chat/completions"

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_CFG['API_KEY']}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"调用失败: {type(last_err).__name__}: {last_err}")


if __name__ == "__main__":
    print(chat("你是一个简洁的助手。", "用一句话说明什么是检索。"))
