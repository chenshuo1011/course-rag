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

# 显式禁用系统代理。
#
# 为什么要写这一行：
#   urllib 默认会读 HTTP_PROXY / HTTPS_PROXY 环境变量。科学上网工具装完之后
#   会往环境里塞一个类似 127.0.0.1:52633 的地址，工具一关，那个端口就没人监听，
#   于是连 api.deepseek.com 也会被塞进代理通道，报
#   WinError 10061「目标计算机积极拒绝」。
#
#   DeepSeek 的接口国内直连可达，不需要代理。所以这里固定走直连，
#   不受环境变量影响。真要挂代理，设 DEEPSEEK_PROXY=http://127.0.0.1:端口
#   就会改用那个地址，不会被环境里乱七八糟的值带偏。
_proxy = os.environ.get("DEEPSEEK_PROXY", "").strip()
_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": _proxy, "https": _proxy} if _proxy else {})
)



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
            with _OPENER.open(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"调用失败: {type(last_err).__name__}: {last_err}")


def chat_msgs(messages: list[dict], temperature: float = 0.0,
              max_tokens: int = 400, retries: int = 3) -> str:
    """多轮消息版本：Agent 需要把工具返回结果作为历史消息带回去，
    而 chat() 只支持 system + 一条 user。"""
    if not _CFG.get("API_KEY"):
        raise RuntimeError("未找到 API_KEY，检查 D:/AI-Study/.env")

    payload = {
        "model": _CFG["MODEL"],
        "messages": messages,
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
            with _OPENER.open(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            usage = data.get("usage", {})
            LAST_USAGE.clear()
            LAST_USAGE.update(usage)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"调用失败: {type(last_err).__name__}: {last_err}")


LAST_USAGE: dict = {}


if __name__ == "__main__":
    print(chat("你是一个简洁的助手。", "用一句话说明什么是检索。"))
