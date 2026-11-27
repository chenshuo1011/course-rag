"""
视觉模型调用（场景描述）。

为什么单独一个文件、不复用 scripts/llm.py：
  那个文件被 10 个脚本依赖（课程问答 7 个 + 多模态 3 个），
  为加一个视觉接口去改它，等于把两条线一起置于风险中。

定位：
  只说"这张图大概是什么场景"，**不做计数、不参与评测**。
  计数走检测器那条线（有框、有置信度、有实测召回率），
  这条路给的是开放式描述，无法核对，所以页面上要分开标注。

用法：
    from llm_vision import describe
    describe("D:/visdrone/x.jpg")            # 默认问场景
    describe(path, "这张图是白天还是晚上")

依赖：只需要标准库 + cv2（压图用）。key 读 D:/AI-Study/.env。
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

ENV = Path("D:/AI-Study/.env")
MAX_SIDE = 1280      # 长边超过就缩小：一张 4000px 的图转 base64 有 5MB+，接口会拒
JPEG_Q = 85


def _env() -> dict:
    cfg = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                cfg[k.strip()] = v.strip()
    # 环境变量优先，方便临时替换
    for k in ("VLM_API_KEY", "VLM_BASE_URL", "VLM_MODEL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def available() -> bool:
    return bool(_env().get("VLM_API_KEY"))


def _to_data_uri(image_path: str | Path) -> tuple[str, int]:
    """读图 → 必要时缩小 → base64 data URI。返回 (uri, 边长)。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"读不出这张图：{image_path}")
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_SIDE:
        s = MAX_SIDE / long_side
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        raise ValueError("编码 JPEG 失败")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}", max(img.shape[:2])


def describe(image_path: str | Path,
             question: str = "用一句话描述这张图的场景，不要数数") -> dict:
    """调视觉模型。返回 {answer, model, ms, side}。失败抛异常，由调用方决定怎么兜。"""
    cfg = _env()
    key = cfg.get("VLM_API_KEY")
    if not key:
        raise RuntimeError("没有配置 VLM_API_KEY（写在 D:/AI-Study/.env）")

    uri, side = _to_data_uri(image_path)
    payload = {
        "model": cfg.get("VLM_MODEL", "glm-4v-flash"),
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": question[:300]},
        ]}],
    }
    req = urllib.request.Request(
        cfg.get("VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    # 显式直连，不读系统代理——和 scripts/llm.py 同一个理由：
    # 科学上网工具关掉后环境变量里还留着端口，会把请求塞进没人监听的通道
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        raise RuntimeError(f"接口返回 {e.code}：{body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"连不上接口：{e.reason}") from e

    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return {"answer": text.strip(),
            "model": payload["model"],
            "ms": round((time.perf_counter() - t0) * 1000),
            "side": side}


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "D:/visdrone/VisDrone2019-DET-val/images/0000280_03401_d_0000629.jpg"
    q = sys.argv[2] if len(sys.argv) > 2 else "用一句话描述这张图的场景，不要数数"
    print("可用:", available())
    r = describe(p, q)
    print(f"\n{r['answer']}\n\n模型 {r['model']}  输入长边 {r['side']}  {r['ms']}ms")
