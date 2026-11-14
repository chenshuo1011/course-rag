"""
把 agent 包成一个 HTTP 服务

为什么要有这一层：
  命令行版只能"你坐在电脑前打字"。服务化之后，网页、小程序、别的系统都能调，
  也才谈得上"并发""延迟""成本"——这三个词在单人对话里根本不存在。

启动：
    "D:/anaconda3/envs/yolo/python.exe" -m uvicorn server:app --port 8000
  然后浏览器打开 http://127.0.0.1:8000/docs ，可以直接点着发请求。

一个关键设计（这件事值得单独讲）：
  ask() 写成普通 def，不是 async def。
    agent.run() 内部要调 DeepSeek（等 2~9 秒）、跑 embedding，全是同步阻塞。
    如果写成 async def，这个阻塞会卡住 uvicorn 唯一的事件循环线程——
    第一个请求还没回来，后面所有人都在排队，看起来像"服务死了"。
    写成 def，FastAPI 会自动把它丢进线程池，每个请求占一条线程。
    代价是线程数上不封顶，所以后面还得加信号量限流。
"""

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import KB, run

ROOT = Path(__file__).parent.parent
LOG = ROOT / "data" / "server_requests.jsonl"

# 并发上限（压测后定的）
#
# 这个计数必须放在**中间件**里，不能放在处理函数里。
# 第一版写在 ask() 开头，100 并发压测时一个 429 都没出现——因为 FastAPI 的
# 线程池只有 40 条线程，另外 60 个请求在"排队等线程"，根本没执行到那行判断。
# 换句话说：函数内的限流只能挡住"已经在跑"的请求，挡不住"在门外等"的请求。
# 中间件在事件循环里同步执行，请求一进来就被计数，这里判断才有意义。
MAX_CONCURRENT = 40
_in_flight = 0
_lock = threading.Lock()
_log_lock = threading.Lock()   # 保护日志写入，见 ask() 里的说明

app = FastAPI(title="课程题库答疑服务", version="0.1")


@app.middleware("http")
async def limit_concurrency(request, call_next):
    global _in_flight
    if request.url.path != "/ask":
        return await call_next(request)

    with _lock:
        if _in_flight >= MAX_CONCURRENT:
            busy = True
        else:
            _in_flight += 1
            busy = False

    if busy:
        # 直接拒，不排队：排队会让所有人都慢，最后一起超时
        return JSONResponse(
            status_code=429,
            content={"detail": f"服务繁忙：当前并发上限 {MAX_CONCURRENT}，请稍后重试"},
        )
    try:
        return await call_next(request)
    finally:
        with _lock:
            _in_flight -= 1


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=200, description="学生的问题")


class AskResponse(BaseModel):
    answer: str
    tools: list[str]
    stop: str
    elapsed_ms: int
    tokens: int
    # 输入和输出要分开记：单价不一样，按 total 算成本会错得离谱
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


kb: KB | None = None


@app.on_event("startup")
def _load():
    """启动时加载索引和模型（约 10 秒），不要放到每个请求里。"""
    global kb
    print("加载索引与 BM25 ...")
    kb = KB()
    print(f"就绪：{len(kb.meta)} 道题")


@app.get("/health")
def health():
    return {"ok": kb is not None, "questions": len(kb.meta) if kb else 0}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """注意这里是 def 不是 async def，原因见文件顶部。

    elapsed_ms 从进入函数算起，包含检索 + 调模型的全过程。
    tokens 取 DeepSeek 返回的 usage.total_tokens，用来算成本。
    """
    # 限流已经交给上面的中间件，这里只管业务
    t0 = time.time()
    r = run(kb, req.q, verbose=False)
    ms = int((time.time() - t0) * 1000)
    usage = r["usage"]
    tokens = int(usage.get("total_tokens") or 0)
    p_tok = int(usage.get("prompt_tokens") or 0)
    c_tok = int(usage.get("completion_tokens") or 0)
    # 缓存命中的输入便宜 50 倍，计价时必须单独拿出来，否则成本会高估一倍
    hit_tok = int(usage.get("prompt_cache_hit_tokens") or 0)

    # 每次请求落一行，后面算 p50/p95 和成本都靠它
    #
    # 这里必须加锁。压测时几十条线程同时以 "a" 模式打开同一个文件写入，
    # 内容会互相截断——实测 362 行里坏了 3 行（0.8%），
    # 表现为某些行只剩结尾一段，json 解析直接报错。
    # 锁 + 一次 write 写完整行，才不会互相插进去。
    line = json.dumps({
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "q": req.q, "ms": ms, "tokens": tokens,
        "prompt_tokens": p_tok, "completion_tokens": c_tok,
        "cache_hit_tokens": hit_tok,
        "tools": r["calls"], "stop": r["stop"],
        "fallback": r["stop"] == "fallback",
        "answer_len": len(r["answer"]),
    }, ensure_ascii=False)
    with _log_lock:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    return AskResponse(
        answer=r["answer"],
        tools=[s["action"] for s in r["steps"] if s["action"] != "final"],
        stop=r["stop"],
        elapsed_ms=ms,
        tokens=tokens,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cache_hit_tokens=hit_tok,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
