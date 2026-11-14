"""
压测：很多人同时问，系统会变成什么样

为什么必须量：
  单人用时 1.7 秒，感觉良好。但"1.7 秒"是在没有任何竞争的情况下测出来的。
  并发上来之后，CPU 只有一颗、模型调用要排队，真实延迟可能是它的十倍。
  "体验好不好"这个判断，只有压过才有资格说。

只用标准库，不装新依赖。

用法（服务必须先跑起来）：
    "D:/anaconda3/envs/yolo/python.exe" loadtest.py
    "D:/anaconda3/envs/yolo/python.exe" loadtest.py -c 20 -r 3
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/ask"
# 绕开系统代理：本机请求不该走代理，否则会像之前那样报 10061
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

QUESTIONS = [
    "第三讲有多少道判断题",
    "多选题比单选题多几道",
    "哪几讲提到了盖伦",
    "第谷接受的是哪种学说",
    "哪一讲的题目最多",
]


def one(q: str, out: list):
    body = json.dumps({"q": q}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with OPENER.open(req, timeout=180) as r:
            d = json.loads(r.read())
        out.append({"ok": True, "wall": time.time() - t0,
                    "ms": d.get("elapsed_ms", 0), "tokens": d.get("tokens", 0)})
    except Exception as e:
        out.append({"ok": False, "wall": time.time() - t0, "err": str(e)[:60]})


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--concurrency", type=int, default=10, help="同时发几个")
    ap.add_argument("-r", "--rounds", type=int, default=1, help="发几轮")
    args = ap.parse_args()

    total = args.concurrency * args.rounds
    print(f"并发 {args.concurrency} × {args.rounds} 轮 = {total} 个请求")
    print("=" * 60)

    results, threads = [], []
    t0 = time.time()
    for r in range(args.rounds):
        for i in range(args.concurrency):
            th = threading.Thread(target=one, args=(QUESTIONS[i % len(QUESTIONS)], results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        threads = []
        ok = sum(1 for x in results if x["ok"])
        print(f"  第 {r + 1} 轮完成，累计成功 {ok}/{len(results)}")

    wall = time.time() - t0
    good = [x for x in results if x["ok"]]
    bad = [x for x in results if not x["ok"]]

    print("=" * 60)
    print(f"成功 {len(good)} / 失败 {len(bad)}")
    if bad:
        print("失败样例：")
        for x in bad[:3]:
            print("  ", x["err"])

    if not good:
        return

    walls = [x["wall"] for x in good]
    tokens = [x["tokens"] for x in good]
    print()
    print(f"端到端耗时（从发出到收到）：")
    print(f"  平均 {statistics.mean(walls):.2f}s   p50 {pct(walls, 50):.2f}s   "
          f"p95 {pct(walls, 95):.2f}s   最慢 {max(walls):.2f}s")
    print()
    print(f"服务端自报耗时（不含网络）：")
    srv = [x["ms"] / 1000 for x in good]
    print(f"  平均 {statistics.mean(srv):.2f}s   p95 {pct(srv, 95):.2f}s")
    print()
    print(f"token：合计 {sum(tokens):,}   平均 {int(statistics.mean(tokens))} / 次")
    print(f"实际吞吐：{len(good) / wall:.2f} 请求/秒（总耗时 {wall:.1f}s）")

    # 存盘，方便不同并发下对比
    out = f"../data/loadtest_c{args.concurrency}_r{args.rounds}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"concurrency": args.concurrency, "rounds": args.rounds,
                   "ok": len(good), "fail": len(bad), "wall": wall,
                   "p50": pct(walls, 50), "p95": pct(walls, 95),
                   "max": max(walls), "tokens_sum": sum(tokens)}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n结果已存 {out}")


if __name__ == "__main__":
    main()
