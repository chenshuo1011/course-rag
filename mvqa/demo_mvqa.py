"""
问答效果演示：一次跑几条不同类型的问题，打印完整过程

为什么要一次跑完：
  单条单进程的话，每条都要重新加载 6MB 权重 + 预热（约 20 秒）。
  一次跑完只付一次加载成本。

挑的 6 条覆盖：
  单类计数 / 低召回类别（看措辞）/ 方位 / 全景 / 组合 / 识别边界

用法：
    "D:\anaconda3\envs\yolo\python.exe" demo_mvqa.py
"""

import json

from mvqa import run

IMG = "0000280_03401_d_0000629"      # 密集场景，51 辆车 / 10 辆摩托

QUESTIONS = [
    "这张图里有多少辆车",
    "图里有多少行人",
    "人主要在哪一片",
    "这张图里都有什么",
    "车和摩托车各有多少",
    "图里有船吗",
]


def main():
    print(f"图片：{IMG}\n")
    for i, q in enumerate(QUESTIONS, 1):
        print("=" * 70)
        print(f"{i}. {q}")
        print("-" * 70)
        r = run(q, IMG, verbose=False)
        for st in r["steps"]:
            args = json.dumps(st["args"], ensure_ascii=False)
            print(f"  调用 {st['tool']}({args})")
            res = st["result"]
            if "各类数量" in res:
                brief = "、".join(f"{x['类别']}{x['数量']}" for x in res["各类数量"][:6])
                print(f"    → 共 {res.get('总计')} 个目标：{brief}")
            elif "方位分布" in res:
                brief = "、".join(f"{x['方位']}{x['数量']}" for x in res["方位分布"])
                print(f"    → {res['检出数量']} 个，分布：{brief}")
            else:
                print(f"    → 检出 {res.get('检出数量')}，"
                      f"建议措辞「{res.get('建议措辞','')}」"
                      f"{('，' + res['说明'][:40]) if res.get('说明') else ''}")
        print(f"  【答】{r['answer']}")
        print()


if __name__ == "__main__":
    main()
