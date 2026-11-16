"""
错题本的验收脚本

不是单元测试框架，就是把三个函数挨个跑一遍，看结果对不对。
未实现的函数会被记成"未实现"，不影响其他步骤继续跑——
这样你可以先只写 add()，跑一次看它过没过。

用法（在 scripts 目录下）：
    "D:/anaconda3/envs/yolo/python.exe" check_wrongbook.py
"""

import json
import shutil
from pathlib import Path

import wrongbook

BOOK = wrongbook.BOOK
BACKUP = Path(str(BOOK) + ".bak")

results = []


def call(fn, *a, **kw):
    """调函数；没写完就返回一个特殊标记，别让整个脚本崩掉。"""
    try:
        return fn(*a, **kw), None
    except NotImplementedError as e:
        return None, str(e)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def check(title, expect, fn, *a, post=None, **kw):
    """post：对返回值再做一层变换，比如 len()。

    加这个参数是因为第一版拿 `2` 直接比一个列表，
    结果两条正确的实现被判成 BAD——测试本身写错了，
    和之前评测判分器太宽是同一类问题：验证工具也要被验证。
    """
    actual, err = call(fn, *a, **kw)
    if err:
        print(f"  [未实现] {title}  ← {err}")
        results.append(False)
        return None
    if post is not None:
        try:
            actual = post(actual)
        except Exception as e:
            print(f"  [BAD] {title}（post 变换失败：{e}）")
            results.append(False)
            return None
    ok = expect == actual
    print(f"  [{'OK ' if ok else 'BAD'}] {title}")
    if not ok:
        print(f"        期望: {expect}")
        print(f"        实际: {actual}")
    results.append(ok)
    return actual


def main():
    if BOOK.exists():
        shutil.copy(BOOK, BACKUP)
        BOOK.unlink()
    try:
        run()
    finally:
        if BACKUP.exists():
            shutil.move(BACKUP, BOOK)


def run():
    print("=" * 60)
    print("第 1 步：空库不该崩（文件还不存在）")
    check("list_by_lecture() 返回空列表", [], wrongbook.list_by_lecture)
    check("stats() 返回零", {"总条数": 0, "按讲次": {}}, wrongbook.stats)

    print("=" * 60)
    print("第 2 步：写三条记录")
    r1, err = call(wrongbook.add, "single-第五讲-1", "第五讲",
                   "第谷用什么观测", "肉眼", "肉眼观测")
    if err:
        print(f"  [未实现] add()  ← {err}")
        results.append(False)
    else:
        results.append(r1.get("qid") == "single-第五讲-1")
        results.append(bool(r1.get("t")))
        print(f"  [{'OK ' if results[-2] else 'BAD'}] add() 返回的 qid = {r1.get('qid')}")
        print(f"  [{'OK ' if results[-1] else 'BAD'}] 返回里有时间字段 t = {r1.get('t')}")
        call(wrongbook.add, "single-第五讲-1", "第五讲", "第谷用啥观测", "望远镜", "肉眼观测")
        call(wrongbook.add, "judge-第三讲　古代中国-13", "第三讲　古代中国",
             "第谷理论谁传到中国", "利玛窦", "耶稣会教士")

        print("=" * 60)
        print("第 3 步：文件里应该是三行，每行都是合法 JSON")
        lines = BOOK.read_text(encoding="utf-8").splitlines() if BOOK.exists() else []
        good = [l for l in lines if l.strip()]
        results.append(len(good) == 3)
        print(f"  [{'OK ' if results[-1] else 'BAD'}] 行数 = {len(good)}（期望 3）")
        try:
            alljson = all(isinstance(json.loads(l), dict) for l in good)
        except Exception:
            alljson = False
        results.append(alljson)
        print(f"  [{'OK ' if alljson else 'BAD'}] 每行都能解析成 JSON")
        if good:
            print(f"        第 1 行: {good[0][:90]}")

    print("=" * 60)
    print("第 4 步：按讲次查")
    fifth, _ = call(wrongbook.list_by_lecture, "第五讲")
    check("第五讲条数", 2, wrongbook.list_by_lecture, "第五讲", post=len)
    if isinstance(fifth, list) and fifth:
        ok = fifth[0].get("wrong") == "望远镜"
        results.append(ok)
        print(f"  [{'OK ' if ok else 'BAD'}] 最近的在前 = {fifth[0].get('wrong')}（期望 望远镜）")
    check("limit 生效", 1, wrongbook.list_by_lecture, "第五讲", post=len, limit=1)
    check("查不存在的讲次返回空", 0, wrongbook.list_by_lecture, "第九讲", post=len)

    print("=" * 60)
    print("第 5 步：统计")
    st, err = call(wrongbook.stats)
    if err:
        print(f"  [未实现] stats()  ← {err}")
        results.append(False)
    else:
        results.append(st.get("总条数") == 3)
        print(f"  [{'OK ' if results[-1] else 'BAD'}] 总条数 = {st.get('总条数')}（期望 3）")
        per = (st.get("按讲次") or {}).get("第五讲")
        results.append(per == 2)
        print(f"  [{'OK ' if per == 2 else 'BAD'}] 第五讲计数 = {per}（期望 2）")
        uni = (st.get("去重") or {}).get("第五讲")
        results.append(uni == 1)
        print(f"  [{'OK ' if uni == 1 else 'BAD'}] 第五讲去重 = {uni}（期望 1，同一题错两次算一道）")

    print("=" * 60)
    print(f"通过 {sum(results)} / {len(results)}")
    if all(results):
        print("全部通过。可以接着加服务端点 /wrong 了。")
    else:
        print("把上面标 BAD / 未实现的补齐，再跑一次。")


if __name__ == "__main__":
    main()
