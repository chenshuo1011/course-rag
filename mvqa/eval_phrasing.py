"""
措辞评测 20 条：模型有没有照着「建议措辞」说话

为什么单独测这个：
  功能评测（eval_mvqa.py）只判工具选对、参数对、数字没编。
  但这一层最特别的设计是"按实测检出率决定措辞"——
  低召回的类别必须说"至少"，没标定的图不能说"至少"，0 个必须说"没有"。
  这些规则错了不会报错，只会悄悄给出让人误判的答案。

判定怎么做的：
  期望措辞**不写死**，而是直接调工具拿它返回的「建议措辞」，再看回答是否遵守。
  这样测的是"模型是否遵守指令"，而不是"我背的数字对不对"。
  另外单独统计各类别的建议措辞分布，万一映射表本身错了也能看出来。

四类判定规则：
  可给确数 回答里不该出现"至少"（那会把偏多的数说成偏少）
  约       回答里要有 约/大约/左右
  至少     回答里要有 至少/可能/以上
  没有     要说没有/没检出，且不能出现"约 0""至少 0"
  拒答     要说明识别范围，不能给数字
  检出     未标定模式，不能出现 至少/大约 接数字

用法：
    "D:\anaconda3\envs\yolo\python.exe" eval_phrasing.py
"""

import re

import mvqa
from eval_common import pick_images

# 类别轮换：每张图问"车"（可给确数）+ 一个轮换类别（多半落在"至少"或"没有"）
ROTATE = ["摩托", "自行车", "卡车", "公交", "面包车", "三轮车", "行人"]


def build_cases():
    """20 张图 × 2 类 = 40 条已标定；加 6 条通用模式、3 条拒答。"""
    picks = pick_images(20)
    cases = []
    for i, (stem, _) in enumerate(picks):
        cases.append((stem, "车", "aerial", True, "检出 113%"))
        cases.append((stem, ROTATE[i % len(ROTATE)], "aerial", True, "轮换类别"))
    # 通用模式：未标定，一律不能说"至少/约"
    for i, (stem, _) in enumerate(picks[:6]):
        cases.append((stem, "车" if i % 2 == 0 else "人", "general", False, "未标定"))
    # 通用模式下"表外类别"——没有实测召回率，必须退回「检出」
    for i, (stem, _) in enumerate(picks[:2]):
        cases.append((stem, ("长椅", "花瓶")[i], "general", False, "表外类别"))

    # 不在标签空间
    cases.append((picks[0][0], "帆船", "aerial", True, "不在 10 类"))
    cases.append((picks[1][0], "飞机", "aerial", True, "不在 10 类"))
    cases.append((picks[2][0], "火车", "general", False, "COCO 有 train"))
    return cases


CASES = build_cases()

HEDGE_APPROX = ("约", "大约", "大概", "左右")
HEDGE_LOWER = ("至少", "可能", "以上", "不止")
ZERO_OK = ("没有", "没检出", "没看到", "未检出", "看不到")
REFUSE_OK = ("看不出", "不在", "识别范围", "只认", "只认识")


def near_number(text: str, words) -> bool:
    """检查答案里有没有"模糊词紧挨数字"的写法。"""
    for w in words:
        # 词后 6 个字符内出现数字，或者数字后 1 个字符内出现词
        if re.search(re.escape(w) + r"[^0-9]{0,6}\d", text):
            return True
        if re.search(r"\d[^0-9]{0,2}" + re.escape(w), text):
            return True
    return False


def check(kind: str, answer: str) -> tuple[bool, str]:
    if kind in ("拒答",):
        ok = any(w in answer for w in REFUSE_OK)
        return ok, "应说明识别范围" if not ok else ""
    if kind == "没有":
        bad = re.search(r"(约|至少)\s*0", answer) is not None
        ok = any(w in answer for w in ZERO_OK) and not bad
        return ok, "应说没有，且不能出现「约 0」" if not ok else ""
    if kind == "可给确数":
        ok = not re.search(r"至少[^0-9]{0,4}\d", answer)
        return ok, "给了确数就不该再说「至少」" if not ok else ""
    if kind == "约":
        ok = near_number(answer, HEDGE_APPROX)
        return ok, "应出现「约/大约/左右」" if not ok else ""
    if kind == "至少":
        # 两种写法都算合规：把限定词放在数字旁（「至少 4 个」），
        # 或在后面补一句（「4 个……实际可能更多」）。语义一样，不该判错。
        ok = near_number(answer, HEDGE_LOWER) or any(
            w in answer for w in ("可能更多", "可能还有", "漏检", "偏少", "不只"))
        return ok, "应说明实际可能更多（至少/可能更多/漏检）" if not ok else ""
    if kind == "检出":
        bad = near_number(answer, HEDGE_APPROX + ("至少",))
        return (not bad), "未标定不能说「约/至少」" if bad else ""
    return True, ""


def main():
    rows = []
    print(f"{'类别':<10}{'图':<26}{'确认/不确定':>12}{'建议':>10}  判定")
    print("-" * 78)
    for img, cls, scene, calib, note in CASES:
        t = mvqa.ImgTools(calibrated=calib, scene=scene)
        try:
            res = t.count(img, cls)
        except Exception as e:
            print(f"{cls:<10}{img[:24]:<26}  工具报错 {e}")
            continue
        advice = res.get("建议措辞", "?")
        n = res.get("检出数量")
        m = res.get("不确定数量", 0)

        if n is None:               # 拒答分支
            counts = "—"
        else:
            counts = f"{n}/{m}"

        q = f"图里有几个{cls}"
        r = mvqa.run(q, img, verbose=False, calibrated=calib, scene=scene)
        ok, why = check(advice, r["answer"])
        mark = "✓" if ok else "✗"
        print(f"{cls:<10}{img[:24]:<26}{counts:>12}{advice:>10}  {mark} {r['answer'][:38]}")
        if not ok:
            print(f"{'':<10}  ↳ {why}")
        rows.append((cls, advice, ok, note))

    n = len(rows)
    print(f"\n总判定 {sum(1 for r in rows if r[2])}/{n}")
    by_adv = {}
    for _, adv, ok, _ in rows:
        a = by_adv.setdefault(adv, [0, 0])
        a[0] += ok
        a[1] += 1
    print("按建议措辞分：")
    for adv, (ok, tot) in sorted(by_adv.items()):
        print(f"  {adv:<10} {ok}/{tot}")


if __name__ == "__main__":
    main()
