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

DENSE = "0000280_03401_d_0000629"     # 车 51、人 8、摩托 10
CYCLE = "0000001_03999_d_0000007"     # 自行车 3
TRI = "0000001_07999_d_0000012"       # 三轮 2
TRUCK = "0000076_02142_d_0000009"     # 卡车 4

# (图片, 类别, 场景, 是否已标定, 备注)
CASES = [
    # A 组：航拍 + 已标定，覆盖高/中/低召回
    (DENSE, "车", "aerial", True, "检出 113%"),
    (DENSE, "行人", "aerial", True, "检出 93%"),
    (DENSE, "摩托", "aerial", True, "检出 80%"),
    (DENSE, "面包车", "aerial", True, "检出 69%"),
    (DENSE, "公交", "aerial", True, "检出 67%"),
    (DENSE, "卡车", "aerial", True, "检出 59%"),
    (DENSE, "人", "aerial", True, "两类合并"),
    (TRI, "行人", "aerial", True, "合并后 47 个，大数字"),
    (CYCLE, "自行车", "aerial", True, "检出 48%"),
    (TRI, "三轮车", "aerial", True, "检出 67%"),
    (TRUCK, "卡车", "aerial", True, "检出 59%"),
    (CYCLE, "卡车", "aerial", True, "可能为 0"),
    (TRUCK, "自行车", "aerial", True, "可能为 0"),
    # B 组：通用模式 / 未标定，不能说"至少"
    (DENSE, "车", "general", False, "未标定"),
    (CYCLE, "车", "general", False, "未标定"),
    (TRUCK, "人", "general", False, "未标定"),
    (TRI, "摩托", "general", False, "未标定"),
    (DENSE, "卡车", "general", False, "未标定"),
    # C 组：不在标签空间，必须拒答
    (DENSE, "帆船", "aerial", True, "不在 10 类"),
    (DENSE, "火车", "general", False, "COCO 有 train，应能答"),
    (CYCLE, "飞机", "aerial", True, "不在 10 类"),
]

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
        ok = near_number(answer, HEDGE_LOWER)
        return ok, "应出现「至少/可能/以上」" if not ok else ""
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
