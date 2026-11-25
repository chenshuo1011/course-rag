"""
多模态问答层：把检测器当工具，让 LLM 自己决定调什么

和题库 Agent 用同一套协议（每步输出一个 JSON 动作），只是工具换成了图像相关的。

三个工具：
  overview  全类别计数，用于"图里都有什么"
  count     单个类别的数量，用于"有几辆车"
  locate    某类别在画面哪个方位，用于"车都停在哪"

一个关键设计：**工具返回里带可靠度**。
  我们实测过每类的检出率（VisDrone 548 张）：
    car 113%、pedestrian 93%、people 58%、bus 67%……
  把这些数字编码进返回，模型据此决定用"有 26 辆"还是"大约 26 辆"或"至少 20 辆"。
  这是让系统"知道自己有多准"——大多数 demo 缺的就是这一步。

用法：
    "D:\anaconda3\envs\yolo\python.exe" mvqa.py "这张图里有多少辆车"
    "D:\anaconda3\envs\yolo\python.exe" mvqa.py "人主要在哪一片" --img 0000001_02999_d_0000005
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import llm

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
IMGSZ = 1280
CONF = 0.25

# 实测检出率（n@1280，conf 0.25，VisDrone val 548 张）
# 用途：让模型知道哪类能说确数、哪类只能说"至少"
RELIABILITY = {
    "car": (1.13, "基本可信，略偏多"),
    "pedestrian": (0.93, "略偏少"),
    "motor": (0.80, "偏少"),
    "tricycle": (0.67, "明显偏少"),
    "bus": (0.67, "明显偏少"),
    "van": (0.69, "明显偏少"),
    "truck": (0.59, "明显偏少"),
    "bicycle": (0.48, "只有一半，不要当准数"),
    "awning-tricycle": (0.40, "只有四成，不要当准数"),
    "people": (0.58, "只有六成，不要当准数"),
}
ALIAS = {  # 中文说法 → 类别
    "车": "car", "汽车": "car", "轿车": "car", "小汽车": "car",
    "人": "pedestrian", "行人": "pedestrian",
    "人群": "people", "群众": "people",
    "摩托": "motor", "摩托车": "motor", "电动车": "motor",
    "三轮": "tricycle", "三轮车": "tricycle",
    "公交": "bus", "公交车": "bus", "大巴": "bus",
    "货车": "truck", "卡车": "truck",
    "面包车": "van", "客车": "van",
    "自行车": "bicycle",
    "棚车": "awning-tricycle", "遮阳三轮": "awning-tricycle",
    # 以下类别模型**不认识**（只训练了 VisDrone 的 10 类）。
    # 加进来是为了能把"帆船""火车"这类问题正确定位到"不在标签空间内"，
    # 而不是当成"数量为 0 的目标"给个数字。
    "船": "boat", "帆船": "boat", "轮船": "boat", "舰船": "boat",
    "火车": "train", "列车": "train", "飞机": "airplane",
}
KNOWN = set(RELIABILITY)        # 模型真正能识别的 10 类

# VisDrone 把"行人"和"人群里的其他人"分成两类（pedestrian / people），
# 中文里都是"人"，用户不会这么区分。问"有多少人"时必须两类一起数，
# 否则会出现"图里没有行人"而画面里明明站着一堆人的回答。
PEOPLE_GROUP = {"pedestrian", "people"}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHT))
        first = sorted((BASE / "images").glob("*.jpg"))[0]
        _model.predict(str(first), verbose=False, device="cpu", imgsz=IMGSZ)
    return _model


class ImgTools:
    """工具实现。坐标用 3x3 网格描述，比"像素坐标"对人有用。"""

    def __init__(self, calibrated: bool = True):
        # calibrated=False 表示这张图没有标注可对照（比如用户上传的图）。
        # 那种情况下不能引用 VisDrone 的实测检出率——换了数据分布就不成立，
        # 措辞只能退到"模型检出 N 个"，不能声称"至少/约"。
        self.calibrated = calibrated
        self.cache = {}

    def _detect(self, image_id: str):
        if image_id in self.cache:
            return self.cache[image_id]
        p = BASE / "images" / f"{image_id}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"没有这张图：{image_id}")
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = get_model().predict(str(p), verbose=False, device="cpu",
                                imgsz=IMGSZ, conf=CONF)[0]
        m = get_model()
        dets = []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                           r.boxes.conf.tolist()):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            zone = ("左" if cx < w / 3 else "中" if cx < 2 * w / 3 else "右") + \
                   ("上" if cy < h / 3 else "中" if cy < 2 * h / 3 else "下")
            dets.append({"cls": m.names[int(c)], "conf": float(s), "zone": zone})
        self.cache[image_id] = (dets, w, h)
        return self.cache[image_id]

    def overview(self, image_id: str) -> dict:
        dets, w, h = self._detect(image_id)
        c = Counter(d["cls"] for d in dets)
        items = []
        for name, n in c.most_common():
            ratio, note = RELIABILITY.get(name, (0.5, "未知"))
            items.append({"类别": name, "数量": n, "可靠度": note,
                          "建议措辞": "可给确数" if ratio > 1.05 else
                                     ("约" if ratio > 0.8 else "至少")})
        return {"图片": image_id, "尺寸": f"{w}x{h}", "各类数量": items,
                "总计": len(dets)}

    def count(self, image_id: str, cls: str) -> dict:
        cls_en = ALIAS.get(cls.strip(), cls.strip())
        if cls_en not in KNOWN:
            # 模型只认 VisDrone 的 10 类。问"有帆船吗"时不能给"0 艘"——
            # 那会被理解成"看过了，确实没有"，而实际是"这一类我根本不会看"。
            return {"图片": image_id, "类别": cls_en, "检出数量": None,
                    "建议措辞": "拒答",
                    "说明": f"模型只识别这 10 类：{'、'.join(KNOWN)}。"
                            f"「{cls}」不在其中，无法判断。"
                            f"回答里应说明识别范围，不能给数字。"}
        dets, _, _ = self._detect(image_id)

        if cls_en in PEOPLE_GROUP:
            # 两类一起数，同时给构成，回答可以说「共 8 人」
            ped = sum(1 for d in dets if d["cls"] == "pedestrian")
            ppl = sum(1 for d in dets if d["cls"] == "people")
            n = ped + ppl
            if not self.calibrated:
                return {"图片": image_id, "类别": "人（行人+人群）", "检出数量": n,
                        "构成": f"行人 {ped}、人群 {ppl}",
                        "建议措辞": "没有" if n == 0 else "检出",
                        "说明": "上传的图没有标注对照，精度未知，照实说「模型检出 N 人」。"}
            return {"图片": image_id, "类别": "人（行人+人群）", "检出数量": n,
                    "构成": f"行人 {ped}、人群 {ppl}",
                    "实测检出率": "行人 93%、人群 58%",
                    "可靠度": "两类合计，人群那一类容易漏",
                    "建议措辞": "至少" if n else "没有",
                    "说明": "数据集把「行人」和「人群里的其他人」分成两类，这里合并统计；"
                            "人群那一类实测只检出 58%，所以实际人数可能更多。"}

        n = sum(1 for d in dets if d["cls"] == cls_en)
        if not self.calibrated:
            return {"图片": image_id, "类别": cls_en, "检出数量": n,
                    "建议措辞": "没有" if n == 0 else "检出",
                    "说明": "这是上传的图，没有人工标注可对照，所以精度未知。"
                            "措辞用「模型检出 N 个」，不要说「至少」或「约」——"
                            "那需要实测依据，这里没有。"}
        ratio, note = RELIABILITY.get(cls_en, (0.5, "未知"))
        return {"图片": image_id, "类别": cls_en, "检出数量": n,
                "实测检出率": f"{ratio:.0%}", "可靠度": note,
                # 0 个时说"约 0 个"不是人话，直接说没有
                "建议措辞": "没有" if n == 0 else
                            ("可给确数" if ratio > 1.05 else
                             ("约" if ratio > 0.8 else "至少")),
                "说明": f"在这份数据集上，这类目标平均只能检出 {ratio:.0%}，"
                        f"所以真实数量可能{'更多' if ratio < 1 else '略少'}。"}

    def locate(self, image_id: str, cls: str) -> dict:
        cls_en = ALIAS.get(cls.strip(), cls.strip())
        if cls_en not in KNOWN:
            return {"图片": image_id, "类别": cls_en, "检出数量": None,
                    "说明": f"「{cls}」不在模型的识别范围（只认 VisDrone 的 10 类）"}
        dets, _, _ = self._detect(image_id)
        want = PEOPLE_GROUP if cls_en in PEOPLE_GROUP else {cls_en}
        sub = [d for d in dets if d["cls"] in want]
        if not sub:
            return {"图片": image_id, "类别": cls_en, "检出数量": 0,
                    "说明": "没检出这类目标"}
        z = Counter(d["zone"] for d in sub)
        return {"图片": image_id, "类别": cls_en, "检出数量": len(sub),
                "方位分布": [{"方位": k, "数量": v} for k, v in z.most_common()]}


TOOLS = {
    "overview": {
        "desc": "看一眼这张图里都有什么，返回各类目标的数量。问「图里有什么」「都有哪些东西」用这个。",
        "args": {"image_id": "图片名（文件名去掉 .jpg）"},
    },
    "count": {
        "desc": "数某个类别有多少个。问「有几辆车」「多少人」用这个。"
                "返回里带「可靠度」和「建议措辞」，回答时必须照着措辞来。",
        "args": {"image_id": "图片名", "cls": "类别，中文即可：车/人/公交/卡车/摩托/三轮/货车/面包车/自行车"},
    },
    "locate": {
        "desc": "问某类目标在画面哪个方位（九宫格：左上/中上/右下…）。问「在哪一片」「都停在哪」用这个。",
        "args": {"image_id": "图片名", "cls": "类别，中文即可"},
    },
}

SYSTEM = """你是遥感图像问答助手，可以调用工具看图。你**不能直接看图**，只看得到工具返回的数据。

工具：
{tools}

每步只输出一个 JSON 对象，不要别的内容。
调用工具：{{"thought":"一句话理由","action":"工具名","args":{{...}}}}
给出回答：{{"thought":"够了","action":"final","answer":"口语回答，60 字内"}}

规则：
1. 数字**必须来自工具返回**。不许自己估算、不许加总、不许四舍五入到别的数。
2. 工具里有「建议措辞」：写「可给确数」就说「有 26 辆」；写「约」就说「大约 26 辆」；
   写「至少」就说「至少 26 辆」。**不要无视它，也不要自己改口径。**
3. 问数量用 count；问都有什么用 overview；问在哪一片用 locate。
4. 图和问题无关（比如问天气、问价格）时，final 直接说「图上看不出这个」。
   工具返回「建议措辞：拒答」时，说明这一类不在识别范围内，
   要如实说「我只认 VisDrone 的那 10 类，看不出有没有X」，**不要给 0**。
5. 一个问题里问两类目标，就分两步各调一次 count（一次只输出一个 JSON，不要一条消息里写两个动作），不要用 overview 代替。
6. 数量是 0（或建议措辞是「没有」）时直接说「没有」，**不要写「大约 0 个」**，那不是人话。
7. 问「人」时工具返回的是行人+人群的合并数，必要时可以说出构成。
8. 建议措辞是「检出」时，说明这张图没有标注对照，只能说「模型检出 N 个」，
   不要加「至少/约」，也不要说"这类目标检出率多少"——那是另一个数据集上的数字。"""


def build_system() -> str:
    lines = []
    for name, t in TOOLS.items():
        args = "；".join(f"{k}: {v}" for k, v in t["args"].items())
        lines.append(f"- {name}({args})\n  {t['desc']}")
    return SYSTEM.format(tools="\n".join(lines))


def _first_json(s: str):
    """取第一个**括号配对**的 JSON 对象。

    不能用正则 r"\{.*\}"：问"车和摩托各多少"时模型常常连着输出两个 JSON，
    贪心匹配会把两个拼在一起，json.loads 直接失败，整轮就空转了。
    """
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def parse_action(text: str) -> dict:
    s = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    obj = _first_json(s)
    if obj is not None:
        return obj
    raise ValueError(f"解析不了：{text[:120]}")


def run(question: str, image_id: str, max_steps: int = 6, verbose: bool = True,
        calibrated: bool = True) -> dict:
    tools = ImgTools(calibrated=calibrated)
    messages = [{"role": "system", "content": build_system()},
                {"role": "user", "content": f"图片：{image_id}\n问题：{question}"}]
    called, steps = [], []

    for _ in range(max_steps):
        raw = llm.chat_msgs(messages, max_tokens=300)
        try:
            act = parse_action(raw)
        except ValueError:
            break
        if verbose:
            print(f"  [{act.get('action')}] {act.get('thought','')}")
        if act.get("action") == "final":
            return {"answer": act.get("answer", ""), "calls": called, "steps": steps,
                    "image": image_id, "question": question}

        name, args = act.get("action"), act.get("args", {}) or {}
        if name not in TOOLS:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"没有 {name} 这个工具。"})
            continue
        try:
            result = getattr(tools, name)(**args)
        except Exception as e:
            result = {"错误": f"{type(e).__name__}: {e}"}
        called.append({"tool": name, "args": args})
        steps.append({"tool": name, "args": args, "result": result})
        if verbose:
            print(f"     → {json.dumps(result, ensure_ascii=False)[:160]}")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": "工具返回：" + json.dumps(result, ensure_ascii=False)})

    return {"answer": "（没能在限定步数内得出答案）", "calls": called, "steps": steps,
            "image": image_id, "question": question}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    img = "0000001_02999_d_0000005"
    if "--img" in sys.argv:
        img = sys.argv[sys.argv.index("--img") + 1]
        args = [a for a in args if a != img]
    q = " ".join(args) or "这张图里有多少辆车"
    print(f"图：{img}\n问：{q}\n")
    r = run(q, img)
    print(f"\n答：{r['answer']}")
    print(f"调用：{r['calls']}")
