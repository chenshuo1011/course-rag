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

# 两个模块共用的模型封装（scripts/llm.py）不在本目录，
# 把它所在目录加进 sys.path，避免复制一份出来各改各的。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import llm

BASE = Path(r"D:/visdrone/VisDrone2019-DET-val")
WEIGHT = Path(r"D:/visdrone/yolov8n_visdrone.pt")
IMGSZ = 1280
CONF = 0.25
SURE_CONF = 0.5   # 高于它才算"能确认"，低于它只说"可能"

# 两套权重，按场景选。
#
# 为什么要两套：VisDrone 权重只见过"白天 + 正射俯视 + 目标十几个像素"的图。
# 实测一张夜景斜视照片（摩托约 200 像素），VisDrone 权重只检出 2 个人、0 辆摩托；
# 换 COCO 通用权重同一张图检出 5 辆摩托 + 5 个人。
# 尺寸差 5~20 倍是最致命的：模型没见过这么大的目标。
SCENES = {
    "aerial": (Path(r"D:/visdrone/yolov8n_visdrone.pt"), IMGSZ,
               "航拍俯视：只认 VisDrone 的 10 类（行人、人群、自行车、汽车、"
               "面包车、卡车、三轮车、遮阳三轮、公交、摩托），目标通常很小"),
    "general": (Path(r"D:/visdrone/yolov8n_coco.pt"), 960,
                "日常场景：COCO 80 类（人、汽车、摩托、自行车、公交、卡车、"
                "船、飞机、火车等），目标较大、视角倾斜"),
}
# 同一类东西在两套权重里名字不同
CANON = {"pedestrian": "person", "people": "person", "motor": "motorcycle",
         "car": "car", "van": "car", "truck": "truck", "bus": "bus",
         "bicycle": "bicycle"}

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
    # 以下只在"日常模式"（COCO 80 类）下有效。
    # 航拍模式下这些类不在 KNOWN 里，会正确拒答"只识别 10 类"。
    # 原来只覆盖了 7 个 COCO 类，导致问"狗""红绿灯"被拒答——模型其实认识。
    "狗": "dog", "小狗": "dog", "猫": "cat", "小猫": "cat", "鸟": "bird",
    "马": "horse", "羊": "sheep", "牛": "cow", "大象": "elephant", "熊": "bear",
    "斑马": "zebra", "长颈鹿": "giraffe",
    "红绿灯": "traffic light", "信号灯": "traffic light",
    "消防栓": "fire hydrant", "停车标志": "stop sign", "限速牌": "stop sign",
    "长椅": "bench", "长凳": "bench", "椅子": "chair", "沙发": "couch",
    "背包": "backpack", "雨伞": "umbrella", "伞": "umbrella", "手提包": "handbag",
    "行李箱": "suitcase", "瓶子": "bottle", "杯子": "cup", "碗": "bowl",
    "书": "book", "手机": "cell phone", "笔记本电脑": "laptop", "键盘": "keyboard",
    "鼠标": "mouse", "电视": "tv", "显示器": "tv", "床": "bed",
    "餐桌": "dining table", "桌子": "dining table", "马桶": "toilet",
    "冰箱": "refrigerator", "微波炉": "microwave", "水槽": "sink",
    "香蕉": "banana", "苹果": "apple", "橙子": "orange", "披萨": "pizza",
    "蛋糕": "cake", "三明治": "sandwich", "剪刀": "scissors", "风筝": "kite",
    "球": "sports ball", "滑板": "skateboard", "冲浪板": "surfboard",
    "棒球棍": "baseball bat", "网球拍": "tennis racket", "滑雪板": "skis",
    # 模型不认识这些（COCO 里也没有），保留是为了能说"不在识别范围"
    "自行车架": "bike rack",
    # 补齐剩下的 COCO 类，让 80 类全部能用中文问到
    "棒球手套": "baseball glove", "西兰花": "broccoli", "胡萝卜": "carrot",
    "钟": "clock", "时钟": "clock", "表": "clock", "甜甜圈": "donut",
    "叉子": "fork", "飞盘": "frisbee", "吹风机": "hair drier",
    "热狗": "hot dog", "刀": "knife", "烤箱": "oven",
    "停车计时器": "parking meter", "盆栽": "potted plant", "绿植": "potted plant",
    "遥控器": "remote", "滑雪单板": "snowboard", "勺子": "spoon",
    "泰迪熊": "teddy bear", "玩具熊": "teddy bear", "领带": "tie",
    "烤面包机": "toaster", "牙刷": "toothbrush", "花瓶": "vase",
    "酒杯": "wine glass", "红酒杯": "wine glass",
}
KNOWN = set(RELIABILITY)        # 航拍模式下能识别的 10 类

# 方位词 → (列, 行)，None 表示该维度不限。
# 只给"左侧"就不限行，给"左上"才限到具体格子——用户说"左边"通常指整条左侧。
ZONE_ALIAS = {
    "左上": ("左", "上"), "中上": ("中", "上"), "右上": ("右", "上"),
    "左中": ("左", "中"), "中中": ("中", "中"), "右中": ("右", "中"),
    "左下": ("左", "下"), "中下": ("中", "下"), "右下": ("右", "下"),
    "左侧": ("左", None), "左边": ("左", None), "左半边": ("左", None),
    "左侧半边": ("左", None), "左方": ("左", None),
    "右侧": ("右", None), "右边": ("右", None), "右半边": ("右", None),
    "右方": ("右", None),
    # "中间"在中文里通常指中央那一格，不是"整张图"。
    # 整图用 count（不传方位）表达，所以这里映射到中中。
    "中间": ("中", "中"), "中央": ("中", "中"), "正中": ("中", "中"),
    "正中间": ("中", "中"), "中心": ("中", "中"),
    "全场": (None, None), "全图": (None, None), "整张图": (None, None),
    "上半": (None, "上"), "上方": (None, "上"), "上边": (None, "上"),
    "上部": (None, "上"), "顶部": (None, "上"),
    "下半": (None, "下"), "下方": (None, "下"), "下边": (None, "下"),
    "下部": (None, "下"), "底部": (None, "下"),
    "上方中间": ("中", "上"), "下方中间": ("中", "下"),
}


def parse_zone(text: str):
    """从中文方位词里解析出 (列, 行)。认不出就返回 (None, None)。"""
    s = (text or "").strip()
    if not s:
        return None, None
    if s in ZONE_ALIAS:
        return ZONE_ALIAS[s]
    # 宽容一点：「画面左上角」「图左下部分」这类
    for k, v in sorted(ZONE_ALIAS.items(), key=lambda kv: -len(kv[0])):
        if k in s:
            return v
    return None, None

# VisDrone 把"行人"和"人群里的其他人"分成两类（pedestrian / people），
# 中文里都是"人"，用户不会这么区分。问"有多少人"时必须两类一起数，
# 否则会出现"图里没有行人"而画面里明明站着一堆人的回答。
PEOPLE_GROUP = {"pedestrian", "people"}

_models = {}


def get_model(scene: str = "aerial"):
    """按场景懒加载。两套权重都不大（各 6MB），但要各自预热一次。"""
    if scene not in _models:
        path, sz, _ = SCENES[scene]
        if not path.exists():
            raise FileNotFoundError(f"缺权重 {path}")
        m = YOLO(str(path))
        first = sorted((BASE / "images").glob("*.jpg"))[0]
        # 预热：首帧含模型图优化，不能算在第一个用户头上
        m.predict(str(first), verbose=False, device="cpu", imgsz=sz)
        _models[scene] = m
    return _models[scene]


class ImgTools:
    """工具实现。坐标用 3x3 网格描述，比"像素坐标"对人有用。"""

    def __init__(self, calibrated: bool = True, scene: str = "aerial"):
        # calibrated=False 表示这张图没有标注可对照（比如用户上传的图）。
        # 那种情况下不能引用 VisDrone 的实测检出率——换了数据分布就不成立，
        # 措辞只能退到"模型检出 N 个"，不能声称"至少/约"。
        self.calibrated = calibrated
        self.scene = scene
        self.cache = {}

    def _detect(self, image_id: str):
        key = (self.scene, image_id)
        if key in self.cache:
            return self.cache[key]
        p = BASE / "images" / f"{image_id}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"没有这张图：{image_id}")
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        _, sz, _ = SCENES[self.scene]
        m = get_model(self.scene)
        r = m.predict(img, verbose=False, device="cpu", imgsz=sz, conf=CONF)[0]
        dets = []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                           r.boxes.conf.tolist()):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            zone = ("左" if cx < w / 3 else "中" if cx < 2 * w / 3 else "右") + \
                   ("上" if cy < h / 3 else "中" if cy < 2 * h / 3 else "下")
            dets.append({"cls": m.names[int(c)], "conf": float(s), "zone": zone})
        self.cache[key] = (dets, w, h)
        return self.cache[key]

    def overview(self, image_id: str) -> dict:
        dets, w, h = self._detect(image_id)
        sure = Counter(d["cls"] for d in dets if d["conf"] >= SURE_CONF)
        lowc = Counter(d["cls"] for d in dets if d["conf"] < SURE_CONF)

        # 必须遍历「确认」和「不确定」两边的并集。
        # 只遍历确认那一侧的话，全部落在 0.25~0.5 的类别会整个消失——
        # 用户会以为图里没有这类目标，比数字偏大严重得多。
        names = set(sure) | set(lowc)
        items = []
        for name in sorted(names, key=lambda k: -(sure.get(k, 0) * 2 + lowc.get(k, 0))):
            n, m = sure.get(name, 0), lowc.get(name, 0)
            ratio, note = RELIABILITY.get(name, (0.5, "未知"))
            it = {"类别": name, "数量": n}
            if m:
                it["不确定数量"] = m
            if n == 0:
                # 这一类只有低置信度框，很可能是误检，措辞必须区分开
                it["可靠度"] = "全部为低置信度，可能是误检"
                it["建议措辞"] = "可能"
            elif not self.calibrated:
                # 通用模式没有实测依据，不能说"至少/约"。
                # 之前这里套用了航拍的检出率表，导致回答里出现"至少 2 个红绿灯"。
                it["可靠度"] = "无实测依据"
                it["建议措辞"] = "检出"
            else:
                it["可靠度"] = note
                it["建议措辞"] = ("可给确数" if ratio > 1.05 else
                                  ("约" if ratio > 0.8 else "至少"))
            items.append(it)

        return {"图片": image_id, "尺寸": f"{w}x{h}", "各类数量": items,
                "总计": sum(sure.values()),
                "不确定总计": sum(lowc.values()),
                "说明": "「数量」只统计置信度 0.5 以上的框；不确定数量是 0.25~0.5 的，"
                        "可能是重复框或误检，回答时单独提，不要并入主数。"
                        "数量为 0 但有不确认数量的类别，说明模型怀疑有、但没把握，"
                        "不能说「没有」。"}

    def _canon(self, name: str) -> str:
        """把中文别名解析出的类别，按当前场景换成模型里的真实类名。"""
        if self.scene == "general":
            return CANON.get(name, name)
        return name

    def _known(self) -> set:
        """当前场景能识别的类名。航拍用实测过的那 10 类；通用直接问模型。"""
        if self.scene == "general":
            return set(get_model("general").names.values())
        return KNOWN

    def count(self, image_id: str, cls: str) -> dict:
        cls_en = self._canon(ALIAS.get(cls.strip(), cls.strip()))
        if cls_en not in self._known():
            # 不认识的类别不能给"0 个"——那会被理解成"看过了，确实没有"，
            # 而实际是"这一类我根本不会看"。
            return {"图片": image_id, "类别": cls_en, "检出数量": None,
                    "建议措辞": "拒答",
                    "说明": f"当前模式只识别这些类别：{'、'.join(sorted(self._known()))}。"
                            f"「{cls}」不在其中，无法判断。"
                            f"回答里应说明识别范围，不能给数字。"}
        dets, _, _ = self._detect(image_id)

        if cls_en in PEOPLE_GROUP and self.scene == "aerial":
            # 两类一起数，同时给构成，回答可以说「共 8 人」
            ped = sum(1 for d in dets if d["cls"] == "pedestrian")
            ppl = sum(1 for d in dets if d["cls"] == "people")
            n = ped + ppl
            if not self.calibrated:
                return {"图片": image_id, "类别": "人（行人+人群）", "检出数量": n,
                        "构成": f"行人 {ped}、人群 {ppl}",
                        "建议措辞": "没有" if n == 0 else "检出",
                        "说明": "上传的图没有标注对照，精度未知，照实说「模型检出 N 人」。"}
            # 合并后的措辞固定用「至少」：两类合计的召回率没有一个有意义的单一数值
            # （行人 93%、人群 58%），取低的那一档更稳妥。
            # 副作用：RELIABILITY 里唯一落在 0.8~1.05 的行人（93%）走不到「约」那条分支，
            # 所以下面单类分支里的「约」目前是不可达的——留着是为了以后加类别时用。
            return {"图片": image_id, "类别": "人（行人+人群）", "检出数量": n,
                    "构成": f"行人 {ped}、人群 {ppl}",
                    "实测检出率": "行人 93%、人群 58%",
                    "可靠度": "两类合计，人群那一类容易漏",
                    "建议措辞": "至少" if n else "没有",
                    "说明": "数据集把「行人」和「人群里的其他人」分成两类，这里合并统计；"
                            "人群那一类实测只检出 58%，所以实际人数可能更多。"}

        # 高置信 / 低置信分开数。
        # 实测教训：一张街景图里"红绿灯 3 个"有 1 个是公交站牌被误判（conf 0.37），
        # 车也有 0.26/0.28 的重复框。不分级就会给出"3 个红绿灯"这种半真半假的答案。
        sure = [d for d in dets if d["cls"] == cls_en and d["conf"] >= SURE_CONF]
        maybe = [d for d in dets if d["cls"] == cls_en and d["conf"] < SURE_CONF]
        n, m = len(sure), len(maybe)
        if not self.calibrated:
            return {"图片": image_id, "类别": cls_en, "检出数量": n,
                    "不确定数量": m,
                    "建议措辞": "没有" if n == 0 else "检出",
                    "说明": "这是上传的图，没有人工标注可对照，所以精度未知。"
                            "措辞用「模型检出 N 个」，不要说「至少」或「约」——"
                            "那需要实测依据，这里没有。"
                            + (f"另有 {m} 个低置信度框，回答里可提「还有 {m} 个不确定」，"
                               f"但不能并入主数。" if m else "")}
        ratio, note = RELIABILITY.get(cls_en, (0.5, "未知"))
        return {"图片": image_id, "类别": cls_en, "检出数量": n,
                "不确定数量": m,
                "实测检出率": f"{ratio:.0%}", "可靠度": note,
                # 0 个时说"约 0 个"不是人话，直接说没有
                "建议措辞": "没有" if n == 0 else
                            ("可给确数" if ratio > 1.05 else
                             ("约" if ratio > 0.8 else "至少")),
                "说明": f"在这份数据集上，这类目标平均只能检出 {ratio:.0%}，"
                        f"所以真实数量可能{'更多' if ratio < 1 else '略少'}。"
                        + (f"另有 {m} 个低置信度框（conf<{SURE_CONF}），"
                           f"可能是同一目标的重复框或误检，不要并入主数。" if m else "")}

    def locate(self, image_id: str, cls: str) -> dict:
        cls_en = self._canon(ALIAS.get(cls.strip(), cls.strip()))
        if cls_en not in self._known():
            return {"图片": image_id, "类别": cls_en, "检出数量": None,
                    "说明": f"「{cls}」不在当前模式的识别范围里"}
        dets, _, _ = self._detect(image_id)
        if self.scene == "aerial" and cls_en in PEOPLE_GROUP:
            want = PEOPLE_GROUP
        else:
            want = {cls_en}
        sub = [d for d in dets if d["cls"] in want]
        if not sub:
            return {"图片": image_id, "类别": cls_en, "检出数量": 0,
                    "说明": "没检出这类目标"}
        z = Counter(d["zone"] for d in sub)
        return {"图片": image_id, "类别": cls_en, "检出数量": len(sub),
                "方位分布": [{"方位": k, "数量": v} for k, v in z.most_common()]}

    def count_in(self, image_id: str, cls: str, zone: str) -> dict:
        """只数某个方位里的目标。看图的人天然会分区提问（"左边那排有几辆车"）。"""
        cls_en = self._canon(ALIAS.get(cls.strip(), cls.strip()))
        if cls_en not in self._known():
            return {"图片": image_id, "类别": cls_en, "检出数量": None,
                    "建议措辞": "拒答",
                    "说明": f"「{cls}」不在当前模式的识别范围里"}
        want = PEOPLE_GROUP if (self.scene == "aerial" and cls_en in PEOPLE_GROUP) else {cls_en}
        dets, _, _ = self._detect(image_id)
        col, row = parse_zone(zone)
        if col is None and row is None and zone.strip() not in (
                "全场", "全图", "整张图"):
            return {"图片": image_id, "类别": cls_en, "建议措辞": "重问",
                    "说明": f"认不出方位「{zone}」，可用：左侧/右边/上方/下方/中间，"
                            f"或九宫格（左上、中上、右下…）。要整张图的数就直接用 count。"}
        sub = [d for d in dets
               if d["cls"] in want
               and (col is None or d["zone"][0] == col)
               and (row is None or d["zone"][1] == row)]
        sure = sum(1 for d in sub if d["conf"] >= SURE_CONF)
        maybe = len(sub) - sure
        whole = sum(1 for d in dets if d["cls"] in want
                    and d["conf"] >= SURE_CONF)
        return {"图片": image_id, "类别": cls_en, "方位": zone,
                "检出数量": sure, "不确定数量": maybe,
                "全图同类别数量": whole,
                "建议措辞": "没有" if sure == 0 else
                            ("检出" if not self.calibrated else "可给确数"),
                "说明": f"只数了「{zone}」这一片。全图这类目标共 {whole} 个；"
                        f"方位按九宫格划分（画面三等分）。"}


TOOLS = {
    "overview": {
        "desc": "看一眼这张图里都有什么，返回各类目标的数量。问「图里有什么」「都有哪些东西」用这个。",
        "args": {"image_id": "图片名，不带路径也不带 .jpg，原样照抄问题里给的"},
    },
    "count": {
        "desc": "数某个类别有多少个。问「有几辆车」「多少人」用这个。"
                "返回里带「可靠度」和「建议措辞」，回答时必须照着措辞来。",
        "args": {"image_id": "图片名", "cls": "类别，中文即可：车/人/公交/卡车/摩托/三轮/货车/面包车/自行车"},
    },
    "locate": {
        "desc": "问某类目标**分布在哪些方位**（返回九宫格分布）。问「在哪一片」「都停在哪」用这个。",
        "args": {"image_id": "图片名", "cls": "类别，中文即可"},
    },
    "count_in": {
        "desc": "**只数某个方位**里的目标。问「左边那条路有几辆车」「右下角有几个人」用这个。"
                "方位可给：左侧/右边/上方/下方/中间，或九宫格（左上、中上、右下…）。"
                "注意它和 locate 的区别：locate 给分布，count_in 给这一片的数字。",
        "args": {"image_id": "图片名", "cls": "类别", "zone": "方位，如 左侧、左上、下方"},
    },
}

SYSTEM = """你是遥感图像问答助手，可以调用工具看图。你**不能直接看图**，只看得到工具返回的数据。

工具：
{tools}

每步只输出一个 JSON 对象，不要别的内容。
调用工具：{{"thought":"一句话理由","action":"工具名","args":{{...}}}}
image_id 直接照抄用户消息里「图片：」后面的那一串，不要改、不要加路径、不要加 .jpg。
给出回答：{{"thought":"够了","action":"final","answer":"口语回答，60 字内"}}

规则：
1. 数字**必须来自工具返回**。不许自己估算、不许加总、不许四舍五入到别的数。
2. 工具里有「建议措辞」：写「可给确数」就说「有 26 辆」；写「约」就说「大约 26 辆」；
   写「至少」就说「至少 26 辆」。**不要无视它，也不要自己改口径。**
3. 问数量用 count；问都有什么用 overview。
   问「在哪一片/分布在哪些方位」用 locate（给分布）；
   问「左边有几辆」「右下角有几个人」这种**指定区域的数量**用 count_in。
4. 图和问题无关（比如问天气、问价格）时，final 直接说「图上看不出这个」。
   工具返回「建议措辞：拒答」时，说明这一类不在识别范围内，
   要如实说「我只认 VisDrone 的那 10 类，看不出有没有X」，**不要给 0**。
5. 一个问题里问两类目标，就分两步各调一次 count（一次只输出一个 JSON，不要一条消息里写两个动作），不要用 overview 代替。
6. 数量是 0（或建议措辞是「没有」）时直接说「没有」，**不要写「大约 0 个」**，那不是人话。
7. 问「人」时工具返回的是行人+人群的合并数，必要时可以说出构成。
8. 工具返回里有「不确定数量」时，那是低置信度的框（<0.5），可能是重复或误检。
   可以提一句「另有 N 个不确定」，**但绝不能加进主数**。
9. 建议措辞是「检出」时，说明这张图没有标注对照，只能说「模型检出 N 个」，
   不要加「至少/约」，也不要说"这类目标检出率多少"——那是另一个数据集上的数字。
10. 对话历史里是之前几轮说过的话，追问可以引用其中的数字和结论
    （「比刚才多了吗」「那左边呢」）。但**历史里没提过的信息不许编**，
    可以直说「上文没提这个」。"""


def build_system(scene: str = "aerial") -> str:
    lines = []
    for name, t in TOOLS.items():
        args = "；".join(f"{k}: {v}" for k, v in t["args"].items())
        lines.append(f"- {name}({args})\n  {t['desc']}")
    # 场景不同，能识别的类别和措辞依据都不同，必须写进提示词，
    # 否则通用模式下模型会继续套用 VisDrone 的检出率说法
    note = SCENES[scene][2]
    measured = ("" if scene != "aerial" else
                "这是航拍模式，可以用实测检出率来措辞（工具返回里会带）。")
    return SYSTEM.format(tools="\n".join(lines)) + f"\n\n当前场景：{note}。{measured}"


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
        calibrated: bool = True, scene: str = "aerial",
        history: list[dict] | None = None) -> dict:
    """history: [{"q": 上一轮问题, "a": 上一轮回答}, ...]，最多取最近 3 轮。

    不存会话，由前端把最近几轮带上来——服务端仍然无状态。
    这样「比刚才多了还是少了」「那左边呢」能接上，而不必引入 session 管理。
    """
    tools = ImgTools(calibrated=calibrated, scene=scene)
    messages = [{"role": "system", "content": build_system(scene)}]

    # 把历史按协议格式回放：user 提问 → assistant 的 final 动作。
    # 这样模型看到的对话形状和它自己产出的形状一致，不用另教一套。
    for h in (history or [])[-3:]:
        q0, a0 = (h.get("q") or "").strip(), (h.get("a") or "").strip()
        if not q0 or not a0:
            continue
        messages.append({"role": "user", "content": f"图片：{image_id}\n问题：{q0}"})
        messages.append({"role": "assistant", "content": json.dumps(
            {"thought": "已答", "action": "final", "answer": a0}, ensure_ascii=False)})

    messages.append({"role": "user", "content": f"图片：{image_id}\n问题：{question}"})
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
            ans = act.get("answer", "")
            if not calibrated:
                # 没有标注可对照时，不能说"至少 N""约 N"——那是在声称精度。
                # 提示词里写了这条，但实测模型会漏遵守（通用模式下仍说"至少 5 辆"），
                # 所以这里加一道确定性的兜底：只删出现在数字前的限定词，
                # 不碰"约束""大约何时"这类词。
                ans = re.sub(r"(至少|大约|大概|约)\s*(?:有|能看到|能确认|能数出|共)?\s*(?=\d)", "", ans)
            return {"answer": ans, "calls": called, "steps": steps,
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
