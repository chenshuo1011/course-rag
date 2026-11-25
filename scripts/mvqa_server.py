"""
多模态问答服务（端口 8001）

为什么单独一个服务，不塞进 server.py：
  · 主服务已经加载了 embedder + 重排模型（约 1GB 内存），启动已经 30 秒
  · 检测依赖 torch+ultralytics+cv2，和文本链路没有关系
  · 分开之后，一边挂了另一边照常用；而且 clone 主仓库的人不必装 CV 依赖

接口：
  GET  /              页面（上传图 + 提问 + 画框）
  POST /ask           提问，返回答案 + 检测框 + 耗时
  POST /detect        只跑检测，返回框（页面上"只看检测"用）
  GET  /status        模型是否加载好（前端用来禁用按钮）

一个原则：**框必须回传**。
  只给数字，用户只能选择相信或不信；给框，他能自己核对——这是多模态版的"引用"。

用法：
    "D:\anaconda3\envs\yolo\python.exe" -m uvicorn mvqa_server:app --port 8001
"""

import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO

import mvqa

HERE = Path(__file__).parent
app = FastAPI(title="遥感图像问答")

_models: dict = {}
_model_lock = threading.Lock()
_ready = threading.Event()
# CPU 推理是独占资源：2~3 路并发就会把核吃满，全部变慢。
# 限 2 路，多的排队——比"看起来能并发但大家都慢"好。
_sem = threading.Semaphore(2)

MAX_BOXES = 200


def get_model(scene: str = "aerial") -> YOLO:
    """两套权重各自懒加载。航拍那套启动时预热，通用那套等第一次用到再加载。"""
    m = _models.get(scene)
    if m is None:
        with _model_lock:
            m = _models.get(scene)
            if m is None:
                m = mvqa.get_model(scene)      # 已含预热
                _models[scene] = m
                if scene == "aerial":
                    _ready.set()
    return m


def detect_array(img: np.ndarray, scene: str = "aerial") -> list[dict]:
    """对一张 BGR 图跑检测，返回框列表（像素坐标）。"""
    m = get_model(scene)
    _, sz, _ = mvqa.SCENES[scene]
    r = m.predict(img, verbose=False, device="cpu", imgsz=sz, conf=mvqa.CONF)[0]
    out = []
    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                       r.boxes.conf.tolist()):
        out.append({"cls": m.names[int(c)], "conf": round(float(s), 3),
                    "x1": round(b[0]), "y1": round(b[1]),
                    "x2": round(b[2]), "y2": round(b[3])})
    out.sort(key=lambda d: -d["conf"])
    return out[:MAX_BOXES]


@app.on_event("startup")
def _warm():
    # 只预热航拍那套（默认场景）。通用那套等用户第一次选"日常"再加载，
    # 免得每次启动都多等一次。
    threading.Thread(target=get_model, args=("aerial",), daemon=True).start()


@app.get("/status")
def status():
    return {"ready": _ready.is_set()}


@app.get("/", response_class=HTMLResponse)
def page():
    return (HERE / "mvqa.html").read_text(encoding="utf-8")


class AskIn(BaseModel):
    image_id: str = Field("", description="数据集里的图片名（不含 .jpg）")
    question: str = Field(..., min_length=1, max_length=100)


@app.post("/detect")
async def detect(file: UploadFile | None = File(None),
                 image_id: str = Form(""),
                 scene: str = Form("aerial")):
    """只跑检测。前端"只看框"用，也方便单独验证模型。"""
    with _sem:
        if file is not None:
            data = await file.read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, "这张图读不出来，换个格式试试")
            source = "upload"
        elif image_id:
            p = mvqa.BASE / "images" / f"{image_id}.jpg"
            if not p.exists():
                raise HTTPException(404, f"没有这张图：{image_id}")
            img = cv2.imread(str(p))
            source = "dataset"
        else:
            raise HTTPException(400, "要么传图，要么给 image_id")

        t0 = time.perf_counter()
        boxes = detect_array(img, scene)
        ms = round((time.perf_counter() - t0) * 1000)

    h, w = img.shape[:2]
    counts = {}
    for b in boxes:
        counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    return {"w": w, "h": h, "boxes": boxes, "counts": counts,
            "detect_ms": ms, "source": source}


@app.post("/ask")
async def ask(question: str = Form(...),
              file: UploadFile | None = File(None),
              image_id: str = Form(""),
              scene: str = Form("aerial")):
    """提问。上传图会先存到临时目录，再走和命令行完全相同的 Agent 流程。"""
    if file is not None:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "这张图读不出来")
        # Agent 是按 image_id 从数据集目录找图的，这里把上传图落到临时目录，
        # 保持"同一套流程"，而不是为上传另写一条分支
        # 直接落在 images/ 下，名字带 up_ 前缀区分。
        # 之前放在 images/_upload/ 下、传 "_upload/up_xxx"，模型会把它当路径，
        # 只传 basename 过来，然后报"没有这张图"。
        name = f"up_{int(time.time()*1000)}"
        cv2.imwrite(str(mvqa.BASE / "images" / f"{name}.jpg"), img)
        target_id = name
        box_id = target_id
    else:
        if not image_id:
            raise HTTPException(400, "要么传图，要么给 image_id")
        if not (mvqa.BASE / "images" / f"{image_id}.jpg").exists():
            raise HTTPException(404, f"没有这张图：{image_id}")
        target_id = image_id
        box_id = image_id

    t0 = time.perf_counter()
    with _sem:
        det_img = cv2.imread(str(mvqa.BASE / "images" / f"{box_id}.jpg"))
        boxes = detect_array(det_img, scene)
        det_ms = round((time.perf_counter() - t0) * 1000)

    try:
        # 通用模式没有实测依据（我们只在 VisDrone 上量过），一律按"未标定"措辞
        r = mvqa.run(question, target_id, verbose=False,
                     calibrated=(file is None and scene == "aerial"),
                     scene=scene)
    except Exception as e:
        raise HTTPException(500, f"问答失败：{type(e).__name__}: {e}")

    counts = {}
    for b in boxes:
        counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    h, w = det_img.shape[:2]
    return {"answer": r["answer"], "calls": r["calls"],
            "boxes": boxes[:MAX_BOXES], "counts": counts,
            "w": w, "h": h, "detect_ms": det_ms,
            "total_ms": round((time.perf_counter() - t0) * 1000),
            "source": "upload" if file is not None else "dataset"}
