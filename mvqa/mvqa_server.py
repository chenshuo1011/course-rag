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
import json
import threading
import uuid
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

import mvqa

# 视觉模型是可选能力：模块缺失或没有 key 时只让 /describe 不可用，
# 计数的 /ask、/detect 必须照常工作。
try:
    import llm_vision
except Exception as _e:          # noqa: BLE001
    llm_vision = None
    _vlm_import_error = f"{type(_e).__name__}: {_e}"
else:
    _vlm_import_error = ""

HERE = Path(__file__).parent
app = FastAPI(title="遥感图像问答")

_models: dict = {}
_model_lock = threading.Lock()
_ready = threading.Event()
# CPU 推理是独占资源：2~3 路并发就会把核吃满，全部变慢。
# 限 2 路，多的排队——比"看起来能并发但大家都慢"好。
_sem = threading.Semaphore(2)

MAX_BOXES = 200
MAX_UPLOADS = 50        # 上传图最多留这么多个，多了删最旧的（原来永不清理，会一直涨）
MAX_QUESTION = 500


def check_scene(scene: str) -> str:
    """非法场景名在入口就挡掉。原来直接 SCENES[scene] 会 KeyError → 500。"""
    if scene not in mvqa.SCENES:
        raise HTTPException(400, f"没有 {scene} 这个场景，可选：{'、'.join(mvqa.SCENES)}")
    return scene


def prune_uploads():
    ups = sorted((mvqa.BASE / "images").glob("up_*.jpg"), key=lambda p: p.stat().st_mtime)
    for p in ups[:-MAX_UPLOADS]:
        try:
            p.unlink()
        except OSError:
            pass


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
    # 注意：这里返回**全量**。计数必须用全量算，
    # 之前先截断到 200 再数，密集场景会把数量封顶在 200。
    out.sort(key=lambda d: -d["conf"])
    return out


@app.on_event("startup")
def _warm():
    # 只预热航拍那套（默认场景）。通用那套等用户第一次选"日常"再加载，
    # 免得每次启动都多等一次。
    pruning = threading.Thread(target=lambda: (prune_uploads(), get_model("aerial")),
                               daemon=True)
    pruning.start()


@app.get("/status")
def status():
    return {"ready": _ready.is_set()}


@app.get("/", response_class=HTMLResponse)
def page():
    return (HERE / "mvqa.html").read_text(encoding="utf-8")


@app.post("/describe")
async def describe(file: UploadFile | None = File(None),
                   image_id: str = Form(""),
                   question: str = Form("用一句话描述这张图的场景，不要数数")):
    """场景描述（不参与计数评测）。

    独立接口，不进 Agent 循环——这样 mvqa.py 的提示词一字不改，
    90/49/10 三套评测的对象保持不变。
    """
    if llm_vision is None:
        raise HTTPException(503, f"场景描述不可用：{_vlm_import_error or '模块未加载'}")
    if not llm_vision.available():
        raise HTTPException(503, "场景描述不可用：没有配置 VLM_API_KEY")

    if file is not None and image_id:
        raise HTTPException(400, "上传图和 image_id 只能给一个")
    if file is not None:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "这张图读不出来")
        tmp = mvqa.BASE / "images" / f"vlm_{uuid.uuid4().hex[:12]}.jpg"
        cv2.imwrite(str(tmp), img)
        path = tmp
    elif image_id:
        path = mvqa.BASE / "images" / f"{image_id}.jpg"
        if not path.exists():
            raise HTTPException(404, f"没有这张图：{image_id}")
    else:
        raise HTTPException(400, "要么传图，要么给 image_id")

    try:
        r = llm_vision.describe(path, question)
    except Exception as e:                      # noqa: BLE001
        raise HTTPException(502, f"视觉模型调用失败：{e}")
    finally:
        if file is not None:
            try:
                path.unlink()
            except OSError:
                pass

    return {**r, "note": "场景描述由通用视觉模型生成，只作参考，不参与计数"}


@app.post("/detect")
async def detect(file: UploadFile | None = File(None),
                 image_id: str = Form(""),
                 scene: str = Form("aerial")):
    """只跑检测。前端"只看框"用，也方便单独验证模型。"""
    check_scene(scene)
    if file is not None and image_id:
        raise HTTPException(400, "上传图和 image_id 只能给一个")
    with _sem:
        t_det0 = time.perf_counter()
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

        boxes_all = detect_array(img, scene)
        ms = round((time.perf_counter() - t_det0) * 1000)

    boxes = boxes_all[:MAX_BOXES]
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
              scene: str = Form("aerial"),
              history: str = Form("")):
    """提问。上传图会先存到临时目录，再走和命令行完全相同的 Agent 流程。"""
    check_scene(scene)
    if file is not None and image_id:
        raise HTTPException(400, "上传图和 image_id 只能给一个")
    if len(question) > MAX_QUESTION:
        raise HTTPException(400, f"问题太长了（{len(question)} 字），控制在 {MAX_QUESTION} 字内")

    # 历史由前端带上来，服务端仍无状态：不存 session，也不做清理。
    # 只在内存里解析最近 3 轮，格式不对就当没有，不让脏数据打断问答。
    hist = []
    if history:
        try:
            raw = json.loads(history)
            if isinstance(raw, list):
                hist = [h for h in raw[-3:]
                        if isinstance(h, dict) and h.get("q") and h.get("a")]
        except json.JSONDecodeError:
            hist = []
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
        # 用 uuid 而不是毫秒时间戳：同一毫秒内的两个请求会互相覆盖
        name = f"up_{uuid.uuid4().hex[:12]}"
        cv2.imwrite(str(mvqa.BASE / "images" / f"{name}.jpg"), img)
        prune_uploads()
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
        # 计时从拿到信号量之后开始，排队时间不该算进"检测耗时"
        t_det0 = time.perf_counter()
        det_img = cv2.imread(str(mvqa.BASE / "images" / f"{box_id}.jpg"))
        if det_img is None:
            raise HTTPException(500, f"图片文件读不出来：{box_id}")
        boxes_all = detect_array(det_img, scene)
        det_ms = round((time.perf_counter() - t_det0) * 1000)

    try:
        # 通用模式没有实测依据（我们只在 VisDrone 上量过），一律按"未标定"措辞
        r = mvqa.run(question, target_id, verbose=False,
                     calibrated=(file is None and scene == "aerial"),
                     scene=scene, history=hist)
    except Exception as e:
        raise HTTPException(500, f"问答失败：{type(e).__name__}: {e}")

    counts = {}
    for b in boxes_all:                     # 计数用全量，别用截断后的
        counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    h, w = det_img.shape[:2]
    return {"answer": r["answer"], "calls": r["calls"],
            "boxes": boxes_all[:MAX_BOXES], "counts": counts,
            "boxes_total": len(boxes_all),
            "w": w, "h": h, "detect_ms": det_ms,
            "total_ms": round((time.perf_counter() - t0) * 1000),
            "source": "upload" if file is not None else "dataset"}
