import os
import shutil
import threading
import json
import uuid
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch

from pvcore.shared.config import device
from pvcore.shared.utils.processor import preprocess_image, preprocess_video
from pvcore.models.model_VisionTriX import get_model as get_visiontrix_model
from pvcore.models.model_MobileNetV3 import get_model as get_mobilenetv3_model

CLASS_NAMES = {0: "REAL", 1: "FAKE"}

BASE_DIR = Path(__file__).resolve().parents[2]  
HISTORY_FILE = BASE_DIR.joinpath("prediction_history.json")
HISTORY_LOCK = threading.Lock()
MAX_HISTORY = 2000  
_HISTORY = []  


def load_history_from_disk():
    global _HISTORY
    try:
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    _HISTORY = data[-MAX_HISTORY:]
    except Exception as e:
        print("Warning: cannot load history file:", e)


def save_history_to_disk():
    try:
        tmp = HISTORY_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_HISTORY[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)
        tmp.replace(HISTORY_FILE)
    except Exception as e:
        print("Warning: cannot save history file:", e)


def add_history_record(record: dict):
    with HISTORY_LOCK:
        _HISTORY.append(record)
        # keep bounded size
        if len(_HISTORY) > MAX_HISTORY:
            del _HISTORY[0 : len(_HISTORY) - MAX_HISTORY]
        save_history_to_disk()


def get_history(limit: int = 100):
    with HISTORY_LOCK:
        if limit is None:
            return list(_HISTORY)
        return list(_HISTORY[-int(limit) :])


def find_history_by_id(entry_id: str):
    with HISTORY_LOCK:
        for r in reversed(_HISTORY):  
            if r.get("id") == entry_id:
                return r
    return None


def clear_history():
    with HISTORY_LOCK:
        _HISTORY.clear()
        save_history_to_disk()


load_history_from_disk()

def load_models():
    print("Loading models...")

    models = {
        "VisionTriX": get_visiontrix_model().to(device),
        "MobileNetV3": get_mobilenetv3_model().to(device),
    }

    for name, m in models.items():
        m.eval()
        print(f"   ➤ Loaded: {name}")

    print(" All models loaded!\n")
    return models


MODELS = load_models()
AVAILABLE_MODELS = list(MODELS.keys())



app = FastAPI(
    title="Face Anti-Spoofing Multi-Model API",
    description="Upload image/video + choose model to predict REAL/FAKE",
    version="1.0.0",
)


def predict(model, frames_tensor, mask):
    """Trả về class_id, class_name, score (probability) nếu có."""
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        mask = mask.to(device)

        try:
            outputs = model(frames_tensor, mask)
        except TypeError:
            outputs = model(frames_tensor)

        try:
            probs = torch.softmax(outputs, dim=1)
            cls = torch.argmax(probs, dim=1).item()
            score = float(probs[0, cls].item())
        except Exception:
            cls = int(torch.argmax(outputs, dim=1).item()) if outputs is not None else 0
            score = None

        return cls, CLASS_NAMES[cls], score

def _get_model_or_error(model_name: str):
    """Lấy model theo tên, nếu sai tên thì trả JSONResponse lỗi."""
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    return MODELS[model_name]

@app.get("/")
def home():
    return {
        "message": "Face Anti-Spoofing API Running!",
        "models_available": AVAILABLE_MODELS,
    }


@app.get("/models")
def list_models():
    """Trả về danh sách các model có thể chọn."""
    return {"models": AVAILABLE_MODELS}


@app.get("/history")
def api_get_history(limit: int = 100):
    """Trả về lịch sử các dự đoán (mặc định limit 100 mới nhất)."""
    try:
        limit = int(limit) if limit is not None else None
    except Exception:
        limit = 100
    return {"count": len(_HISTORY), "items": get_history(limit)}


@app.get("/history/{entry_id}")
def api_get_history_entry(entry_id: str):
    item = find_history_by_id(entry_id)
    if not item:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return item


@app.delete("/history")
def api_clear_history():
    clear_history()
    return {"status": "ok", "message": "history cleared"}


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("VisionTriX"),  
):
    """
    Upload 1 ảnh + chọn model → trả về REAL/FAKE

    - model_name: "VisionTriX" hoặc "MobileNetV3"
    """
    temp_path = f"temp_{file.filename}"
    start_ts = time.perf_counter()
    try:
        model = _get_model_or_error(model_name)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        frames_tensor, mask, _, _ = preprocess_image(temp_path)
        cls_id, cls_name, score = predict(model, frames_tensor, mask)

        duration_ms = int((time.perf_counter() - start_ts) * 1000)

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "image",
            "filename": file.filename,
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "duration_ms": duration_ms,
        }
        add_history_record(record)

        return {
            "status": "success",
            "type": "image",
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "duration_ms": duration_ms,
        }

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    model_name: str = Form("VisionTriX"),  
):
    """
    Upload video + chọn model → trả về REAL/FAKE

    - model_name: "VisionTriX" hoặc "MobileNetV3"
    """
    temp_path = f"temp_{file.filename}"
    start_ts = time.perf_counter()
    try:
        model = _get_model_or_error(model_name)

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        frames_tensor, mask, _, _, total_frames = preprocess_video(temp_path)
        cls_id, cls_name, score = predict(model, frames_tensor, mask)

        duration_ms = int((time.perf_counter() - start_ts) * 1000)

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "video",
            "filename": file.filename,
            "frames": total_frames,
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "duration_ms": duration_ms,
        }
        add_history_record(record)

        return {
            "status": "success",
            "type": "video",
            "frames": total_frames,
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": score,
            "duration_ms": duration_ms,
        }

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
