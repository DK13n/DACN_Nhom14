import os
import shutil

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

def load_models():
    print("Loading models...")

    models = {
        "VisionTriX": get_visiontrix_model().to(device),
        "MobileNetV3": get_mobilenetv3_model().to(device),
    }

    for name, m in models.items():
        m.eval()
        print(f"   ➤ Loaded: {name}")

    print("✅ All models loaded!\n")
    return models


MODELS = load_models()
AVAILABLE_MODELS = list(MODELS.keys())



app = FastAPI(
    title="Face Anti-Spoofing Multi-Model API",
    description="Upload image/video + choose model to predict REAL/FAKE",
    version="1.0.0",
)


def predict(model, frames_tensor, mask):
    """Trả về class_id, class_name."""
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        mask = mask.to(device)

        try:
            outputs = model(frames_tensor, mask)
        except TypeError:
            # nếu model không nhận mask
            outputs = model(frames_tensor)

        cls = torch.argmax(outputs, dim=1).item()
        return cls, CLASS_NAMES[cls]


def _get_model_or_error(model_name: str):
    """Lấy model theo tên, nếu sai tên thì trả JSONResponse lỗi."""
    if model_name not in MODELS:
        # có thể custom thêm: gợi ý danh sách model
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


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("VisionTriX"),  # mặc định VisionTriX nếu không truyền
):
    """
    Upload 1 ảnh + chọn model → trả về REAL/FAKE

    - model_name: "VisionTriX" hoặc "MobileNetV3"
    """
    temp_path = f"temp_{file.filename}"
    try:
        model = _get_model_or_error(model_name)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        frames_tensor, mask, _, _ = preprocess_image(temp_path)
        cls_id, cls_name = predict(model, frames_tensor, mask)

        return {
            "status": "success",
            "type": "image",
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
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
    model_name: str = Form("VisionTriX"),  # mặc định VisionTriX
):
    """
    Upload video + chọn model → trả về REAL/FAKE

    - model_name: "VisionTriX" hoặc "MobileNetV3"
    """
    temp_path = f"temp_{file.filename}"
    try:
        model = _get_model_or_error(model_name)

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        frames_tensor, mask, _, _, total_frames = preprocess_video(temp_path)
        cls_id, cls_name = predict(model, frames_tensor, mask)

        return {
            "status": "success",
            "type": "video",
            "frames": total_frames,
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
        }

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

