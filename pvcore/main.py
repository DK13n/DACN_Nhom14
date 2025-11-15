import os
import shutil

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch

from pvcore.api.routers import app 
path_MobileNetV3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights", "MobileNetV3.pth")
path_VisionTriX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights", "Hybrid-CDCN-ResViT.pth")

if os.path.exists(path_MobileNetV3) and os.path.exists(path_VisionTriX):
    print("✅ Tất cả model đã tồn tại, bỏ qua bước build_model.")
else: 
    from pvcore.models.down_model import build_model
    build_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    #uvicorn pvcore.main:app --host 0.0.0.0 --port 8000 --reload
    #http://127.0.0.1:8000