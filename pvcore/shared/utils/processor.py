import os
import cv2
import torch
import numpy as np

from decord import VideoReader, cpu
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pvcore.shared.config import device, num_frames  
#python3 -m pvcore.shared.utils.processor

transform = A.Compose([
    A.Resize(224, 224),  
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def preprocess_image(image_path: str):
    """
    Preprocess image cho inference - trả về:
    frames_tensor: (1, T, C, H, W)
    mask:         (1, T)
    image_rgb:    ảnh gốc (RGB)
    original_size: (H, W)
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2] 

    frames = []
    for _ in range(num_frames):
        transformed = transform(image=image_rgb)["image"]  
        frames.append(transformed)
        
    frames_tensor = torch.stack(frames)

    mask = torch.ones(num_frames, dtype=torch.float32)

    frames_tensor = frames_tensor.unsqueeze(0)
    mask = mask.unsqueeze(0)
        
    return frames_tensor, mask, image_rgb, original_size
    

def preprocess_video(video_path: str):
    """
    Preprocess video - trả về:
    frames_tensor: (1, T, C, H, W)
    mask:         (1, T)
    frames:       numpy (T, H, W, C) RGB đã sample
    sample_frame: frame đầu tiên (RGB) hoặc None
    total_frames: tổng số frame của video
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames")
        
    if total_frames > num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) - RGB
    else:
        frames = vr.get_batch(list(range(total_frames))).asnumpy()
        
    sample_frame = frames[0] if len(frames) > 0 else None

    transformed_frames = []
    for frame in frames:
        transformed = transform(image=frame)["image"]
        transformed_frames.append(transformed)

    frames_tensor = torch.stack(transformed_frames)
    mask = torch.ones(num_frames, dtype=torch.float32)
    if len(frames) < num_frames:
        mask[len(frames):] = 0  # phần không có frame thì mask = 0

    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
    mask = mask.unsqueeze(0)                    # (1, T)
        
    return frames_tensor, mask, frames, sample_frame, total_frames

    