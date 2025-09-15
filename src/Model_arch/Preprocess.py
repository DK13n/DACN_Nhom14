import os

import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from mpmath.identification import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.io import read_video, read_image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split

VIDEO_EXTENSIONS = {".mp4",".mov"}

def load_video_tensor(path,frames,transformer=None):
    video,_,info = read_video(path, output_format="TCHW")

    indices = make_indices(video.shape[0],frames)

    idx = torch.as_tensor(sorted(indices))
    video = video.index_select(0,idx)

    if transformer is not None:
        video = transformer(video)

    return video


def make_indices(nums, frames):
    if nums<=0:
        return [0]*frames
    if nums>=frames:
        return torch.linspace(0,nums-1,frames).long().tolist()

    reps = frames - nums
    idx = list(range(nums)) + [nums-1]*reps

    return idx


def load_image_tensor(path,frames,transformer=None):
    img = read_image(path)
    img = img.unsqueeze(0).repeat(frames,1,1,1)

    if transformer is not None:
        img = transformer(img)

    return img


def is_video(path):
    ext = Path(path).suffix.lower()
    return ext in VIDEO_EXTENSIONS

def is_image(path):
    ext = Path(path).suffix.lower()
    return ext in IMG_EXTENSIONS


class MixDataset(Dataset):
    def __init__(
            self,
            path,
            base_dir="",
            frames_size = (224,224),
            frames = 8,
            train_mode = True,
    ):
        self.df = pd.read_csv(path)
        self.for_train = "label" in self.df.columns
        self.base_dir = base_dir
        self.frames = frames

        if train_mode:
            self.transformer = T.Compose([
                T.ToDtype(torch.float32, scale=True),                  # uint8 -> float32 [0..1]
                T.Resize((224, 224), antialias=True),                  # resize toàn tensor theo batch
                T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),
                T.RandomResizedCrop(frames_size,scale=(0.7,1.0)),
                T.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.transformer = [
                T.Resize(frames_size),
                T.CenterCrop(frames_size),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]

        if self.base_dir:
            self.df["abs_path"] = self.df["path"].apply(
                lambda p: p if os.path.isabs(p) else os.path.join(self.base_dir, p)
            )
        else:
            self.df["abs_path"] = self.df["path"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i:int):
        row = self.df.iloc[i]
        path = row["abs_path"]
        uid = str(row["uuid"])


        if is_video(path):
            x = load_video_tensor(path,self.frames,self.transformer)
        elif is_image(path):
            x = load_image_tensor(path,self.frames,self.transformer)
        else:
            raise ValueError(f"Không hỗ trợ loại tệp {Path(path).suffix.lower()}")

        if self.for_train:
            y = int(row["label"])
            return {
            "x": x,                 # (T, C, H, W)
            "y": y,
            "uuid": uid}

        return {
            "x":x,
            "uuid":uid
        }


def collate_train(batch):
    x = torch.stack([b["x"] for b in batch],dim=0)
    y = torch.stack([b["y"] for b in batch],dim=0)
    return {
        "x":x,
        "y":y
    }

def collate_test(batch):
    x = torch.stack([b["x"] for b in batch],dim=0)
    return {"x":x}


def make_train_dataset(
        path,
        base_dir = "",
        frames = 8,
        frames_size = (224,224),
        train_ratio = 0.8,
        seed = 42,
        batch_size = 8,
        num_workers = 4
):
    df = pd.read_csv(path)
    train_idx, val_idx = train_test_split(df.index.values,
                                          test_size=1-train_ratio,
                                          random_state=seed,
                                          stratify=df["label"]
                                          )

    train_csv = df.iloc[train_idx].reset_index(drop=True)
    val_csv = df.iloc[val_idx].reset_index(drop=True)

    tmp_train_csv = "_tmp_train.csv"
    tmp_val_csv = "_tmp_val.csv"
    train_csv.to_csv(tmp_train_csv, index=False)
    val_csv.to_csv(tmp_val_csv, index=False)

    train_ds = MixDataset(
        path=tmp_train_csv,
        base_dir = base_dir,
        frames = frames,
        frames_size=frames_size
    )

    val_ds = MixDataset(
        path=tmp_val_csv,
        base_dir=base_dir,
        frames=frames,
        frames_size=frames_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=True,
        collate_fn=collate_train,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_train,
        persistent_workers=True
    )

    return train_loader,val_loader


if __name__ == "__main__":
    csv_path = r"Data\metadata\metadata\publics_train_metadata.csv"
    base_dir = r"Data\publics_data_train"

    train_loader, val_loader = make_train_dataset(csv_path, base_dir)

    batch = next(iter(train_loader))
    print(batch["x"].shape)  # (B, T, C, H, W)
    print(batch["y"].shape)  # (B,)
