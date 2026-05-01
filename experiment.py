#!/usr/bin/env python3
"""
Lean experiment runner for ML2 Assignment 2.

Persistent files by design:
  - experiment.py : all code and recipes
  - log.txt       : append-only JSONL experiment log
  - model.pt      : current submission candidate, overwritten

First submission recipe:
  python experiment.py run --recipe supervised_v1

Quick smoke test:
  python experiment.py run --recipe smoke --no-final-all
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except Exception:  # timm is only needed for teacher/distillation recipes.
    timm = None


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "train"
LOG_PATH = ROOT / "log.txt"
MODEL_PATH = ROOT / "model.pt"
NUM_CLASSES = 7

# Measured on provided train split. Stored inside submitted model.
DATA_MEAN = [0.5309596, 0.43856254, 0.3481864]
DATA_STD = [0.23571, 0.24416329, 0.23907928]


RECIPES: Dict[str, dict] = {
    "smoke": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 2,
        "final_epochs": None,
        "batch_size": 32,
        "input_size": 128,
        "lr": 3e-3,
        "min_lr": 1e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "mixup_alpha": 0.2,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "micro_mobilenet",
        "dropout": 0.25,
        "notes": "2-epoch pipeline sanity check",
    },
    "supervised_v1": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 90,
        "final_epochs": None,  # if None, retrain-all uses best_epoch from validation phase
        "batch_size": 32,
        "input_size": 128,
        "lr": 3e-3,
        "min_lr": 1e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "mixup_alpha": 0.2,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "strong",
        "model": "micro_mobilenet",
        "dropout": 0.25,
        "notes": "MobileNetV2-style student, supervised only",
    },
    "first_submit": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 70,
        "final_epochs": None,
        "batch_size": 32,
        "input_size": 128,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "plain_cnn",
        "dropout": 0.35,
        "notes": "first submission candidate: conservative plain CNN, basic augmentation, final retrain on all labels",
    },
    "kd_v1_t4": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 70,
        "batch_size": 32,
        "input_size": 128,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "plain_cnn",
        "dropout": 0.35,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 15,
        "teacher_batch_size": 16,
        "teacher_input_size": 224,
        "teacher_lr": 3e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "strong",
        "T": 4.0,
        "alpha": 0.70,
        "labeled_kd_weight": 0.50,
        "notes": "first KD test: EfficientNet-B0 teacher in memory, logits for labeled+unlabeled, distill into plain CNN",
    },
    "kd_robust_t4": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 75,
        "batch_size": 32,
        "input_size": 128,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 7e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "robust",
        "model": "plain_cnn",
        "dropout": 0.40,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 15,
        "teacher_batch_size": 16,
        "teacher_input_size": 224,
        "teacher_lr": 3e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "robust",
        "T": 4.0,
        "alpha": 0.65,
        "labeled_kd_weight": 0.50,
        "notes": "KD with robustness-focused training augmentation to improve stress-test generalization",
    },
    "kd_b0_tta_plain160": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 70,
        "batch_size": 32,
        "input_size": 160,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "plain_cnn",
        "dropout": 0.35,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 15,
        "teacher_batch_size": 16,
        "teacher_input_size": "auto",
        "teacher_lr": 3e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "strong",
        "teacher_tta": True,
        "normalize_kd": True,
        "T": 4.0,
        "alpha": 0.80,
        "labeled_kd_weight": 0.50,
        "unlabeled_kd_weight": 1.0,
        "notes": "B0 teacher with model-specific preprocessing + TTA logits; plain CNN student at 160px; normalized KD",
    },
    "kd_b0_tta_eca160": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 70,
        "batch_size": 32,
        "input_size": 160,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "plain_eca_head",
        "head_ch": 224,
        "dropout": 0.30,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 15,
        "teacher_batch_size": 16,
        "teacher_input_size": "auto",
        "teacher_lr": 3e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "strong",
        "teacher_tta": True,
        "normalize_kd": True,
        "T": 4.0,
        "alpha": 0.80,
        "labeled_kd_weight": 0.50,
        "unlabeled_kd_weight": 1.0,
        "notes": "B0 TTA teacher; PlainCNN+ECA+224 head student at 160px; normalized KD",
    },
    "kd_convnext_tta_eca160": {
        "seed": 0,
        "val_frac": 0.20,
        "epochs": 70,
        "batch_size": 32,
        "input_size": 160,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": 0,
        "augment": "basic",
        "model": "plain_eca_head",
        "head_ch": 224,
        "dropout": 0.30,
        "teacher_model": "convnext_tiny",
        "teacher_epochs": 12,
        "teacher_batch_size": 8,
        "teacher_input_size": "auto",
        "teacher_lr": 2e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "strong",
        "teacher_tta": True,
        "normalize_kd": True,
        "T": 4.0,
        "alpha": 0.80,
        "labeled_kd_weight": 0.50,
        "unlabeled_kd_weight": 1.0,
        "notes": "ConvNeXt-Tiny TTA teacher; PlainCNN+ECA+224 head student at 160px; normalized KD",
    },
}


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]
    lr: float


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(record: dict) -> None:
    record = {"time": now(), **record}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# Data and augmentation, deliberately torchvision-free.
# -----------------------------------------------------------------------------

class FoodDataset(Dataset):
    def __init__(
        self,
        root: Path,
        indices: Optional[List[int]] = None,
        train: bool = False,
        augment: str = "none",
        return_index: bool = False,
        stress: str = "none",
    ):
        self.root = Path(root)
        full = pd.read_csv(self.root / "labels.csv")
        self.indices = list(range(len(full))) if indices is None else list(indices)
        self.df = full.iloc[self.indices].reset_index(drop=True)
        self.train = train
        self.augment = augment
        self.return_index = return_index
        self.stress = stress

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        img = Image.open(self.root / row["filename"]).convert("RGB")
        if self.train and self.augment != "none":
            img = augment_pil(img, mode=self.augment)
        elif (not self.train) and self.stress != "none":
            img = stress_pil(img, mode=self.stress)
        x = pil_to_tensor(img)
        if self.train and self.augment in {"strong", "robust"}:
            x = random_erasing(x, p=0.20 if self.augment == "robust" else 0.25)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.return_index:
            return x, y, torch.tensor(int(self.indices[i]), dtype=torch.long)
        return x, y


class UnlabeledDataset(Dataset):
    def __init__(self, root: Path, train: bool = False, augment: str = "none", return_index: bool = False):
        self.root = Path(root)
        self.filenames = sorted(p.name for p in self.root.glob("*.jpg"))
        self.train = train
        self.augment = augment
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.filenames[i]).convert("RGB")
        if self.train and self.augment != "none":
            img = augment_pil(img, mode=self.augment)
        x = pil_to_tensor(img)
        if self.train and self.augment in {"strong", "robust"}:
            x = random_erasing(x, p=0.20 if self.augment == "robust" else 0.25)
        if self.return_index:
            return x, torch.tensor(i, dtype=torch.long)
        return x


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr[:, :, :3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def augment_pil(img: Image.Image, mode: str) -> Image.Image:
    # RandomResizedCrop back to server shape 256x256. Food photos tolerate crop/scale,
    # but not extreme geometry.
    if mode == "robust":
        img = random_resized_crop(img, size=256, scale=(0.65, 1.0), ratio=(0.80, 1.25))
    else:
        img = random_resized_crop(img, size=256, scale=(0.72, 1.0), ratio=(0.85, 1.18))

    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    if mode == "robust":
        if random.random() < 0.50:
            angle = random.uniform(-12.0, 12.0)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(255 * m) for m in DATA_MEAN))
        img = color_jitter(img, brightness=0.35, contrast=0.35, saturation=0.30, hue=0.04)
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
        if random.random() < 0.15:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(55, 90))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
    elif mode == "strong":
        if random.random() < 0.35:
            angle = random.uniform(-10.0, 10.0)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(255 * m) for m in DATA_MEAN))
        img = color_jitter(img, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04)
    else:
        img = color_jitter(img, brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02)
    return img


def random_resized_crop(img: Image.Image, size: int, scale: Tuple[float, float], ratio: Tuple[float, float]) -> Image.Image:
    width, height = img.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))
            return img.resize((size, size), Image.Resampling.BILINEAR)
    # Fallback center crop.
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.BILINEAR)


STRESS_TESTS = [
    "brightness_down",
    "brightness_up",
    "contrast_down",
    "contrast_up",
    "mild_blur",
    "crop_90",
    "jpeg_55",
    "rotate_left",
    "rotate_right",
]


def stress_pil(img: Image.Image, mode: str) -> Image.Image:
    """Deterministic, realistic validation stress tests on provided validation images only."""
    if mode == "brightness_down":
        return ImageEnhance.Brightness(img).enhance(0.75)
    if mode == "brightness_up":
        return ImageEnhance.Brightness(img).enhance(1.25)
    if mode == "contrast_down":
        return ImageEnhance.Contrast(img).enhance(0.75)
    if mode == "contrast_up":
        return ImageEnhance.Contrast(img).enhance(1.25)
    if mode == "mild_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=0.8))
    if mode == "crop_90":
        w, h = img.size
        cw, ch = int(0.90 * w), int(0.90 * h)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.BILINEAR)
    if mode == "jpeg_55":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if mode == "rotate_left":
        return img.rotate(-7.0, resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(255 * m) for m in DATA_MEAN))
    if mode == "rotate_right":
        return img.rotate(7.0, resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(255 * m) for m in DATA_MEAN))
    raise ValueError(f"unknown stress mode: {mode}")


def color_jitter(img: Image.Image, brightness: float, contrast: float, saturation: float, hue: float) -> Image.Image:
    ops = []
    if brightness > 0:
        ops.append(lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(1 - brightness, 1 + brightness)))
    if contrast > 0:
        ops.append(lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(1 - contrast, 1 + contrast)))
    if saturation > 0:
        ops.append(lambda im: ImageEnhance.Color(im).enhance(random.uniform(1 - saturation, 1 + saturation)))
    random.shuffle(ops)
    for op in ops:
        img = op(img)
    if hue > 0 and random.random() < 0.35:
        hsv = np.asarray(img.convert("HSV"), dtype=np.uint8).copy()
        delta = int(random.uniform(-hue, hue) * 255)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + delta) % 256
        img = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return img


def random_erasing(x: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    if random.random() > p:
        return x
    _, h, w = x.shape
    area = h * w
    for _ in range(10):
        erase_area = random.uniform(0.02, 0.16) * area
        aspect = random.uniform(0.3, 3.3)
        eh = int(round(math.sqrt(erase_area * aspect)))
        ew = int(round(math.sqrt(erase_area / aspect)))
        if 0 < eh < h and 0 < ew < w:
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            fill = torch.tensor(DATA_MEAN, dtype=x.dtype).view(3, 1, 1)
            x = x.clone()
            x[:, top:top + eh, left:left + ew] = fill
            return x
    return x


def stratified_split(labels: Iterable[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    train_idx, val_idx = [], []
    for y in sorted(by_class):
        idxs = by_class[y]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_kfold(labels: Iterable[int], folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Return deterministic stratified train/val indices for k-fold CV."""
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    fold_vals: List[List[int]] = [[] for _ in range(folds)]
    for y in sorted(by_class):
        idxs = by_class[y]
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            fold_vals[j % folds].append(idx)

    all_idx = set(range(len(list(labels)))) if not isinstance(labels, list) else set(range(len(labels)))
    splits: List[Tuple[List[int], List[int]]] = []
    for f in range(folds):
        val_idx = sorted(fold_vals[f])
        train_idx = sorted(all_idx - set(val_idx))
        splits.append((train_idx, val_idx))
    return splits


# -----------------------------------------------------------------------------
# Student model: MobileNetV2-ish under 500K params.
# -----------------------------------------------------------------------------

class Preprocess(nn.Module):
    def __init__(self, net: nn.Module, size: int = 128, mean=DATA_MEAN, std=DATA_STD):
        super().__init__()
        self.net = net
        self.size = int(size)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=[self.size, self.size], mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel=1, stride=1))
        layers.append(ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden))
        layers.append(nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            return x + y
        return y


class MicroMobileNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.25):
        super().__init__()
        layers: List[nn.Module] = [ConvBNAct(3, 24, kernel=3, stride=2)]  # 128 -> 64
        spec = [
            # out, stride, expand, repeats
            (24, 1, 1, 1),
            (32, 2, 3, 2),
            (48, 2, 3, 2),
            (64, 2, 3, 2),
            (96, 2, 3, 2),
            (128, 1, 3, 2),
        ]
        in_ch = 24
        for out_ch, stride, exp, reps in spec:
            for r in range(reps):
                layers.append(InvertedResidual(in_ch, out_ch, stride if r == 0 else 1, exp))
                in_ch = out_ch
        layers.append(ConvBNAct(in_ch, 512, kernel=1, stride=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ECA(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class PlainCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, 1),
            ConvBNAct(32, 32, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            ConvBNAct(32, 64, 3, 1),
            ConvBNAct(64, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),

            ConvBNAct(64, 96, 3, 1),
            ConvBNAct(96, 96, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            ConvBNAct(96, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class PlainCNNECAHead(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.30, head_ch: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, 1),
            ConvBNAct(32, 32, 3, 1),
            ECA(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            ConvBNAct(32, 64, 3, 1),
            ConvBNAct(64, 64, 3, 1),
            ECA(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),

            ConvBNAct(64, 96, 3, 1),
            ConvBNAct(96, 96, 3, 1),
            ECA(96),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            ConvBNAct(96, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            ECA(128),
            nn.MaxPool2d(2),

            ConvBNAct(128, head_ch, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]
    if name == "micro_mobilenet":
        net = MicroMobileNet(dropout=float(cfg["dropout"]))
    elif name == "plain_cnn":
        net = PlainCNN(dropout=float(cfg["dropout"]))
    elif name == "plain_eca_head":
        net = PlainCNNECAHead(dropout=float(cfg["dropout"]), head_ch=int(cfg.get("head_ch", 224)))
    else:
        raise ValueError(f"unknown model: {name}")
    return Preprocess(net, size=int(cfg["input_size"]))


class TeacherPreprocess(nn.Module):
    def __init__(self, net: nn.Module, size: int = 224, mean=None, std=None, interpolation: str = "bilinear"):
        super().__init__()
        self.net = net
        self.size = int(size)
        mode = "bicubic" if "bicubic" in str(interpolation).lower() else "bilinear"
        self.mode = mode
        mean = [0.485, 0.456, 0.406] if mean is None else list(mean)
        std = [0.229, 0.224, 0.225] if std is None else list(std)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=[self.size, self.size], mode=self.mode, align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


def build_teacher(cfg: dict) -> nn.Module:
    if timm is None:
        raise RuntimeError("timm is required for teacher recipes. Install with: python3 -m pip install timm")
    net = timm.create_model(str(cfg["teacher_model"]), pretrained=True, num_classes=NUM_CLASSES)
    data_cfg = timm.data.resolve_model_data_config(net)
    resolved_size = int(data_cfg.get("input_size", (3, 224, 224))[-1])
    requested = cfg.get("teacher_input_size", "auto")
    size = resolved_size if str(requested) == "auto" else int(requested)
    return TeacherPreprocess(
        net,
        size=size,
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
        interpolation=data_cfg.get("interpolation", "bilinear"),
    )


# -----------------------------------------------------------------------------
# Training.
# -----------------------------------------------------------------------------

def cross_entropy_maybe_smooth(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, cfg: dict) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    clip = float(cfg.get("clip_grad_norm", 0.0) or 0.0)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=device)
            mixed_x = lam * x + (1.0 - lam) * x[perm]
            logits = model(mixed_x)
            loss = lam * cross_entropy_maybe_smooth(logits, y, label_smoothing) + (1.0 - lam) * cross_entropy_maybe_smooth(logits, y[perm], label_smoothing)
            # Accuracy is measured against hard original label only; approximate but useful.
            pred = logits.argmax(1)
            correct = (pred == y).sum().item()
        else:
            logits = model(x)
            loss = cross_entropy_maybe_smooth(logits, y, label_smoothing)
            correct = (logits.argmax(1) == y).sum().item()

        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total += bs
    return total_loss / total, total_correct / total


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_correct, total, conf_sum = 0.0, 0, 0, 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        probs = F.softmax(logits, dim=1)
        conf_sum += probs.max(1).values.sum().item()
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total, conf_sum / total


def evaluate_stress(
    model: nn.Module,
    val_indices: List[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    """Run every validation run through realistic robustness checks."""
    if not val_indices:
        return {}
    stress = {}
    for mode in STRESS_TESTS:
        ds = FoodDataset(DATA_ROOT, val_indices, train=False, augment="none", stress=mode)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        loss, acc, conf = evaluate(model, loader, device)
        stress[mode] = {"loss": float(loss), "acc": float(acc), "conf": float(conf)}
    accs = [v["acc"] for v in stress.values()]
    losses = [v["loss"] for v in stress.values()]
    stress["mean"] = {"acc": float(np.mean(accs)), "loss": float(np.mean(losses))}
    stress["worst"] = {"acc": float(np.min(accs)), "loss": float(np.max(losses))}
    return stress


def train_teacher_model(cfg: dict, train_idx: List[int], val_idx: List[int], device: torch.device) -> Tuple[nn.Module, dict]:
    set_seed(int(cfg["seed"]) + 777)
    teacher = build_teacher(cfg).to(device)
    train_ds = FoodDataset(DATA_ROOT, train_idx, train=True, augment=str(cfg.get("teacher_augment", "strong")))
    val_ds = FoodDataset(DATA_ROOT, val_idx, train=False, augment="none")
    train_loader = DataLoader(train_ds, batch_size=int(cfg["teacher_batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["teacher_batch_size"]), shuffle=False, num_workers=int(cfg.get("num_workers", 0)))

    opt = torch.optim.AdamW(teacher.parameters(), lr=float(cfg["teacher_lr"]), weight_decay=float(cfg["teacher_weight_decay"]))
    epochs = int(cfg["teacher_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg.get("min_lr", 1e-5)))
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in teacher.state_dict().items()})
    best = {"teacher_best_val_acc": -1.0, "teacher_best_val_loss": float("inf"), "teacher_best_epoch": 0, "teacher_best_conf": None}

    for epoch in range(1, epochs + 1):
        teacher.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = teacher(x)
            loss = F.cross_entropy(logits, y, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(cfg.get("clip_grad_norm", 3.0)))
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        val_loss, val_acc, val_conf = evaluate(teacher, val_loader, device)
        if (val_acc > best["teacher_best_val_acc"]) or (val_acc == best["teacher_best_val_acc"] and val_loss < best["teacher_best_val_loss"]):
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in teacher.state_dict().items()})
            best.update({"teacher_best_val_acc": val_acc, "teacher_best_val_loss": val_loss, "teacher_best_epoch": epoch, "teacher_best_conf": val_conf})
        print(f"teacher epoch {epoch:03d}/{epochs} train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} conf={val_conf:.4f}")

    teacher.load_state_dict(best_state)
    return teacher, best


def tensor_center_crop_resize(x: torch.Tensor, frac: float = 0.90) -> torch.Tensor:
    _b, _c, h, w = x.shape
    ch, cw = int(h * frac), int(w * frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    y = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(y, size=[h, w], mode="bilinear", align_corners=False)


@torch.inference_mode()
def teacher_predict_logits(teacher: nn.Module, x: torch.Tensor, device: torch.device, use_tta: bool = False) -> torch.Tensor:
    x = x.to(device)
    if not use_tta:
        return teacher(x)
    views = [
        x,
        torch.flip(x, dims=[3]),
        tensor_center_crop_resize(x, frac=0.90),
        torch.clamp((x - 0.5) * 1.15 + 0.5, 0.0, 1.0),
        torch.clamp((x - 0.5) * 0.90 + 0.5, 0.0, 1.0),
    ]
    return torch.stack([teacher(v) for v in views], dim=0).mean(dim=0)


@torch.inference_mode()
def collect_labeled_logits(teacher: nn.Module, device: torch.device, batch_size: int, use_tta: bool = False) -> torch.Tensor:
    ds = FoodDataset(DATA_ROOT, indices=None, train=False, augment="none", return_index=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = torch.empty(len(ds), NUM_CLASSES, dtype=torch.float32)
    teacher.eval()
    for x, _y, idx in loader:
        logits_all[idx] = teacher_predict_logits(teacher, x, device, use_tta=use_tta).detach().cpu()
    return logits_all


@torch.inference_mode()
def collect_unlabeled_logits(teacher: nn.Module, device: torch.device, batch_size: int, use_tta: bool = False) -> torch.Tensor:
    ds = UnlabeledDataset(ROOT / "data" / "unlabeled", train=False, augment="none", return_index=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = torch.empty(len(ds), NUM_CLASSES, dtype=torch.float32)
    teacher.eval()
    for x, idx in loader:
        logits_all[idx] = teacher_predict_logits(teacher, x, device, use_tta=use_tta).detach().cpu()
    return logits_all


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)


def train_kd_epoch(
    student: nn.Module,
    lab_loader: DataLoader,
    un_loader: DataLoader,
    labeled_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    opt: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    ema_state: Optional[dict] = None,
    ema_decay: float = 0.0,
) -> Tuple[float, float, float, float]:
    student.train()
    T = float(cfg["T"])
    alpha = float(cfg["alpha"])
    labeled_kd_weight = float(cfg.get("labeled_kd_weight", 0.5))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    lab_iter = iter(lab_loader)
    un_iter = iter(un_loader)
    total_loss, ce_sum, kd_sum, correct, total = 0.0, 0.0, 0.0, 0, 0
    steps = max(len(lab_loader), len(un_loader))

    for _step in range(steps):
        try:
            x_lab, y_lab, idx_lab = next(lab_iter)
        except StopIteration:
            lab_iter = iter(lab_loader)
            x_lab, y_lab, idx_lab = next(lab_iter)
        try:
            x_un, idx_un = next(un_iter)
        except StopIteration:
            un_iter = iter(un_loader)
            x_un, idx_un = next(un_iter)

        x_lab, y_lab = x_lab.to(device), y_lab.to(device)
        x_un = x_un.to(device)
        t_lab = labeled_logits[idx_lab].to(device)
        t_un = unlabeled_logits[idx_un].to(device)

        z_lab = student(x_lab)
        z_un = student(x_un)
        ce = F.cross_entropy(z_lab, y_lab, label_smoothing=label_smoothing)
        kd_lab = kd_loss(z_lab, t_lab, T)
        kd_un = kd_loss(z_un, t_un, T)
        unlabeled_kd_weight = float(cfg.get("unlabeled_kd_weight", 1.0))
        kd = labeled_kd_weight * kd_lab + unlabeled_kd_weight * kd_un
        if bool(cfg.get("normalize_kd", False)):
            kd = kd / max(1e-8, labeled_kd_weight + unlabeled_kd_weight)
        loss = (1.0 - alpha) * ce + alpha * kd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.get("clip_grad_norm", 3.0)))
        opt.step()
        if ema_state is not None and ema_decay > 0:
            with torch.no_grad():
                for k, v in student.state_dict().items():
                    vc = v.detach().cpu()
                    if torch.is_floating_point(vc):
                        ema_state[k].mul_(ema_decay).add_(vc, alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(vc)

        bs = x_lab.size(0)
        total_loss += loss.item() * bs
        ce_sum += ce.item() * bs
        kd_sum += kd.item() * bs
        correct += (z_lab.argmax(1) == y_lab).sum().item()
        total += bs
    return total_loss / total, ce_sum / total, kd_sum / total, correct / total


def make_loaders(
    cfg: dict,
    train_all: bool = False,
    split_indices: Optional[Tuple[List[int], List[int]]] = None,
) -> Tuple[DataLoader, Optional[DataLoader], dict]:
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    if split_indices is not None:
        train_idx, val_idx = split_indices
    elif train_all:
        train_idx = list(range(len(df)))
        val_idx = []
    else:
        train_idx, val_idx = stratified_split(df["label"].tolist(), float(cfg["val_frac"]), int(cfg["seed"]))

    train_ds = FoodDataset(DATA_ROOT, train_idx, train=True, augment=str(cfg["augment"]))
    val_ds = FoodDataset(DATA_ROOT, val_idx, train=False, augment="none") if val_idx else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )
    info = {"n_train": len(train_idx), "n_val": len(val_idx), "val_indices": list(val_idx)}
    return train_loader, val_loader, info


def fit(
    cfg: dict,
    recipe: str,
    train_all: bool,
    epochs: int,
    device: torch.device,
    split_indices: Optional[Tuple[List[int], List[int]]] = None,
) -> Tuple[nn.Module, dict]:
    set_seed(int(cfg["seed"]))
    train_loader, val_loader, data_info = make_loaders(cfg, train_all=train_all, split_indices=split_indices)
    model = build_model(cfg).to(device)
    params = count_params(model)
    assert params < 500_000, f"Over parameter cap: {params:,}"

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg["min_lr"]))

    best_state = copy.deepcopy(model.state_dict())
    best = {"best_val_acc": -1.0, "best_val_loss": float("inf"), "best_epoch": 0, "best_conf": None}
    history: List[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, cfg)
        scheduler.step()
        lr = float(opt.param_groups[0]["lr"])
        val_loss = val_acc = val_conf = None
        if val_loader is not None:
            val_loss, val_acc, val_conf = evaluate(model, val_loader, device)
            if (val_acc > best["best_val_acc"]) or (val_acc == best["best_val_acc"] and val_loss < best["best_val_loss"]):
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best.update({"best_val_acc": val_acc, "best_val_loss": val_loss, "best_epoch": epoch, "best_conf": val_conf})
            print(f"epoch {epoch:03d}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} conf={val_conf:.4f} lr={lr:.5g}")
        else:
            print(f"epoch {epoch:03d}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} lr={lr:.5g}")
        history.append(EpochMetrics(epoch, tr_loss, tr_acc, val_loss, val_acc, lr))

    stress_result = {}
    if val_loader is not None:
        model.load_state_dict(best_state)
        stress_result = evaluate_stress(model, data_info["val_indices"], device, batch_size=int(cfg["batch_size"]))
        print(f"stress mean_acc={stress_result['mean']['acc']:.4f} worst_acc={stress_result['worst']['acc']:.4f} mean_loss={stress_result['mean']['loss']:.4f}")
    else:
        best = {"best_val_acc": None, "best_val_loss": None, "best_epoch": epochs, "best_conf": None}

    data_public = {k: v for k, v in data_info.items() if k != "val_indices"}
    result = {
        "recipe": recipe,
        "train_all": train_all,
        "epochs_requested": epochs,
        "params": params,
        **data_public,
        **best,
        "stress_tests": stress_result,
        "stress_mean_acc": stress_result.get("mean", {}).get("acc"),
        "stress_worst_acc": stress_result.get("worst", {}).get("acc"),
        "stress_mean_loss": stress_result.get("mean", {}).get("loss"),
        "last_train_loss": history[-1].train_loss if history else None,
        "last_train_acc": history[-1].train_acc if history else None,
    }
    return model, result


def export_torchscript(model: nn.Module, path: Path = MODEL_PATH) -> None:
    model_cpu = model.cpu().eval()
    with torch.inference_mode():
        dummy = torch.rand(2, 3, 256, 256)
        out = model_cpu(dummy)
        assert out.shape == (2, NUM_CLASSES), f"bad output shape {tuple(out.shape)}"
    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, str(path))


def train_kd_student_from_logits(
    cfg: dict,
    recipe: str,
    train_idx: List[int],
    val_idx: List[int],
    labeled_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    lab_ds = FoodDataset(DATA_ROOT, train_idx, train=True, augment=str(cfg["augment"]), return_index=True)
    val_ds = FoodDataset(DATA_ROOT, val_idx, train=False, augment="none")
    un_ds = UnlabeledDataset(ROOT / "data" / "unlabeled", train=True, augment=str(cfg["augment"]), return_index=True)
    lab_loader = DataLoader(lab_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg.get("num_workers", 0)))
    un_loader = DataLoader(un_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))

    set_seed(int(cfg["seed"]))
    student = build_model(cfg).to(device)
    params = count_params(student)
    assert params < 500_000, f"Over parameter cap: {params:,}"
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg["min_lr"]))

    ema_decay = float(cfg.get("ema_decay", 0.0) or 0.0)
    ema_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()}) if ema_decay > 0 else None
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
    best = {"best_val_acc": -1.0, "best_val_loss": float("inf"), "best_epoch": 0, "best_conf": None, "best_uses_ema": bool(ema_decay > 0)}

    for epoch in range(1, epochs + 1):
        loss, ce, kd, tr_acc = train_kd_epoch(student, lab_loader, un_loader, labeled_logits, unlabeled_logits, opt, device, cfg, ema_state=ema_state, ema_decay=ema_decay)
        scheduler.step()
        if ema_state is not None:
            raw_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
            student.load_state_dict(ema_state)
            val_loss, val_acc, val_conf = evaluate(student, val_loader, device)
            student.load_state_dict(raw_state)
        else:
            val_loss, val_acc, val_conf = evaluate(student, val_loader, device)
        if (val_acc > best["best_val_acc"]) or (val_acc == best["best_val_acc"] and val_loss < best["best_val_loss"]):
            if ema_state is not None:
                best_state = copy.deepcopy(ema_state)
            else:
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
            best.update({"best_val_acc": val_acc, "best_val_loss": val_loss, "best_epoch": epoch, "best_conf": val_conf})
        tag = "ema" if ema_state is not None else "raw"
        print(f"{recipe} {tag} kd epoch {epoch:03d}/{epochs} loss={loss:.4f} ce={ce:.4f} kd={kd:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} conf={val_conf:.4f}")

    student.load_state_dict(best_state)
    stress_result = evaluate_stress(student, val_idx, device, batch_size=int(cfg["batch_size"]))
    print(f"{recipe} stress mean_acc={stress_result['mean']['acc']:.4f} worst_acc={stress_result['worst']['acc']:.4f} mean_loss={stress_result['mean']['loss']:.4f}")
    result = {
        "recipe": recipe,
        "params": params,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_unlabeled": len(un_ds),
        **best,
        "stress_tests": stress_result,
        "stress_mean_acc": stress_result.get("mean", {}).get("acc"),
        "stress_worst_acc": stress_result.get("worst", {}).get("acc"),
        "stress_mean_loss": stress_result.get("mean", {}).get("loss"),
    }
    return student, result


def run_kd(recipe: str, export: bool = True) -> None:
    if recipe not in RECIPES:
        raise SystemExit(f"Unknown recipe {recipe}. Available: {', '.join(RECIPES)}")
    cfg = copy.deepcopy(RECIPES[recipe])
    device = get_device()
    set_seed(int(cfg["seed"]))
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    train_idx, val_idx = stratified_split(df["label"].tolist(), float(cfg["val_frac"]), int(cfg["seed"]))
    print(f"KD recipe={recipe} device={device} train={len(train_idx)} val={len(val_idx)} cfg={json.dumps(cfg, sort_keys=True)}")

    teacher, teacher_result = train_teacher_model(cfg, train_idx, val_idx, device)
    print("Collecting teacher logits for labeled and unlabeled images in memory...")
    use_tta = bool(cfg.get("teacher_tta", False))
    labeled_logits = collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=use_tta)
    unlabeled_logits = collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=use_tta)
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    pseudo_counts = torch.bincount(unlabeled_probs.argmax(1), minlength=NUM_CLASSES).tolist()
    pseudo_conf = float(unlabeled_probs.max(1).values.mean().item())
    del teacher

    lab_ds = FoodDataset(DATA_ROOT, train_idx, train=True, augment=str(cfg["augment"]), return_index=True)
    val_ds = FoodDataset(DATA_ROOT, val_idx, train=False, augment="none")
    un_ds = UnlabeledDataset(ROOT / "data" / "unlabeled", train=True, augment=str(cfg["augment"]), return_index=True)
    lab_loader = DataLoader(lab_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg.get("num_workers", 0)))
    un_loader = DataLoader(un_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))

    student = build_model(cfg).to(device)
    params = count_params(student)
    assert params < 500_000, f"Over parameter cap: {params:,}"
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg["min_lr"]))

    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
    best = {"best_val_acc": -1.0, "best_val_loss": float("inf"), "best_epoch": 0, "best_conf": None}
    for epoch in range(1, epochs + 1):
        loss, ce, kd, tr_acc = train_kd_epoch(student, lab_loader, un_loader, labeled_logits, unlabeled_logits, opt, device, cfg)
        scheduler.step()
        val_loss, val_acc, val_conf = evaluate(student, val_loader, device)
        if (val_acc > best["best_val_acc"]) or (val_acc == best["best_val_acc"] and val_loss < best["best_val_loss"]):
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
            best.update({"best_val_acc": val_acc, "best_val_loss": val_loss, "best_epoch": epoch, "best_conf": val_conf})
        print(f"kd epoch {epoch:03d}/{epochs} loss={loss:.4f} ce={ce:.4f} kd={kd:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} conf={val_conf:.4f}")

    student.load_state_dict(best_state)
    stress_result = evaluate_stress(student, val_idx, device, batch_size=int(cfg["batch_size"]))
    print(f"kd stress mean_acc={stress_result['mean']['acc']:.4f} worst_acc={stress_result['worst']['acc']:.4f} mean_loss={stress_result['mean']['loss']:.4f}")
    result = {
        "type": "kd_validation_run",
        "recipe": recipe,
        "params": params,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_unlabeled": len(un_ds),
        **teacher_result,
        **best,
        "stress_tests": stress_result,
        "stress_mean_acc": stress_result.get("mean", {}).get("acc"),
        "stress_worst_acc": stress_result.get("worst", {}).get("acc"),
        "stress_mean_loss": stress_result.get("mean", {}).get("loss"),
        "unlabeled_pseudo_counts": pseudo_counts,
        "unlabeled_mean_teacher_conf": pseudo_conf,
        "config": cfg,
    }
    if export:
        export_torchscript(student, MODEL_PATH)
        result["model_path"] = str(MODEL_PATH)
    append_log(result)
    print("KD SUMMARY")
    print(json.dumps(result, indent=2, sort_keys=True))


def run_branches(export_best: bool = True) -> None:
    """Run the three currently most promising branches with one shared teacher.

    Branches:
      1. T=2 temperature test
      2. alpha=0.5 teacher-weight test
      3. EMA student weights
    """
    base_name = "kd_v1_t4"
    cfg_base = copy.deepcopy(RECIPES[base_name])
    device = get_device()
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    train_idx, val_idx = stratified_split(df["label"].tolist(), float(cfg_base["val_frac"]), int(cfg_base["seed"]))
    print(f"BRANCHES shared teacher base={base_name} device={device} train={len(train_idx)} val={len(val_idx)}")

    teacher, teacher_result = train_teacher_model(cfg_base, train_idx, val_idx, device)
    print("Collecting shared teacher logits for branch tests...")
    use_tta = bool(cfg_base.get("teacher_tta", False))
    labeled_logits = collect_labeled_logits(teacher, device, batch_size=int(cfg_base["teacher_batch_size"]), use_tta=use_tta)
    unlabeled_logits = collect_unlabeled_logits(teacher, device, batch_size=int(cfg_base["teacher_batch_size"]), use_tta=use_tta)
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    pseudo_counts = torch.bincount(unlabeled_probs.argmax(1), minlength=NUM_CLASSES).tolist()
    pseudo_conf = float(unlabeled_probs.max(1).values.mean().item())
    del teacher

    branch_cfgs = []
    c = copy.deepcopy(cfg_base); c["T"] = 2.0; c["notes"] = "branch 1: lower KD temperature T=2"; branch_cfgs.append(("branch_t2_a07", c))
    c = copy.deepcopy(cfg_base); c["alpha"] = 0.50; c["notes"] = "branch 2: lower KD alpha=0.5"; branch_cfgs.append(("branch_t4_a05", c))
    c = copy.deepcopy(cfg_base); c["ema_decay"] = 0.995; c["notes"] = "branch 3: EMA student weights"; branch_cfgs.append(("branch_t4_a07_ema", c))

    records = []
    best_model = None
    best_record = None
    for name, cfg in branch_cfgs:
        print(f"\n=== Running {name}: T={cfg['T']} alpha={cfg['alpha']} ema={cfg.get('ema_decay', 0)} ===")
        model, result = train_kd_student_from_logits(cfg, name, train_idx, val_idx, labeled_logits, unlabeled_logits, device)
        record = {
            "type": "branch_run",
            "base_recipe": base_name,
            "recipe": name,
            "config": cfg,
            **teacher_result,
            **result,
            "unlabeled_pseudo_counts": pseudo_counts,
            "unlabeled_mean_teacher_conf": pseudo_conf,
        }
        append_log(record)
        records.append(record)
        if best_record is None or (record["best_val_acc"], -record["best_val_loss"], record["stress_mean_acc"]) > (best_record["best_val_acc"], -best_record["best_val_loss"], best_record["stress_mean_acc"]):
            best_record = record
            best_model = copy.deepcopy(model).cpu()

    summary = {
        "type": "branch_summary",
        "base_recipe": base_name,
        "teacher_best_val_acc": teacher_result.get("teacher_best_val_acc"),
        "teacher_best_val_loss": teacher_result.get("teacher_best_val_loss"),
        "branches": [
            {
                "recipe": r["recipe"],
                "best_val_acc": r["best_val_acc"],
                "best_val_loss": r["best_val_loss"],
                "stress_mean_acc": r["stress_mean_acc"],
                "stress_worst_acc": r["stress_worst_acc"],
                "best_epoch": r["best_epoch"],
            } for r in records
        ],
        "selected_recipe": best_record["recipe"] if best_record else None,
        "selection_rule": "clean val acc, then lower val loss, then stress mean acc",
    }
    if export_best and best_model is not None:
        export_torchscript(best_model, MODEL_PATH)
        summary["model_path"] = str(MODEL_PATH)
    append_log(summary)
    print("BRANCH SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_recipe(recipe: str, final_all: bool = True) -> None:
    if recipe not in RECIPES:
        raise SystemExit(f"Unknown recipe {recipe}. Available: {', '.join(RECIPES)}")
    cfg = copy.deepcopy(RECIPES[recipe])
    if "teacher_model" in cfg:
        raise SystemExit("This is a KD recipe. Use `kd` for a split run or `kd_cv` for anti-overfit validation.")
    device = get_device()
    print(f"recipe={recipe} device={device} cfg={json.dumps(cfg, sort_keys=True)}")

    # Validation phase: select epoch and verify the model actually generalizes locally.
    model, val_result = fit(cfg, recipe=recipe, train_all=False, epochs=int(cfg["epochs"]), device=device)
    append_log({"type": "validation_run", "device": str(device), "config": cfg, **val_result})

    if final_all:
        best_epoch = int(cfg["final_epochs"] or val_result["best_epoch"] or cfg["epochs"])
        # Reset seed with an offset so final-all does not exactly replay the split trajectory.
        cfg_final = copy.deepcopy(cfg)
        cfg_final["seed"] = int(cfg["seed"]) + 1000
        print(f"Retraining on all labeled data for {best_epoch} epochs, then exporting {MODEL_PATH.name}")
        final_model, final_result = fit(cfg_final, recipe=recipe, train_all=True, epochs=best_epoch, device=device)
        export_torchscript(final_model, MODEL_PATH)
        append_log({"type": "final_all_export", "device": str(device), "source_validation": val_result, "config": cfg_final, "model_path": str(MODEL_PATH), **final_result})
    else:
        export_torchscript(model, MODEL_PATH)
        append_log({"type": "split_export", "device": str(device), "config": cfg, "model_path": str(MODEL_PATH), **val_result})

    print(f"Saved {MODEL_PATH} params={count_params(build_model(cfg)):,}")


def run_cv(recipe: str, folds: int, epochs_override: Optional[int] = None) -> None:
    if recipe not in RECIPES:
        raise SystemExit(f"Unknown recipe {recipe}. Available: {', '.join(RECIPES)}")
    cfg_base = copy.deepcopy(RECIPES[recipe])
    if "teacher_model" in cfg_base:
        raise SystemExit("This is a KD recipe. Use `python3 experiment.py kd_cv --recipe ...` instead of supervised `cv`.")
    epochs = int(epochs_override or cfg_base["epochs"])
    device = get_device()
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    splits = stratified_kfold(df["label"].tolist(), folds=folds, seed=int(cfg_base["seed"]))

    fold_records = []
    print(f"CV recipe={recipe} folds={folds} epochs={epochs} device={device}")
    for fold, split in enumerate(splits):
        cfg = copy.deepcopy(cfg_base)
        cfg["seed"] = int(cfg_base["seed"]) + fold
        print(f"\n=== fold {fold + 1}/{folds}: train={len(split[0])} val={len(split[1])} seed={cfg['seed']} ===")
        _, result = fit(cfg, recipe=recipe, train_all=False, epochs=epochs, device=device, split_indices=split)
        result = {"fold": fold, **result}
        fold_records.append(result)
        append_log({"type": "cv_fold", "folds": folds, "device": str(device), "config": cfg, **result})

    accs = [float(r["best_val_acc"]) for r in fold_records]
    losses = [float(r["best_val_loss"]) for r in fold_records]
    epochs_best = [int(r["best_epoch"]) for r in fold_records]
    params = int(fold_records[0]["params"]) if fold_records else count_params(build_model(cfg_base))
    stress_accs = [float(r["stress_mean_acc"]) for r in fold_records if r.get("stress_mean_acc") is not None]
    stress_worsts = [float(r["stress_worst_acc"]) for r in fold_records if r.get("stress_worst_acc") is not None]
    summary = {
        "type": "cv_run",
        "recipe": recipe,
        "folds": folds,
        "epochs": epochs,
        "params": params,
        "fold_accs": accs,
        "fold_losses": losses,
        "fold_best_epochs": epochs_best,
        "mean_val_acc": float(np.mean(accs)),
        "std_val_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_val_loss": float(np.mean(losses)),
        "std_val_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "stress_mean_acc": float(np.mean(stress_accs)) if stress_accs else None,
        "stress_worst_acc": float(np.min(stress_worsts)) if stress_worsts else None,
        "config": cfg_base,
        "notes": "CV score is the primary model-selection signal; leaderboard is secondary. Stress tests are required for every validation run.",
    }
    append_log(summary)
    print("\nCV SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_kd_cv(recipe: str, folds: int, epochs_override: Optional[int] = None, teacher_epochs_override: Optional[int] = None) -> None:
    """KD cross-validation: retrain teacher and student per fold.

    This is the anti-leaderboard validation path for any promising KD model.
    It is expensive but gives much cleaner evidence than a single split.
    """
    if recipe not in RECIPES:
        raise SystemExit(f"Unknown recipe {recipe}. Available: {', '.join(RECIPES)}")
    cfg_base = copy.deepcopy(RECIPES[recipe])
    if "teacher_model" not in cfg_base:
        raise SystemExit("This is not a KD recipe. Use supervised `cv` instead.")
    if epochs_override is not None:
        cfg_base["epochs"] = int(epochs_override)
    if teacher_epochs_override is not None:
        cfg_base["teacher_epochs"] = int(teacher_epochs_override)

    device = get_device()
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    splits = stratified_kfold(df["label"].tolist(), folds=folds, seed=int(cfg_base["seed"]))
    print(f"KD CV recipe={recipe} folds={folds} student_epochs={cfg_base['epochs']} teacher_epochs={cfg_base['teacher_epochs']} device={device}")

    fold_records = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        cfg = copy.deepcopy(cfg_base)
        cfg["seed"] = int(cfg_base["seed"]) + fold
        print(f"\n=== KD fold {fold + 1}/{folds}: train={len(train_idx)} val={len(val_idx)} seed={cfg['seed']} ===")
        teacher, teacher_result = train_teacher_model(cfg, train_idx, val_idx, device)
        use_tta = bool(cfg.get("teacher_tta", False))
        print("Collecting fold teacher logits...")
        labeled_logits = collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=use_tta)
        unlabeled_logits = collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=use_tta)
        unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
        pseudo_counts = torch.bincount(unlabeled_probs.argmax(1), minlength=NUM_CLASSES).tolist()
        pseudo_conf = float(unlabeled_probs.max(1).values.mean().item())
        del teacher

        model, result = train_kd_student_from_logits(cfg, recipe, train_idx, val_idx, labeled_logits, unlabeled_logits, device)
        del model
        record = {
            "type": "kd_cv_fold",
            "fold": fold,
            "folds": folds,
            "recipe": recipe,
            "device": str(device),
            "config": cfg,
            **teacher_result,
            **result,
            "unlabeled_pseudo_counts": pseudo_counts,
            "unlabeled_mean_teacher_conf": pseudo_conf,
        }
        append_log(record)
        fold_records.append(record)

    accs = [float(r["best_val_acc"]) for r in fold_records]
    losses = [float(r["best_val_loss"]) for r in fold_records]
    stress_means = [float(r["stress_mean_acc"]) for r in fold_records]
    stress_worsts = [float(r["stress_worst_acc"]) for r in fold_records]
    teacher_accs = [float(r["teacher_best_val_acc"]) for r in fold_records]
    teacher_losses = [float(r["teacher_best_val_loss"]) for r in fold_records]
    summary = {
        "type": "kd_cv_run",
        "recipe": recipe,
        "folds": folds,
        "params": int(fold_records[0]["params"]) if fold_records else count_params(build_model(cfg_base)),
        "student_epochs": int(cfg_base["epochs"]),
        "teacher_epochs": int(cfg_base["teacher_epochs"]),
        "fold_accs": accs,
        "fold_losses": losses,
        "fold_stress_mean_accs": stress_means,
        "fold_stress_worst_accs": stress_worsts,
        "fold_teacher_accs": teacher_accs,
        "fold_teacher_losses": teacher_losses,
        "mean_val_acc": float(np.mean(accs)),
        "std_val_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_val_loss": float(np.mean(losses)),
        "std_val_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "stress_mean_acc": float(np.mean(stress_means)),
        "stress_worst_acc": float(np.min(stress_worsts)),
        "teacher_mean_val_acc": float(np.mean(teacher_accs)),
        "teacher_mean_val_loss": float(np.mean(teacher_losses)),
        "config": cfg_base,
        "notes": "KD CV retrains teacher and student per fold; this is the primary anti-overfit evidence for KD recipes.",
    }
    append_log(summary)
    print("\nKD CV SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


def inspect() -> None:
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    info = {
        "n_images": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "model_params": {name: count_params(build_model(cfg)) for name, cfg in RECIPES.items()},
        "recipes": list(RECIPES.keys()),
    }
    print(json.dumps(info, indent=2, sort_keys=True))


def summarize_log() -> None:
    if not LOG_PATH.exists():
        print("No log.txt yet")
        return
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") in {"validation_run", "split_export", "final_all_export", "cv_fold", "cv_run", "kd_validation_run", "branch_run", "branch_summary", "kd_cv_fold", "kd_cv_run"}:
            print(json.dumps({
                "time": rec.get("time"),
                "type": rec.get("type"),
                "recipe": rec.get("recipe"),
                "fold": rec.get("fold"),
                "folds": rec.get("folds"),
                "params": rec.get("params"),
                "n_train": rec.get("n_train"),
                "n_val": rec.get("n_val"),
                "best_epoch": rec.get("best_epoch"),
                "best_val_acc": rec.get("best_val_acc"),
                "best_val_loss": rec.get("best_val_loss"),
                "mean_val_acc": rec.get("mean_val_acc"),
                "std_val_acc": rec.get("std_val_acc"),
                "mean_val_loss": rec.get("mean_val_loss"),
                "teacher_mean_val_acc": rec.get("teacher_mean_val_acc"),
                "stress_mean_acc": rec.get("stress_mean_acc"),
                "stress_worst_acc": rec.get("stress_worst_acc"),
                "stress_mean_loss": rec.get("stress_mean_loss"),
                "teacher_best_val_acc": rec.get("teacher_best_val_acc"),
                "teacher_best_val_loss": rec.get("teacher_best_val_loss"),
                "model_path": rec.get("model_path"),
            }, sort_keys=True))
        else:
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean ML2 experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    p_run = sub.add_parser("run")
    p_run.add_argument("--recipe", default="first_submit", choices=sorted(RECIPES.keys()))
    p_run.add_argument("--no-final-all", action="store_true", help="Export best split-val checkpoint instead of retraining on all labels")
    p_cv = sub.add_parser("cv")
    p_cv.add_argument("--recipe", default="first_submit", choices=sorted(RECIPES.keys()))
    p_cv.add_argument("--folds", type=int, default=7)
    p_cv.add_argument("--epochs", type=int, default=None, help="Override recipe epochs for a faster probe")
    p_kd = sub.add_parser("kd")
    p_kd.add_argument("--recipe", default="kd_v1_t4", choices=sorted(RECIPES.keys()))
    p_kd.add_argument("--no-export", action="store_true", help="Do not overwrite model.pt")
    p_kdcv = sub.add_parser("kd_cv")
    p_kdcv.add_argument("--recipe", default="kd_b0_tta_eca160", choices=sorted(RECIPES.keys()))
    p_kdcv.add_argument("--folds", type=int, default=3)
    p_kdcv.add_argument("--epochs", type=int, default=None, help="Override student epochs for a faster probe")
    p_kdcv.add_argument("--teacher-epochs", type=int, default=None, help="Override teacher epochs for a faster probe")
    p_br = sub.add_parser("branches")
    p_br.add_argument("--no-export", action="store_true", help="Do not overwrite model.pt with the best branch")
    sub.add_parser("summarize")
    args = parser.parse_args()

    if args.cmd == "inspect":
        inspect()
    elif args.cmd == "run":
        run_recipe(args.recipe, final_all=not args.no_final_all)
    elif args.cmd == "cv":
        run_cv(args.recipe, folds=args.folds, epochs_override=args.epochs)
    elif args.cmd == "kd":
        run_kd(args.recipe, export=not args.no_export)
    elif args.cmd == "kd_cv":
        run_kd_cv(args.recipe, folds=args.folds, epochs_override=args.epochs, teacher_epochs_override=args.teacher_epochs)
    elif args.cmd == "branches":
        run_branches(export_best=not args.no_export)
    elif args.cmd == "summarize":
        summarize_log()


if __name__ == "__main__":
    main()
