#!/usr/bin/env python3
"""Export the current best ML2 CV submission model.pt.

Best validated recipe from autoresearch:
  teacher: EfficientNet-B0 with teacher TTA logits
  student: plain_eca_head, input_size=160, head_ch=224, <500k params
  KD: T=2, alpha=0.70, labeled=0.50, unlabeled=1.00
  epochs: teacher=10, student=75

This script trains the teacher on all labeled images, collects soft logits for
all labeled + legal unlabeled images, trains the final student for the fixed
chosen epoch count, and exports a TorchScript model.pt suitable for submission.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

import experiment as ex

ROOT = Path(__file__).resolve().parent
EXPORT_PATH = ROOT / "model.pt"
META_PATH = ROOT / "best_b0_kd_export.json"


def best_cfg() -> dict:
    cfg = copy.deepcopy(ex.RECIPES["kd_b0_tta_eca160"])
    cfg.update({
        "seed": 30,
        "val_frac": 0.20,
        "epochs": 75,
        "batch_size": 32,
        "input_size": 160,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": int(os.environ.get("FINAL_NUM_WORKERS", "0")),
        "augment": "basic",
        "model": "plain_eca_head",
        "head_ch": 224,
        "dropout": 0.30,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 10,
        "teacher_batch_size": int(os.environ.get("FINAL_TEACHER_BATCH_SIZE", "16")),
        "teacher_input_size": "auto",
        "teacher_lr": 3e-4,
        "teacher_weight_decay": 1e-4,
        "teacher_augment": "strong",
        "teacher_tta": True,
        "normalize_kd": True,
        "T": 2.0,
        "alpha": 0.70,
        "labeled_kd_weight": 0.50,
        "unlabeled_kd_weight": 1.00,
        "notes": "FINAL export: best B0 KD incumbent from autoresearch, trained on all labeled data.",
    })
    # Ensure no MaxViT staged fields leak into the B0 final config.
    for key in ["teacher_head_warmup_epochs", "teacher_head_lr", "teacher_backbone_lr", "teacher_label_smoothing"]:
        cfg.pop(key, None)
    return cfg


def train_teacher_all(cfg: dict, all_idx: List[int], device: torch.device) -> torch.nn.Module:
    print("TRAIN FINAL TEACHER START", flush=True)
    ex.set_seed(int(cfg["seed"]) + 777)
    teacher = ex.build_teacher(cfg).to(device)
    ds = ex.FoodDataset(ex.DATA_ROOT, all_idx, train=True, augment=str(cfg.get("teacher_augment", "strong")))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["teacher_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )
    opt = torch.optim.AdamW(teacher.parameters(), lr=float(cfg["teacher_lr"]), weight_decay=float(cfg["teacher_weight_decay"]))
    epochs = int(cfg["teacher_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg.get("min_lr", 1e-5)))
    for epoch in range(1, epochs + 1):
        teacher.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = teacher(x)
            loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(cfg.get("clip_grad_norm", 3.0)))
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        print(f"final teacher epoch {epoch:03d}/{epochs} train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f}", flush=True)
    return teacher


def train_final_student_all(cfg: dict, labeled_logits: torch.Tensor, unlabeled_logits: torch.Tensor, all_idx: List[int], device: torch.device) -> Tuple[torch.nn.Module, dict]:
    print("TRAIN FINAL STUDENT START", flush=True)
    ex.set_seed(int(cfg["seed"]))
    lab_ds = ex.FoodDataset(ex.DATA_ROOT, all_idx, train=True, augment=str(cfg["augment"]), return_index=True)
    un_ds = ex.UnlabeledDataset(ROOT / "data" / "unlabeled", train=True, augment=str(cfg["augment"]), return_index=True)
    lab_loader = DataLoader(lab_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    un_loader = DataLoader(un_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))

    student = ex.build_model(cfg).to(device)
    params = ex.count_params(student)
    assert params < 500_000, f"Over parameter cap: {params:,}"
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cfg["min_lr"]))
    history = []
    for epoch in range(1, epochs + 1):
        loss, ce, kd, tr_acc = ex.train_kd_epoch(student, lab_loader, un_loader, labeled_logits, unlabeled_logits, opt, device, cfg)
        scheduler.step()
        row = {"epoch": epoch, "loss": loss, "ce": ce, "kd": kd, "train_acc": tr_acc}
        history.append(row)
        print(f"final student epoch {epoch:03d}/{epochs} loss={loss:.4f} ce={ce:.4f} kd={kd:.4f} train_acc={tr_acc:.4f}", flush=True)
    return student, {"params": params, "history": history, "final_train_acc": history[-1]["train_acc"] if history else None}


def main() -> int:
    t0 = time.time()
    cfg = best_cfg()
    device = ex.get_device()
    ex.set_seed(int(cfg["seed"]))
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    all_idx = list(range(len(df)))
    print("FINAL B0 KD EXPORT", flush=True)
    print(json.dumps({k: cfg[k] for k in ["seed", "teacher_model", "teacher_epochs", "teacher_tta", "model", "input_size", "head_ch", "epochs", "T", "alpha", "labeled_kd_weight", "unlabeled_kd_weight"]}, indent=2), flush=True)
    print(f"device={device} labeled={len(all_idx)} export={EXPORT_PATH}", flush=True)

    teacher = train_teacher_all(cfg, all_idx, device)
    print("COLLECT FINAL TEACHER LOGITS", flush=True)
    labeled_logits = ex.collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    unlabeled_logits = ex.collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    probs = torch.nn.functional.softmax(unlabeled_logits, dim=1)
    pseudo_counts = torch.bincount(probs.argmax(1), minlength=ex.NUM_CLASSES).tolist()
    pseudo_conf = float(probs.max(1).values.mean().item())
    print(f"logits labeled={tuple(labeled_logits.shape)} unlabeled={tuple(unlabeled_logits.shape)} pseudo_counts={pseudo_counts} pseudo_conf={pseudo_conf:.4f}", flush=True)

    student, result = train_final_student_all(cfg, labeled_logits, unlabeled_logits, all_idx, device)
    ex.export_torchscript(student, EXPORT_PATH)
    loaded = torch.jit.load(str(EXPORT_PATH), map_location="cpu").eval()
    with torch.inference_mode():
        out = loaded(torch.rand(2, 3, 256, 256))
    assert tuple(out.shape) == (2, ex.NUM_CLASSES), tuple(out.shape)

    meta = {
        "type": "final_best_b0_kd_export",
        "model_path": str(EXPORT_PATH),
        "config": cfg,
        "n_labeled": len(all_idx),
        "n_unlabeled": int(unlabeled_logits.shape[0]),
        "unlabeled_pseudo_counts": pseudo_counts,
        "unlabeled_mean_teacher_conf": pseudo_conf,
        "elapsed_sec": round(time.time() - t0, 3),
        **result,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ex.append_log(meta)
    print("EXPORT COMPLETE", flush=True)
    print(json.dumps({
        "model_path": str(EXPORT_PATH),
        "model_size_bytes": EXPORT_PATH.stat().st_size,
        "params": result["params"],
        "final_train_acc": result["final_train_acc"],
        "meta_path": str(META_PATH),
        "elapsed_sec": meta["elapsed_sec"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
