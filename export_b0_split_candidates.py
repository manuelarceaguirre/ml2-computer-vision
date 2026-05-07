#!/usr/bin/env python3
"""Train/export leaderboard candidate models from honest B0 KD split checkpoints.

Motivation: the final-all export overfit the 399 labeled images and scored poorly
on hidden/client data. This script instead trains the validated EfficientNet-B0
KD recipe on honest train/validation splits and exports the best validation
checkpoint(s), optionally with validation-fitted logit temperature for NLL.

Outputs:
  - model_b0_split_seed{seed}_acc{...}_loss{...}_t{...}.pt
  - model.pt copied from the best selected candidate
  - b0_split_candidates.json metadata
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as ex

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "b0_split_candidates.json"


class TemperatureScaled(nn.Module):
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / torch.clamp(self.temperature, min=1e-4)


def cfg_best(seed: int, teacher_batch_size: int, batch_size: int, num_workers: int) -> dict:
    cfg = copy.deepcopy(ex.RECIPES["kd_b0_tta_eca160"])
    cfg.update({
        "seed": int(seed),
        "val_frac": 0.20,
        "epochs": 75,
        "batch_size": int(batch_size),
        "input_size": 160,
        "lr": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.0,
        "clip_grad_norm": 3.0,
        "num_workers": int(num_workers),
        "augment": "basic",
        "model": "plain_eca_head",
        "head_ch": 224,
        "dropout": 0.30,
        "teacher_model": "efficientnet_b0",
        "teacher_epochs": 10,
        "teacher_batch_size": int(teacher_batch_size),
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
        "notes": "Leaderboard candidate: honest split checkpoint export for best B0 KD incumbent.",
    })
    for key in ["teacher_head_warmup_epochs", "teacher_head_lr", "teacher_backbone_lr", "teacher_label_smoothing"]:
        cfg.pop(key, None)
    return cfg


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@torch.inference_mode()
def collect_model_logits(model: nn.Module, indices: List[int], device: torch.device, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = ex.FoodDataset(ex.DATA_ROOT, indices, train=False, augment="none")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_parts, labels_parts = [], []
    model.eval()
    for x, y in loader:
        logits_parts.append(model(x.to(device)).detach().cpu())
        labels_parts.append(y.detach().cpu())
    return torch.cat(logits_parts), torch.cat(labels_parts).long()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 80) -> Tuple[float, float, float]:
    """Fit scalar temperature on validation logits for NLL only."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    base_nll = float(F.cross_entropy(logits, labels).item())
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(0.2, 5.0)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass
    temp = float(torch.exp(log_t.detach()).clamp(0.2, 5.0).item())
    cal_nll = float(F.cross_entropy(logits / temp, labels).item())
    return temp, base_nll, cal_nll


@torch.inference_mode()
def eval_model(model: nn.Module, val_idx: List[int], device: torch.device, batch_size: int) -> dict:
    logits, labels = collect_model_logits(model, val_idx, device, batch_size=batch_size)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    cm = torch.zeros(ex.NUM_CLASSES, ex.NUM_CLASSES, dtype=torch.long)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1
    per_class_acc = (cm.diag().float() / cm.sum(1).clamp_min(1).float()).tolist()
    return {
        "val_acc": float(pred.eq(labels).float().mean().item()),
        "val_loss": float(F.cross_entropy(logits, labels).item()),
        "val_conf": float(conf.mean().item()),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }


def export_scripted(model: nn.Module, path: Path) -> None:
    model_cpu = model.cpu().eval()
    with torch.inference_mode():
        out = model_cpu(torch.rand(2, 3, 256, 256))
    assert tuple(out.shape) == (2, ex.NUM_CLASSES), tuple(out.shape)
    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        out2 = loaded(torch.rand(2, 3, 256, 256))
    assert tuple(out2.shape) == (2, ex.NUM_CLASSES), tuple(out2.shape)


def candidate_score(record: dict) -> float:
    # Accuracy is leaderboard primary; loss is secondary. Stress mean is local tie-breaker.
    return float(record["val_acc"]) + 0.02 * float(record.get("stress_mean_acc") or 0.0) - 0.01 * float(record["val_loss"])


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> dict:
    cfg = cfg_best(seed, args.teacher_batch_size, args.batch_size, args.num_workers)
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    train_idx, val_idx = ex.stratified_split(df["label"].tolist(), float(cfg["val_frac"]), seed)
    print(f"\n=== B0 SPLIT CANDIDATE seed={seed} train={len(train_idx)} val={len(val_idx)} device={device} ===", flush=True)
    print(json.dumps({k: cfg[k] for k in ["teacher_model", "teacher_epochs", "teacher_tta", "model", "input_size", "head_ch", "epochs", "T", "alpha", "labeled_kd_weight", "unlabeled_kd_weight"]}, indent=2), flush=True)

    t0 = time.time()
    teacher, teacher_result = ex.train_teacher_model(cfg, train_idx, val_idx, device)
    print("Collecting teacher logits...", flush=True)
    labeled_logits = ex.collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=True)
    unlabeled_logits = ex.collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=True)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student, student_result = ex.train_kd_student_from_logits(cfg, f"b0_split_seed{seed}", train_idx, val_idx, labeled_logits, unlabeled_logits, device)
    raw_eval = eval_model(student, val_idx, device, batch_size=max(32, args.batch_size))
    val_logits, val_labels = collect_model_logits(student, val_idx, device, batch_size=max(32, args.batch_size))
    temp, raw_nll, cal_nll = fit_temperature(val_logits, val_labels)
    export_model: nn.Module = student
    if args.calibrate:
        export_model = TemperatureScaled(student, temperature=temp)
    cal_eval = eval_model(export_model.to(device), val_idx, device, batch_size=max(32, args.batch_size))
    stress = ex.evaluate_stress(export_model.to(device), val_idx, device, batch_size=max(32, args.batch_size))

    tag = f"seed{seed}_acc{cal_eval['val_acc']:.4f}_loss{cal_eval['val_loss']:.4f}_t{temp:.3f}".replace(".", "p")
    out_path = ROOT / f"model_b0_split_{tag}.pt"
    export_scripted(export_model, out_path)
    params = sum(p.numel() for p in export_model.parameters())
    record = {
        "type": "b0_split_candidate_export",
        "seed": seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "config": cfg,
        "model_path": str(out_path),
        "params": int(params),
        "temperature": float(temp),
        "raw_nll_before_temperature": float(raw_nll),
        "cal_nll_after_temperature": float(cal_nll),
        "elapsed_sec": round(time.time() - t0, 3),
        **teacher_result,
        **student_result,
        "raw_eval": raw_eval,
        **cal_eval,
        "stress_tests": stress,
        "stress_mean_acc": stress.get("mean", {}).get("acc"),
        "stress_worst_acc": stress.get("worst", {}).get("acc"),
        "selection_score": None,
    }
    record["selection_score"] = candidate_score(record)
    ex.append_log(record)
    print("CANDIDATE SUMMARY", json.dumps({
        "seed": seed,
        "path": str(out_path),
        "val_acc": record["val_acc"],
        "val_loss": record["val_loss"],
        "temperature": temp,
        "stress_mean_acc": record["stress_mean_acc"],
        "stress_worst_acc": record["stress_worst_acc"],
        "selection_score": record["selection_score"],
        "best_epoch": record.get("best_epoch"),
    }, indent=2), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Export honest split B0 KD leaderboard candidates.")
    parser.add_argument("--seeds", default="40", help="Comma-separated split seeds. Use 40 for fastest, 30,40,41 for stronger search.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true", default=True)
    args = parser.parse_args()

    device = ex.get_device()
    seeds = parse_seeds(args.seeds)
    records = []
    for seed in seeds:
        records.append(run_seed(seed, args, device))
    best = max(records, key=candidate_score)
    shutil.copyfile(best["model_path"], ROOT / "model.pt")
    payload = {"best_model_path": str(ROOT / "model.pt"), "best_source_path": best["model_path"], "best_seed": best["seed"], "records": records}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nBEST SPLIT CANDIDATE")
    print(json.dumps({
        "best_seed": best["seed"],
        "best_source_path": best["model_path"],
        "model_pt": str(ROOT / "model.pt"),
        "val_acc": best["val_acc"],
        "val_loss": best["val_loss"],
        "temperature": best["temperature"],
        "stress_mean_acc": best["stress_mean_acc"],
        "stress_worst_acc": best["stress_worst_acc"],
        "selection_score": best["selection_score"],
        "params": best["params"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
