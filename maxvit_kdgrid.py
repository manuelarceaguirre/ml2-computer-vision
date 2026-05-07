#!/usr/bin/env python3
"""MaxViT-specific KD grid for ML2 Assignment 2.

Use after teacher diagnostics showed MaxViT is a very strong teacher but the
B0-tuned KD recipe did not transfer optimally. This script trains the staged
MaxViT teacher once per seed, caches logits, then tries a bounded grid of
student/KD combinations.

Outputs:
  - maxvit_kdgrid_results.jsonl
  - maxvit_kdgrid_results.json
  - appends rows to log.txt
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

import experiment as ex

ROOT = Path(__file__).resolve().parent
RESULTS_JSONL = ROOT / "maxvit_kdgrid_results.jsonl"
RESULTS_JSON = ROOT / "maxvit_kdgrid_results.json"
CACHE_DIR = ROOT / ".teacher_cache"
TEACHER_NAME = "maxvit_base_tf_384.in21k_ft_in1k"

BASE_CFG = {
    "seed": 30,
    "val_frac": 0.20,
    "epochs": 75,
    "batch_size": 256,
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
    "teacher_model": TEACHER_NAME,
    "teacher_epochs": 15,
    "teacher_batch_size": 16,
    "teacher_input_size": "auto",
    "teacher_lr": 3e-4,
    "teacher_head_warmup_epochs": 5,
    "teacher_head_lr": 1e-3,
    "teacher_backbone_lr": 2e-5,
    "teacher_weight_decay": 1e-4,
    "teacher_label_smoothing": 0.0,
    "teacher_augment": "basic",
    "teacher_tta": False,
    "normalize_kd": True,
    "T": 2.0,
    "alpha": 0.70,
    "labeled_kd_weight": 0.50,
    "unlabeled_kd_weight": 1.00,
    "notes": "MaxViT staged teacher KD grid",
}

VARIANT_PRESETS: Dict[str, List[Tuple[str, dict]]] = {
    "small": [
        # Control: reproduces the first MaxViT transfer attempt.
        ("eca_t2_a070_l050_u100", {"T": 2.0, "alpha": 0.70, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        # Main hypothesis: MaxViT is too sharp; soften and reduce teacher weight.
        ("eca_t4_a050_l050_u100", {"T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        # Separate T and alpha effects.
        ("eca_t4_a070_l050_u100", {"T": 4.0, "alpha": 0.70, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t2_a050_l050_u100", {"T": 2.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        # Same softened KD but a different under-cap student.
        ("hybrid_t4_a050_l050_u100", {"model": "grid_hybridse", "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-3, "T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
    ],
    "full": [
        ("eca_t2_a070_l050_u100", {"T": 2.0, "alpha": 0.70, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t4_a050_l050_u100", {"T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t4_a070_l050_u100", {"T": 4.0, "alpha": 0.70, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t2_a050_l050_u100", {"T": 2.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t6_a050_l050_u100", {"T": 6.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t4_a040_l050_u100", {"T": 4.0, "alpha": 0.40, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("eca_t4_a050_l050_u050", {"T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 0.50}),
        ("eca_t4_a050_l100_u100", {"T": 4.0, "alpha": 0.50, "labeled_kd_weight": 1.00, "unlabeled_kd_weight": 1.00}),
        ("hybrid_t4_a050_l050_u100", {"model": "grid_hybridse", "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-3, "T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
        ("tinymobile_t4_a050_l050_u100", {"model": "grid_tinymobilenetv2", "dropout": 0.15, "lr": 2e-3, "weight_decay": 1e-3, "T": 4.0, "alpha": 0.50, "labeled_kd_weight": 0.50, "unlabeled_kd_weight": 1.00}),
    ],
}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def refresh_json() -> None:
    if not RESULTS_JSONL.exists():
        return
    rows = [json.loads(line) for line in RESULTS_JSONL.read_text().splitlines() if line.strip()]
    RESULTS_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def log_record(record: dict) -> None:
    row = {"grid": "maxvit_kdgrid", "time_unix": time.time(), **record}
    append_jsonl(RESULTS_JSONL, row)
    refresh_json()
    ex.append_log(row)


def clean_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def score_student(result: dict) -> float:
    acc = float(result.get("best_val_acc") or 0.0)
    loss = float(result.get("best_val_loss") or 10.0)
    stress_mean = float(result.get("stress_mean_acc") or 0.0)
    stress_worst = float(result.get("stress_worst_acc") or 0.0)
    return float(acc + 0.25 * stress_mean + 0.15 * stress_worst - 0.05 * loss)


def make_cfg(seed: int, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update({
        "seed": int(seed),
        "epochs": int(args.student_epochs),
        "teacher_epochs": int(args.teacher_epochs),
        "batch_size": int(args.batch_size),
        "teacher_batch_size": int(args.teacher_batch_size),
        "num_workers": int(args.num_workers),
        "teacher_tta": bool(args.teacher_tta),
    })
    return cfg


@torch.inference_mode()
def teacher_diag(teacher: torch.nn.Module, cfg: dict, val_idx: Sequence[int], device: torch.device) -> dict:
    ds = ex.FoodDataset(ex.DATA_ROOT, list(val_idx), train=False, augment="none")
    loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg["teacher_batch_size"]), shuffle=False, num_workers=0)
    logits_parts, labels_parts = [], []
    teacher.eval()
    for x, y in loader:
        logits_parts.append(ex.teacher_predict_logits(teacher, x, device, use_tta=bool(cfg.get("teacher_tta", False))).detach().cpu())
        labels_parts.append(y.detach().cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts).long()
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1)
    conf = probs.max(1).values
    correct = pred.eq(labels)
    sorted_idx = probs.argsort(dim=1, descending=True)
    top3 = sorted_idx[:, :3].eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
    wrong_conf = conf[~correct].mean().item() if (~correct).any() else float("nan")
    return {
        "teacher_val_acc": float(correct.float().mean().item()),
        "teacher_val_loss": float(F.cross_entropy(logits, labels).item()),
        "teacher_top3_acc": float(top3),
        "teacher_wrong_conf": float(wrong_conf),
    }


def load_or_train_teacher_logits(seed: int, args: argparse.Namespace, device: torch.device) -> Tuple[dict, torch.Tensor, torch.Tensor, List[int], List[int]]:
    CACHE_DIR.mkdir(exist_ok=True)
    cfg = make_cfg(seed, args)
    suffix = "tta" if cfg.get("teacher_tta") else "notta"
    cache_path = CACHE_DIR / f"maxvit_seed{seed}_{suffix}_e{cfg['teacher_epochs']}_hw{cfg['teacher_head_warmup_epochs']}.pt"
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    train_idx, val_idx = ex.stratified_split(df["label"].tolist(), float(cfg["val_frac"]), int(seed))
    if cache_path.exists() and not args.refresh_teacher:
        print(f"Loading cached MaxViT logits from {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu")
        return payload["teacher_result"], payload["labeled_logits"], payload["unlabeled_logits"], train_idx, val_idx

    print(f"Training staged MaxViT teacher seed={seed}", flush=True)
    teacher, teacher_result = ex.train_teacher_model(cfg, train_idx, val_idx, device)
    teacher_result.update(teacher_diag(teacher, cfg, val_idx, device))
    print("Collecting MaxViT logits...", flush=True)
    labeled_logits = ex.collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    unlabeled_logits = ex.collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    probs = F.softmax(unlabeled_logits, dim=1)
    teacher_result.update({
        "unlabeled_pseudo_counts": torch.bincount(probs.argmax(1), minlength=ex.NUM_CLASSES).tolist(),
        "unlabeled_mean_teacher_conf": float(probs.max(1).values.mean().item()),
    })
    torch.save({
        "teacher_result": teacher_result,
        "labeled_logits": labeled_logits,
        "unlabeled_logits": unlabeled_logits,
    }, cache_path)
    print(f"Saved MaxViT logits cache to {cache_path}", flush=True)
    del teacher
    clean_cache()
    return teacher_result, labeled_logits, unlabeled_logits, train_idx, val_idx


def run_variant(name: str, overrides: dict, seed: int, args: argparse.Namespace, teacher_result: dict, labeled_logits: torch.Tensor, unlabeled_logits: torch.Tensor, train_idx: List[int], val_idx: List[int], device: torch.device) -> dict:
    cfg = make_cfg(seed, args)
    cfg.update(overrides)
    cfg["notes"] = f"MaxViT KD grid variant {name}"
    params = ex.count_params(ex.build_model(cfg))
    if params >= 500_000:
        raise RuntimeError(f"variant {name} over param cap: {params}")
    start = time.time()
    print(f"\n=== VARIANT {name} seed={seed} model={cfg['model']} T={cfg['T']} alpha={cfg['alpha']} labeled={cfg['labeled_kd_weight']} unlabeled={cfg['unlabeled_kd_weight']} params={params} ===", flush=True)
    model, result = ex.train_kd_student_from_logits(cfg, f"maxvit_{name}_s{seed}", train_idx, val_idx, labeled_logits, unlabeled_logits, device)
    record = {
        "type": "maxvit_kdgrid_run",
        "status": "ok",
        "variant": name,
        "seed": int(seed),
        "teacher": TEACHER_NAME,
        "config": cfg,
        **teacher_result,
        **result,
        "student_generalization_score": score_student(result),
        "elapsed_sec": round(time.time() - start, 3),
    }
    del model
    clean_cache()
    log_record(record)
    print("VARIANT SUMMARY", json.dumps({
        "variant": name,
        "score": record["student_generalization_score"],
        "best_val_acc": record.get("best_val_acc"),
        "best_val_loss": record.get("best_val_loss"),
        "stress_mean_acc": record.get("stress_mean_acc"),
        "stress_worst_acc": record.get("stress_worst_acc"),
        "best_epoch": record.get("best_epoch"),
    }, sort_keys=True), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search MaxViT KD transfer settings and students.")
    parser.add_argument("--preset", choices=sorted(VARIANT_PRESETS), default="small")
    parser.add_argument("--variants", default="", help="Optional comma subset of variant names from selected preset.")
    parser.add_argument("--seeds", default=os.environ.get("MAXVIT_KDGRID_SEEDS", "30"))
    parser.add_argument("--teacher-epochs", type=int, default=int(os.environ.get("MAXVIT_KDGRID_TEACHER_EPOCHS", "15")))
    parser.add_argument("--student-epochs", type=int, default=int(os.environ.get("MAXVIT_KDGRID_STUDENT_EPOCHS", "75")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MAXVIT_KDGRID_BATCH_SIZE", "256")))
    parser.add_argument("--teacher-batch-size", type=int, default=int(os.environ.get("MAXVIT_KDGRID_TEACHER_BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("MAXVIT_KDGRID_NUM_WORKERS", "0")))
    parser.add_argument("--teacher-tta", action="store_true")
    parser.add_argument("--refresh-teacher", action="store_true")
    parser.add_argument("--keep-going", action="store_true", default=os.environ.get("MAXVIT_KDGRID_KEEP_GOING", "1") == "1")
    args = parser.parse_args()

    device = ex.get_device()
    seeds = [int(x) for x in parse_csv(args.seeds)]
    variants = VARIANT_PRESETS[args.preset]
    if args.variants:
        keep = set(parse_csv(args.variants))
        variants = [(n, v) for n, v in variants if n in keep]
    print(f"maxvit_kdgrid device={device} seeds={seeds} preset={args.preset} variants={[n for n,_ in variants]}", flush=True)

    all_records = []
    for seed in seeds:
        teacher_result, labeled_logits, unlabeled_logits, train_idx, val_idx = load_or_train_teacher_logits(seed, args, device)
        log_record({
            "type": "maxvit_kdgrid_teacher",
            "status": "ok",
            "seed": int(seed),
            "teacher": TEACHER_NAME,
            **teacher_result,
        })
        for name, overrides in variants:
            try:
                all_records.append(run_variant(name, overrides, seed, args, teacher_result, labeled_logits, unlabeled_logits, train_idx, val_idx, device))
            except Exception as e:
                err = {
                    "type": "maxvit_kdgrid_run",
                    "status": "crash",
                    "variant": name,
                    "seed": int(seed),
                    "teacher": TEACHER_NAME,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                log_record(err)
                print(err["traceback"], flush=True)
                clean_cache()
                if not args.keep_going:
                    raise

    summary = {"type": "maxvit_kdgrid_summary", "status": "done", "n_records": len(all_records), "seeds": seeds, "preset": args.preset}
    if all_records:
        best = max(all_records, key=lambda r: float(r.get("student_generalization_score") or -1e9))
        summary.update({
            "best_variant": best.get("variant"),
            "best_student_generalization_score": best.get("student_generalization_score"),
            "best_val_acc": best.get("best_val_acc"),
            "best_val_loss": best.get("best_val_loss"),
            "best_stress_mean_acc": best.get("stress_mean_acc"),
            "best_stress_worst_acc": best.get("stress_worst_acc"),
        })
    log_record(summary)
    print("\nMAXVIT KDGRID SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
