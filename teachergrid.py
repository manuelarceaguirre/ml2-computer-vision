#!/usr/bin/env python3
"""Teacher tournament for ML2 Assignment 2.

This script answers: which large pretrained teacher produces the best small
student, and why?  It keeps the student/distillation recipe fixed while
swapping only the teacher model.

Outputs:
  - teachergrid_results.jsonl  append-only records
  - teachergrid_results.json   refreshed JSON array

Examples:
  python teachergrid.py --quick --teachers efficientnet_b0,convnextv2_base.fcmae_ft_in22k_in1k_384 --no-distill
  python teachergrid.py --teachers efficientnet_b0,eva02_base_patch14_448.mim_in22k_ft_in1k,convnextv2_base.fcmae_ft_in22k_in1k_384 --seeds 30,31,32
  python teachergrid.py --teachers eva02_base_patch14_448.mim_in22k_ft_in1k,convnextv2_base.fcmae_ft_in22k_in1k_384,maxvit_base_tf_384.in21k_ft_in1k --make-committee
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
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import experiment as ex

ROOT = Path(__file__).resolve().parent
RESULTS_JSONL = ROOT / "teachergrid_results.jsonl"
RESULTS_JSON = ROOT / "teachergrid_results.json"
MAX_PARAMS = 500_000

DEFAULT_TEACHERS = [
    # Control / incumbent teacher.
    "efficientnet_b0",
    # High-upside practical timm teachers.
    "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "maxvit_base_tf_384.in21k_ft_in1k",
    # Aggressive large teacher; slower/heavier.
    "eva02_large_patch14_448.mim_in22k_ft_in1k",
]

BASE_STUDENT_CFG = {
    # Current robust incumbent family, with autoresearch-selected KD knobs.
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
    "num_workers": 0,
    "augment": "basic",
    "model": "plain_eca_head",
    "head_ch": 224,
    "dropout": 0.30,
    "teacher_model": "efficientnet_b0",
    "teacher_epochs": 10,
    "teacher_batch_size": 8,
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
    "notes": "teachergrid fixed student/KD recipe; only teacher_model changes",
}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def refresh_results_json() -> None:
    if not RESULTS_JSONL.exists():
        return
    rows = [json.loads(line) for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    RESULTS_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_record(record: dict) -> None:
    row = {"grid": "teachergrid", "time_unix": time.time(), **record}
    append_jsonl(RESULTS_JSONL, row)
    refresh_results_json()
    ex.append_log(row)


def clean_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def ece_score(conf: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> float:
    conf = conf.detach().float().cpu()
    correct = correct.detach().float().cpu()
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(correct[mask].mean().item()))
    return float(ece)


@torch.inference_mode()
def collect_labeled_logits_indices(
    teacher: torch.nn.Module,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    use_tta: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = ex.FoodDataset(ex.DATA_ROOT, list(indices), train=False, augment="none", return_index=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_parts, labels_parts = [], []
    teacher.eval()
    for x, y, _idx in loader:
        logits_parts.append(ex.teacher_predict_logits(teacher, x, device, use_tta=use_tta).detach().cpu())
        labels_parts.append(y.detach().cpu())
    return torch.cat(logits_parts, dim=0), torch.cat(labels_parts, dim=0).long()


@torch.inference_mode()
def teacher_ranking_and_calibration(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1)
    conf = probs.max(1).values
    correct = pred.eq(labels)
    true_prob = probs[torch.arange(labels.numel()), labels]
    sorted_idx = probs.argsort(dim=1, descending=True)
    true_rank = sorted_idx.eq(labels.view(-1, 1)).float().argmax(dim=1) + 1
    top2 = sorted_idx[:, :2].eq(labels.view(-1, 1)).any(dim=1)
    top3 = sorted_idx[:, :3].eq(labels.view(-1, 1)).any(dim=1)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(1)
    onehot = F.one_hot(labels, num_classes=ex.NUM_CLASSES).float()
    brier = ((probs - onehot) ** 2).sum(1)
    cm = torch.zeros(ex.NUM_CLASSES, ex.NUM_CLASSES, dtype=torch.long)
    for t, p in zip(labels.cpu(), pred.cpu()):
        cm[int(t), int(p)] += 1
    denom = cm.sum(1).clamp_min(1)
    per_class_acc = (cm.diag().float() / denom.float()).tolist()
    wrong_conf = conf[~correct].mean().item() if (~correct).any() else float("nan")
    correct_conf = conf[correct].mean().item() if correct.any() else float("nan")
    return {
        "teacher_val_acc": float(correct.float().mean().item()),
        "teacher_top2_acc": float(top2.float().mean().item()),
        "teacher_top3_acc": float(top3.float().mean().item()),
        "teacher_mean_true_prob": float(true_prob.mean().item()),
        "teacher_mean_true_rank": float(true_rank.float().mean().item()),
        "teacher_mean_conf": float(conf.mean().item()),
        "teacher_correct_conf": float(correct_conf),
        "teacher_wrong_conf": float(wrong_conf),
        "teacher_entropy": float(entropy.mean().item()),
        "teacher_brier": float(brier.mean().item()),
        "teacher_ece10": ece_score(conf, correct, bins=10),
        "teacher_per_class_acc": per_class_acc,
        "teacher_confusion_matrix": cm.tolist(),
    }


@torch.inference_mode()
def teacher_tta_stability(
    teacher: torch.nn.Module,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    max_examples: int = 256,
) -> dict:
    # Measures prediction stability under harmless views. This is diagnostic only;
    # logits used for distillation can still use the normal TTA average.
    sample_indices = list(indices)[:max_examples]
    ds = ex.FoodDataset(ex.DATA_ROOT, sample_indices, train=False, augment="none")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    agreements, kls = [], []
    teacher.eval()
    for x, _y in loader:
        x = x.to(device)
        views = [
            x,
            torch.flip(x, dims=[3]),
            ex.tensor_center_crop_resize(x, frac=0.90),
            torch.clamp((x - 0.5) * 1.15 + 0.5, 0.0, 1.0),
            torch.clamp((x - 0.5) * 0.90 + 0.5, 0.0, 1.0),
        ]
        logits = [teacher(v) for v in views]
        probs = [F.softmax(z, dim=1) for z in logits]
        base_pred = probs[0].argmax(1)
        for p in probs[1:]:
            agreements.append(p.argmax(1).eq(base_pred).float().detach().cpu())
            kls.append(F.kl_div(torch.clamp(p, min=1e-12).log(), probs[0], reduction="none").sum(1).detach().cpu())
    if not agreements:
        return {"teacher_tta_agreement": None, "teacher_tta_mean_kl": None}
    return {
        "teacher_tta_agreement": float(torch.cat(agreements).mean().item()),
        "teacher_tta_mean_kl": float(torch.cat(kls).mean().item()),
    }


def unlabeled_diagnostics(unlabeled_logits: torch.Tensor) -> dict:
    probs = F.softmax(unlabeled_logits, dim=1)
    conf, pseudo = probs.max(1)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(1)
    return {
        "unlabeled_pseudo_counts": torch.bincount(pseudo, minlength=ex.NUM_CLASSES).tolist(),
        "unlabeled_mean_teacher_conf": float(conf.mean().item()),
        "unlabeled_entropy": float(entropy.mean().item()),
        "unlabeled_high_conf_080": float((conf >= 0.80).float().mean().item()),
        "unlabeled_high_conf_090": float((conf >= 0.90).float().mean().item()),
        "unlabeled_high_conf_095": float((conf >= 0.95).float().mean().item()),
    }


def diagnostic_score(metrics: dict) -> float:
    # Triage score only. Student transfer is the real winner criterion.
    counts = np.array(metrics.get("unlabeled_pseudo_counts") or [1] * ex.NUM_CLASSES, dtype=np.float64)
    counts = counts / max(1.0, counts.sum())
    uniform = np.ones(ex.NUM_CLASSES) / ex.NUM_CLASSES
    collapse_penalty = float(np.abs(counts - uniform).sum())
    wrong_conf = metrics.get("teacher_wrong_conf")
    wrong_conf = 0.75 if wrong_conf is None or np.isnan(wrong_conf) else float(wrong_conf)
    return float(
        metrics.get("teacher_val_acc", 0.0)
        + 0.20 * metrics.get("teacher_top3_acc", 0.0)
        + 0.10 * metrics.get("teacher_mean_true_prob", 0.0)
        - 0.08 * metrics.get("teacher_val_loss", 0.0)
        - 0.08 * wrong_conf
        - 0.05 * collapse_penalty
        + 0.05 * (metrics.get("teacher_tta_agreement") or 0.0)
    )


def student_generalization_score(result: dict) -> float:
    # Same spirit as autoresearch: accuracy first, but penalize loss and brittle stress.
    acc = float(result.get("best_val_acc") or 0.0)
    loss = float(result.get("best_val_loss") or 10.0)
    stress_mean = float(result.get("stress_mean_acc") or 0.0)
    stress_worst = float(result.get("stress_worst_acc") or 0.0)
    return float(acc + 0.25 * stress_mean + 0.15 * stress_worst - 0.05 * loss)


def make_cfg(args: argparse.Namespace, teacher_name: str, seed: int) -> dict:
    cfg = copy.deepcopy(BASE_STUDENT_CFG)
    cfg.update({
        "seed": int(seed),
        "teacher_model": teacher_name,
        "epochs": int(args.student_epochs),
        "teacher_epochs": int(args.teacher_epochs),
        "batch_size": int(args.batch_size),
        "teacher_batch_size": int(args.teacher_batch_size),
        "num_workers": int(args.num_workers),
        "teacher_tta": not args.no_tta,
    })
    if args.quick:
        cfg["epochs"] = 1
        cfg["teacher_epochs"] = 1
        cfg["teacher_tta"] = False
    if args.teacher_lr is not None:
        cfg["teacher_lr"] = float(args.teacher_lr)
    if args.teacher_head_warmup_epochs:
        cfg["teacher_head_warmup_epochs"] = int(args.teacher_head_warmup_epochs)
    if args.teacher_head_lr is not None:
        cfg["teacher_head_lr"] = float(args.teacher_head_lr)
    if args.teacher_backbone_lr is not None:
        cfg["teacher_backbone_lr"] = float(args.teacher_backbone_lr)
    if args.teacher_label_smoothing is not None:
        cfg["teacher_label_smoothing"] = float(args.teacher_label_smoothing)
    if args.teacher_augment is not None:
        cfg["teacher_augment"] = str(args.teacher_augment)
    return cfg


def run_one_teacher(
    teacher_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
    cfg = make_cfg(args, teacher_name, seed)
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    train_idx, val_idx = ex.stratified_split(df["label"].tolist(), float(cfg["val_frac"]), int(seed))
    start = time.time()
    print(f"\n=== teacher={teacher_name} seed={seed} device={device} train={len(train_idx)} val={len(val_idx)} ===")
    teacher, teacher_fit = ex.train_teacher_model(cfg, train_idx, val_idx, device)

    val_logits, val_labels = collect_labeled_logits_indices(
        teacher, val_idx, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False))
    )
    ce = F.cross_entropy(val_logits, val_labels).item()
    diagnostics = teacher_ranking_and_calibration(val_logits, val_labels)
    diagnostics["teacher_val_loss"] = float(ce)
    diagnostics.update(teacher_tta_stability(teacher, val_idx, device, batch_size=int(cfg["teacher_batch_size"])))

    print("Collecting teacher logits for labeled and unlabeled images...")
    labeled_logits = ex.collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    unlabeled_logits = ex.collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=bool(cfg.get("teacher_tta", False)))
    diagnostics.update(unlabeled_diagnostics(unlabeled_logits))
    diagnostics["teacher_diagnostic_score"] = diagnostic_score(diagnostics)

    record = {
        "type": "teachergrid_teacher",
        "status": "ok",
        "teacher": teacher_name,
        "seed": int(seed),
        "device": str(device),
        "config": cfg,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        **teacher_fit,
        **diagnostics,
    }
    # Checkpoint teacher diagnostics immediately. Distillation can take a long
    # time, and this preserves the teacher-quality evidence if Colab disconnects
    # or the student run is interrupted.
    teacher_checkpoint = {**record, "type": "teachergrid_teacher_checkpoint", "elapsed_sec": round(time.time() - start, 3)}
    log_record(teacher_checkpoint)
    print("TEACHER DIAGNOSTIC CHECKPOINT")
    print(json.dumps({
        "teacher": teacher_name,
        "seed": int(seed),
        "teacher_val_acc": record.get("teacher_val_acc"),
        "teacher_val_loss": record.get("teacher_val_loss"),
        "teacher_top3_acc": record.get("teacher_top3_acc"),
        "teacher_wrong_conf": record.get("teacher_wrong_conf"),
        "teacher_diagnostic_score": record.get("teacher_diagnostic_score"),
        "unlabeled_pseudo_counts": record.get("unlabeled_pseudo_counts"),
    }, indent=2, sort_keys=True))

    model = None
    if not args.no_distill:
        print(f"Distilling fixed student from teacher={teacher_name} seed={seed}...")
        model, student_result = ex.train_kd_student_from_logits(
            cfg, f"teachergrid_{safe_name(teacher_name)}_s{seed}", train_idx, val_idx, labeled_logits, unlabeled_logits, device
        )
        record.update({
            "type": "teachergrid_run",
            **student_result,
            "student_generalization_score": student_generalization_score(student_result),
        })

    record["elapsed_sec"] = round(time.time() - start, 3)
    log_record(record)
    if args.export_best and model is not None:
        # Do not export every model here; final export should be manual after locked confirmation.
        pass
    del teacher, model
    clean_cache()
    return record, labeled_logits if args.keep_logits_for_committee else None, unlabeled_logits if args.keep_logits_for_committee else None


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def run_committee(
    committee_name: str,
    teacher_names: Sequence[str],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    logits_by_teacher: dict,
) -> Optional[dict]:
    missing = [t for t in teacher_names if (t, seed) not in logits_by_teacher]
    if missing:
        print(f"Skipping committee {committee_name}; missing logits for {missing}")
        return None
    cfg = make_cfg(args, committee_name, seed)
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    train_idx, val_idx = ex.stratified_split(df["label"].tolist(), float(cfg["val_frac"]), int(seed))
    labeled_logits = torch.stack([logits_by_teacher[(t, seed)][0] for t in teacher_names], dim=0).mean(0)
    unlabeled_logits = torch.stack([logits_by_teacher[(t, seed)][1] for t in teacher_names], dim=0).mean(0)
    record = {
        "type": "teachergrid_committee",
        "status": "ok",
        "teacher": committee_name,
        "committee_teachers": list(teacher_names),
        "seed": int(seed),
        "device": str(device),
        "config": cfg,
        **unlabeled_diagnostics(unlabeled_logits),
    }
    if not args.no_distill:
        model, student_result = ex.train_kd_student_from_logits(
            cfg, f"teachergrid_{safe_name(committee_name)}_s{seed}", train_idx, val_idx, labeled_logits, unlabeled_logits, device
        )
        record.update({**student_result, "student_generalization_score": student_generalization_score(student_result)})
        del model
    log_record(record)
    clean_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a teacher tournament for KD transfer quality.")
    parser.add_argument("--teachers", default=",".join(DEFAULT_TEACHERS), help="Comma-separated timm teacher names.")
    parser.add_argument("--seeds", default="30", help="Comma-separated split seeds, e.g. 30,31,32.")
    parser.add_argument("--teacher-epochs", type=int, default=int(os.environ.get("TEACHERGRID_TEACHER_EPOCHS", "10")))
    parser.add_argument("--student-epochs", type=int, default=int(os.environ.get("TEACHERGRID_STUDENT_EPOCHS", "75")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TEACHERGRID_BATCH_SIZE", "32")))
    parser.add_argument("--teacher-batch-size", type=int, default=int(os.environ.get("TEACHERGRID_TEACHER_BATCH_SIZE", "8")))
    parser.add_argument("--teacher-lr", type=float, default=None)
    parser.add_argument("--teacher-head-warmup-epochs", type=int, default=int(os.environ.get("TEACHERGRID_HEAD_WARMUP_EPOCHS", "0")))
    parser.add_argument("--teacher-head-lr", type=float, default=None)
    parser.add_argument("--teacher-backbone-lr", type=float, default=None)
    parser.add_argument("--teacher-label-smoothing", type=float, default=None)
    parser.add_argument("--teacher-augment", default=None, help="Override teacher augmentation, e.g. basic,strong,none.")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("TEACHERGRID_NUM_WORKERS", "0")))
    parser.add_argument("--quick", action="store_true", help="One-epoch smoke test, disables TTA.")
    parser.add_argument("--no-tta", action="store_true", help="Disable teacher TTA logit collection.")
    parser.add_argument("--no-distill", action="store_true", help="Only compute teacher diagnostics/logits, do not train student.")
    parser.add_argument("--make-committee", action="store_true", help="After individual teachers, distill average-logit committee of all teachers in this run.")
    parser.add_argument("--keep-going", action="store_true", default=os.environ.get("TEACHERGRID_KEEP_GOING", "1") == "1")
    parser.add_argument("--export-best", action="store_true", help="Reserved; teachergrid does not overwrite model.pt by default.")
    args = parser.parse_args()
    args.keep_logits_for_committee = bool(args.make_committee)

    device = ex.get_device()
    teachers = parse_csv(args.teachers)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    print(f"teachergrid device={device} teachers={teachers} seeds={seeds} quick={args.quick} no_distill={args.no_distill}")
    print(f"results={RESULTS_JSON} jsonl={RESULTS_JSONL}")

    # Ensure student is legal before spending teacher compute.
    params = ex.count_params(ex.build_model(BASE_STUDENT_CFG))
    if params >= MAX_PARAMS:
        raise RuntimeError(f"Base student over parameter cap: {params}")
    print(f"fixed student params={params}")

    logits_by_teacher = {}
    records = []
    for seed in seeds:
        for teacher_name in teachers:
            try:
                rec, lab_logits, un_logits = run_one_teacher(teacher_name, seed, args, device)
                records.append(rec)
                if lab_logits is not None and un_logits is not None:
                    logits_by_teacher[(teacher_name, seed)] = (lab_logits, un_logits)
            except Exception as e:
                err = {
                    "type": "teachergrid_run",
                    "status": "crash",
                    "teacher": teacher_name,
                    "seed": int(seed),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                log_record(err)
                print(err["traceback"])
                clean_cache()
                if not args.keep_going:
                    raise

        if args.make_committee and len(teachers) >= 2:
            committee_name = "committee__" + "__".join(safe_name(t) for t in teachers)
            rec = run_committee(committee_name, teachers, seed, args, device, logits_by_teacher)
            if rec:
                records.append(rec)

    done = {
        "type": "teachergrid_summary",
        "status": "done",
        "n_records": len(records),
        "teachers": teachers,
        "seeds": seeds,
        "quick": bool(args.quick),
    }
    if records:
        scored = [r for r in records if r.get("student_generalization_score") is not None]
        if scored:
            best = max(scored, key=lambda r: float(r["student_generalization_score"]))
            done.update({
                "best_teacher_by_student": best.get("teacher"),
                "best_student_generalization_score": best.get("student_generalization_score"),
                "best_val_acc": best.get("best_val_acc"),
                "best_val_loss": best.get("best_val_loss"),
                "best_stress_mean_acc": best.get("stress_mean_acc"),
                "best_stress_worst_acc": best.get("stress_worst_acc"),
            })
        else:
            best = max(records, key=lambda r: float(r.get("teacher_diagnostic_score") or -1e9))
            done.update({
                "best_teacher_by_diagnostics": best.get("teacher"),
                "best_teacher_diagnostic_score": best.get("teacher_diagnostic_score"),
            })
    log_record(done)
    print("\nTEACHERGRID SUMMARY")
    print(json.dumps(done, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
