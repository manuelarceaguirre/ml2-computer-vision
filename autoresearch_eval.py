#!/usr/bin/env python3
"""Autoresearch benchmark wrapper for ML2 CV robust generalization.

Reads autoresearch_candidate.json, applies global KD recipe knobs, runs local
cross-validation, and prints METRIC lines for pi-autoresearch.
"""
from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "autoresearch_candidate.json"
LOG = ROOT / "log.txt"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def last_kd_summary(before_lines: int) -> dict:
    lines = LOG.read_text(encoding="utf-8").splitlines() if LOG.exists() else []
    for line in reversed(lines[before_lines:]):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "kd_cv_run":
            return rec
    raise RuntimeError("experiment.py did not append a kd_cv_run summary to log.txt")


def main() -> int:
    cfg = load_json(CANDIDATE)
    recipe = str(cfg.get("recipe", "kd_b0_tta_eca160"))
    folds = int(cfg.get("seeds", 3))
    start_seed = int(cfg.get("start_seed", 30))
    teacher_epochs = int(cfg.get("teacher_epochs", 10))
    checkpoint_epochs = [int(x) for x in cfg.get("checkpoint_epochs", [65])]
    student_epochs = int(cfg.get("epochs", cfg.get("selected_epoch", max(checkpoint_epochs))))

    # Import lazily after py_compile so syntax failures are separated from runtime.
    import experiment  # type: ignore

    if recipe not in experiment.RECIPES:
        raise SystemExit(f"Unknown recipe {recipe}")

    original = copy.deepcopy(experiment.RECIPES[recipe])
    patched = copy.deepcopy(original)
    patched["seed"] = start_seed
    patched["teacher_epochs"] = teacher_epochs
    patched["epochs"] = student_epochs
    if "T" in cfg:
        patched["T"] = float(cfg["T"])
    if "alpha" in cfg:
        patched["alpha"] = float(cfg["alpha"])
    if "labeled" in cfg:
        patched["labeled_kd_weight"] = float(cfg["labeled"])
    if "unlabeled" in cfg:
        patched["unlabeled_kd_weight"] = float(cfg["unlabeled"])
    if "teacher_tta" in cfg:
        patched["teacher_tta"] = bool(cfg["teacher_tta"])

    # Optional broad recipe overrides. These let autoresearch move from the
    # original B0 recipe into the MaxViT-teacher branch without rewriting
    # experiment.py for each candidate. Keep this list global/non-class-specific.
    float_fields = [
        "teacher_lr", "teacher_head_lr", "teacher_backbone_lr", "teacher_weight_decay",
        "teacher_label_smoothing", "lr", "weight_decay", "dropout", "mixup_alpha",
        "label_smoothing", "unlabeled_hard_weight", "unlabeled_conf_threshold",
    ]
    int_fields = [
        "teacher_batch_size", "teacher_head_warmup_epochs", "batch_size",
        "input_size", "head_ch",
    ]
    str_fields = ["teacher_model", "teacher_input_size", "teacher_augment", "augment", "model"]
    bool_fields = ["normalize_kd"]
    for key in float_fields:
        if key in cfg:
            patched[key] = float(cfg[key])
    for key in int_fields:
        if key in cfg:
            patched[key] = int(cfg[key])
    for key in str_fields:
        if key in cfg:
            patched[key] = str(cfg[key])
    for key in bool_fields:
        if key in cfg:
            patched[key] = bool(cfg[key])
    # More readable alias in candidate JSON.
    if "student_model" in cfg:
        patched["model"] = str(cfg["student_model"])

    before_lines = len(LOG.read_text(encoding="utf-8").splitlines()) if LOG.exists() else 0
    old_log = LOG.read_text(encoding="utf-8") if LOG.exists() else None
    try:
        experiment.RECIPES[recipe] = patched
        experiment.run_kd_cv(recipe, folds=folds, epochs_override=student_epochs, teacher_epochs_override=teacher_epochs)
        summary = last_kd_summary(before_lines)
    finally:
        experiment.RECIPES[recipe] = original
        # Keep benchmark output clean; assignment log should not drift during search.
        if old_log is not None:
            LOG.write_text(old_log, encoding="utf-8")
        elif LOG.exists():
            LOG.unlink()

    mean_val_acc = float(summary["mean_val_acc"])
    cal_loss = float(summary.get("mean_val_loss", 0.0))
    stress_mean = float(summary.get("stress_mean_acc", 0.0))
    stress_worst = float(summary.get("stress_worst_acc", 0.0))
    fold_epochs = [float(r.get("best_epoch", student_epochs)) for r in summary.get("fold_records", []) if isinstance(r, dict)]
    selected_epoch = float(statistics.median(fold_epochs)) if fold_epochs else float(student_epochs)
    cal_target = float(cfg.get("scoring", {}).get("cal_loss_target", 1.20))
    generalization_score = mean_val_acc + 0.20 * stress_mean + 0.10 * stress_worst - 0.03 * max(0.0, cal_loss - cal_target)

    print("AUTORESEARCH SUMMARY")
    print(json.dumps({
        "description": cfg.get("description"),
        "recipe": recipe,
        "folds": folds,
        "student_epochs": student_epochs,
        "teacher_epochs": teacher_epochs,
        "teacher_model": patched.get("teacher_model"),
        "student_model": patched.get("model"),
        "T": patched.get("T"),
        "alpha": patched.get("alpha"),
        "labeled": patched.get("labeled_kd_weight"),
        "unlabeled": patched.get("unlabeled_kd_weight"),
        "mean_val_acc": mean_val_acc,
        "cal_loss": cal_loss,
        "stress_mean": stress_mean,
        "stress_worst": stress_worst,
        "generalization_score": generalization_score,
    }, indent=2, sort_keys=True))
    print(f"METRIC generalization_score={generalization_score:.8f}")
    print(f"METRIC mean_val_acc={mean_val_acc:.8f}")
    print(f"METRIC cal_loss={cal_loss:.8f}")
    print(f"METRIC stress_mean={stress_mean:.8f}")
    print(f"METRIC stress_worst={stress_worst:.8f}")
    print(f"METRIC selected_epoch={selected_epoch:.2f}")
    print(f"METRIC tau=1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
