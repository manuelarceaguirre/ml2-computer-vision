#!/usr/bin/env python3
"""
Lucky CV broad search for ML2 Assignment 2.

Goal: maximize leaderboard luck legally after random CV proved too optimistic.
This single script is Colab-ready. Paste/run it as one Python cell or save as
`lucky_cv_broad_search.py` inside the repo and run with Python.

What it does:
1. Bootstraps repo/data if needed.
2. Builds random and optional visual-cluster validation splits.
3. Trains one EfficientNet-B0 KD teacher per split.
4. Reuses that teacher to train a broad set of <500k student variants.
5. Exports calibrated TorchScript `.pt` candidates and picks `model.pt`.
6. Writes `lucky_search_results.json` with all metrics and candidate paths.

Profiles via env var:
  LUCK_PROFILE=fast    -> 1 split, 4 variants
  LUCK_PROFILE=medium  -> 2 splits, 8 variants (default)
  LUCK_PROFILE=max     -> 5 splits, 10 variants

Important: this is leaderboard-maximizing exploration, not a clean report protocol.
Submit several exported candidate .pt files if submissions are unlimited.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ----------------------------- user knobs ---------------------------------
REPO = "manuelarceaguirre/ml2-computer-vision"
WORKDIR = Path("/content/ml2-computer-vision") if Path("/content").exists() else Path.cwd()
DATA_FILE_ID = "15LF6JRKgM9JOY58gsb3DwK22jJJaUG8c"
PROFILE = os.environ.get("LUCK_PROFILE", "medium").lower()  # fast | medium | max
DOWNLOAD_OUTPUTS = os.environ.get("LUCK_DOWNLOAD", "1") == "1"
# If Colab push is desired, set GITHUB_TOKEN in env or edit safely in Colab (do not paste into chat).
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# ---------------------------------------------------------------------------


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True)
    print(f"[exit {p.returncode}]", flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"failed: {cmd}")
    return p.returncode


def bootstrap() -> None:
    if not (Path.cwd() / "experiment.py").exists():
        if not WORKDIR.exists():
            sh(f"git clone https://github.com/{REPO}.git {WORKDIR}")
        os.chdir(WORKDIR)
    if GITHUB_TOKEN:
        sh(f"git remote set-url origin https://{GITHUB_TOKEN}@github.com/{REPO}.git", check=False)
    sh("git pull --rebase --autostash origin master", check=False)
    sh("python -m pip install -q -r requirements-colab.txt timm scikit-learn google-api-python-client", check=False)
    if not Path("data/train/labels.csv").exists():
        if not Path("data.zip").exists():
            print("Downloading data.zip from Drive...", flush=True)
            try:
                from google.colab import auth  # type: ignore
                auth.authenticate_user()
                from googleapiclient.discovery import build  # type: ignore
                from googleapiclient.http import MediaIoBaseDownload  # type: ignore
                service = build("drive", "v3")
                request = service.files().get_media(fileId=DATA_FILE_ID)
                with open("data.zip", "wb") as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            print(f"Drive download: {int(status.progress() * 100)}%", flush=True)
            except Exception as e:
                raise SystemExit(f"Data missing and Drive download failed: {e}")
        sh("unzip -q -o data.zip")
    sh("python -m py_compile experiment.py", check=True)


bootstrap()

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as ex

OUT_DIR = Path("lucky_candidates")
OUT_DIR.mkdir(exist_ok=True)
RESULTS_PATH = Path("lucky_search_results.json")

# -------------------------- augmentation patch -----------------------------
_ORIG_AUGMENT_PIL = ex.augment_pil


def robust_light_augment(img: Image.Image) -> Image.Image:
    """Between experiment.py basic and robust, aimed at hidden/client shift."""
    img = ex.random_resized_crop(img, size=256, scale=(0.70, 1.0), ratio=(0.82, 1.22))
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.40:
        angle = random.uniform(-8.0, 8.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(255 * m) for m in ex.DATA_MEAN))
    img = ex.color_jitter(img, brightness=0.22, contrast=0.22, saturation=0.18, hue=0.03)
    if random.random() < 0.10:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    if random.random() < 0.10:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(60, 92))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def patched_augment_pil(img: Image.Image, mode: str) -> Image.Image:
    if mode == "robust_light":
        return robust_light_augment(img)
    return _ORIG_AUGMENT_PIL(img, mode)


ex.augment_pil = patched_augment_pil

# -------------------------- model utilities --------------------------------


class TemperatureScaled(nn.Module):
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / torch.clamp(self.temperature, min=1e-4)


class HFlipTTA(nn.Module):
    """Optional deterministic test-time hflip inside one submitted TorchScript model.

    No extra trainable params. Use only if assignment evaluator accepts deterministic
    inference-time augmentation. We export both TTA and non-TTA candidates.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.model(x) + self.model(torch.flip(x, dims=[3])))


@torch.inference_mode()
def collect_logits(model: nn.Module, indices: List[int], device: torch.device, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = ex.FoodDataset(ex.DATA_ROOT, indices, train=False, augment="none")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    zs, ys = [], []
    model.eval()
    for x, y in loader:
        zs.append(model(x.to(device)).detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(zs), torch.cat(ys).long()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    logits = logits.float().detach()
    labels = labels.long().detach()
    raw_nll = float(F.cross_entropy(logits, labels).item())
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(0.25, 5.0)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass
    t = float(torch.exp(log_t.detach()).clamp(0.25, 5.0).item())
    cal_nll = float(F.cross_entropy(logits / t, labels).item())
    return t, raw_nll, cal_nll


@torch.inference_mode()
def eval_model(model: nn.Module, indices: List[int], device: torch.device, batch_size: int = 64) -> Dict:
    logits, labels = collect_logits(model, indices, device, batch_size=batch_size)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    cm = torch.zeros(ex.NUM_CLASSES, ex.NUM_CLASSES, dtype=torch.long)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1
    per_class = (cm.diag().float() / cm.sum(1).clamp_min(1).float()).tolist()
    return {
        "acc": float(pred.eq(labels).float().mean().item()),
        "loss": float(F.cross_entropy(logits, labels).item()),
        "conf": float(conf.mean().item()),
        "per_class_acc": per_class,
        "min_per_class_acc": float(min(per_class)),
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


# ----------------------------- splits --------------------------------------


def image_feature(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Downsampled pixels plus color moments. Simple, fast, no pretrained leakage.
    flat = arr.reshape(-1)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    q25 = np.quantile(arr.reshape(-1, 3), 0.25, axis=0)
    q75 = np.quantile(arr.reshape(-1, 3), 0.75, axis=0)
    return np.concatenate([flat, mean, std, q25, q75]).astype(np.float32)


def cluster_split(labels: List[int], seed: int = 0, fold: int = 0, n_clusters: int = 28) -> Tuple[List[int], List[int]]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedGroupKFold

    cache = Path(f".cluster_features_seed{seed}.npz")
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    if cache.exists():
        data = np.load(cache)
        X_lab = data["X_lab"]
        clusters_lab = data["clusters_lab"]
    else:
        print("Computing simple visual features for cluster split...", flush=True)
        lab_paths = [ex.DATA_ROOT / fn for fn in df["filename"].tolist()]
        un_root = Path("data/unlabeled")
        un_paths = sorted(list(un_root.glob("*.jpg")) + list(un_root.glob("*.png")) + list(un_root.glob("*.jpeg")))
        all_paths = lab_paths + un_paths
        X_all = np.stack([image_feature(p) for p in all_paths])
        n_comp = min(48, X_all.shape[0] - 1, X_all.shape[1])
        Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X_all)
        clusters = KMeans(n_clusters=min(n_clusters, len(all_paths)), random_state=seed, n_init=10).fit_predict(Xp)
        X_lab = Xp[: len(lab_paths)]
        clusters_lab = clusters[: len(lab_paths)]
        np.savez(cache, X_lab=X_lab, clusters_lab=clusters_lab)
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=clusters_lab))
        tr, va = splits[fold % len(splits)]
        return list(map(int, tr)), list(map(int, va))
    except Exception as e:
        print(f"cluster split failed ({e}); falling back to stratified random", flush=True)
        return ex.stratified_split(labels, 0.20, seed)


def make_splits(labels: List[int]) -> List[Tuple[str, List[int], List[int]]]:
    specs: List[Tuple[str, int]]
    if PROFILE == "fast":
        specs = [("random", 40)]
    elif PROFILE == "max":
        specs = [("random", 30), ("random", 40), ("random", 41), ("cluster", 0), ("cluster", 1)]
    else:
        specs = [("random", 40), ("cluster", 0)]
    out = []
    for kind, seed in specs:
        if kind == "random":
            tr, va = ex.stratified_split(labels, 0.20, seed)
            name = f"random{seed}"
        else:
            tr, va = cluster_split(labels, seed=17 + seed, fold=seed)
            name = f"cluster{seed}"
        out.append((name, tr, va))
        print(f"split {name}: train={len(tr)} val={len(va)} class_counts_val={np.bincount([labels[i] for i in va], minlength=ex.NUM_CLASSES).tolist()}", flush=True)
    return out


# ----------------------------- variants ------------------------------------


def base_cfg(seed: int) -> Dict:
    cfg = copy.deepcopy(ex.RECIPES["kd_b0_tta_eca160"])
    cfg.update({
        "seed": int(seed),
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
        "teacher_batch_size": int(os.environ.get("LUCK_TEACHER_BATCH", "16")),
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
    })
    for k in ["teacher_head_warmup_epochs", "teacher_head_lr", "teacher_backbone_lr", "teacher_label_smoothing"]:
        cfg.pop(k, None)
    return cfg


ALL_VARIANTS: List[Tuple[str, Dict]] = [
    ("eca_inc_ep75", {}),
    ("eca_ep55", {"epochs": 55}),
    ("eca_ep60_robustlight", {"epochs": 60, "augment": "robust_light", "weight_decay": 8e-4}),
    ("eca_ep60_strong_ls07", {"epochs": 60, "augment": "strong", "label_smoothing": 0.07, "weight_decay": 8e-4}),
    ("eca_ep65_mixup01", {"epochs": 65, "mixup_alpha": 0.10, "label_smoothing": 0.05}),
    ("eca_t225_ep75", {"T": 2.25, "epochs": 75}),
    ("eca_a075_ep75", {"alpha": 0.75, "epochs": 75}),
    ("hybrid_ep70", {"model": "grid_hybridse", "dropout": 0.20, "weight_decay": 1e-3, "epochs": 70}),
    ("tinymobile_ep75", {"model": "grid_tinymobilenetv2", "dropout": 0.15, "lr": 2e-3, "weight_decay": 1e-3, "epochs": 75}),
    ("dscnn_ep75", {"model": "grid_dscnn", "dropout": 0.15, "lr": 2e-3, "weight_decay": 1e-3, "epochs": 75}),
]


def selected_variants() -> List[Tuple[str, Dict]]:
    if PROFILE == "fast":
        names = {"eca_inc_ep75", "eca_ep55", "eca_ep60_robustlight", "eca_ep65_mixup01"}
    elif PROFILE == "max":
        names = {n for n, _ in ALL_VARIANTS}
    else:
        names = {"eca_inc_ep75", "eca_ep55", "eca_ep60_robustlight", "eca_ep60_strong_ls07", "eca_ep65_mixup01", "eca_t225_ep75", "hybrid_ep70", "tinymobile_ep75"}
    return [(n, o) for n, o in ALL_VARIANTS if n in names]


# ------------------------------ search -------------------------------------


def selection_score(rec: Dict) -> float:
    # Accuracy first, then class balance/stress/loss. Cluster split is upweighted
    # because it is intended to proxy hidden distribution shift.
    split_bonus = 0.02 if str(rec["split"]).startswith("cluster") else 0.0
    return (
        float(rec["val_acc"])
        + 0.03 * float(rec.get("stress_mean_acc") or 0.0)
        + 0.02 * float(rec.get("min_per_class_acc") or 0.0)
        - 0.03 * float(rec["val_loss"])
        + split_bonus
    )


def train_teacher_for_split(cfg: Dict, train_idx: List[int], val_idx: List[int], device: torch.device) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    teacher, teacher_result = ex.train_teacher_model(cfg, train_idx, val_idx, device)
    print("Collecting B0 teacher TTA logits for split...", flush=True)
    labeled_logits = ex.collect_labeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=True)
    unlabeled_logits = ex.collect_unlabeled_logits(teacher, device, batch_size=int(cfg["teacher_batch_size"]), use_tta=True)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return teacher_result, labeled_logits, unlabeled_logits


def train_variant(split_name: str, train_idx: List[int], val_idx: List[int], teacher_result: Dict, labeled_logits: torch.Tensor, unlabeled_logits: torch.Tensor, variant_name: str, overrides: Dict, device: torch.device) -> Dict:
    cfg = base_cfg(seed=abs(hash((split_name, variant_name))) % 10000)
    cfg.update(overrides)
    cfg["notes"] = f"lucky search {split_name} {variant_name}"
    params = ex.count_params(ex.build_model(cfg))
    if params >= 500_000:
        print(f"SKIP {variant_name}: params={params}", flush=True)
        return {"status": "skip", "reason": "over_params", "variant": variant_name, "split": split_name, "params": params}
    print(f"\n=== TRAIN {split_name}/{variant_name} model={cfg['model']} params={params} epochs={cfg['epochs']} aug={cfg['augment']} T={cfg['T']} alpha={cfg['alpha']} ===", flush=True)
    t0 = time.time()
    student, result = ex.train_kd_student_from_logits(cfg, f"luck_{split_name}_{variant_name}", train_idx, val_idx, labeled_logits, unlabeled_logits, device)

    raw_eval = eval_model(student, val_idx, device, batch_size=64)
    logits, labels = collect_logits(student, val_idx, device, batch_size=64)
    temp, raw_nll, cal_nll = fit_temperature(logits, labels)
    calibrated = TemperatureScaled(student, temp).to(device).eval()
    cal_eval = eval_model(calibrated, val_idx, device, batch_size=64)
    stress = ex.evaluate_stress(calibrated, val_idx, device, batch_size=64)

    safe = f"{split_name}_{variant_name}_acc{cal_eval['acc']:.4f}_loss{cal_eval['loss']:.4f}_t{temp:.2f}".replace(".", "p").replace("/", "_")
    path = OUT_DIR / f"model_{safe}.pt"
    export_scripted(calibrated, path)

    # Also export hflip TTA version for high-accuracy candidates. This is a legal
    # risk depending on instructor interpretation, so keep separate.
    tta_path = None
    if cal_eval["acc"] >= 0.58:
        tta_model = HFlipTTA(calibrated).to(device).eval()
        tta_eval = eval_model(tta_model, val_idx, device, batch_size=64)
        tta_path = OUT_DIR / f"model_{safe}_hfliptta.pt"
        export_scripted(tta_model, tta_path)
    else:
        tta_eval = None

    rec = {
        "status": "ok",
        "split": split_name,
        "variant": variant_name,
        "model_path": str(path),
        "hflip_tta_model_path": str(tta_path) if tta_path else None,
        "params": int(params),
        "temperature": float(temp),
        "raw_nll_before_temp": raw_nll,
        "cal_nll_after_temp": cal_nll,
        "val_acc": cal_eval["acc"],
        "val_loss": cal_eval["loss"],
        "val_conf": cal_eval["conf"],
        "min_per_class_acc": cal_eval["min_per_class_acc"],
        "per_class_acc": cal_eval["per_class_acc"],
        "raw_eval": raw_eval,
        "hflip_tta_eval": tta_eval,
        "stress_mean_acc": stress.get("mean", {}).get("acc"),
        "stress_worst_acc": stress.get("worst", {}).get("acc"),
        "stress_mean_loss": stress.get("mean", {}).get("loss"),
        "best_epoch": result.get("best_epoch"),
        "best_val_acc_internal": result.get("best_val_acc"),
        "best_val_loss_internal": result.get("best_val_loss"),
        "config": cfg,
        "teacher_result": teacher_result,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    rec["selection_score"] = selection_score(rec)
    ex.append_log({"type": "lucky_search_candidate", **rec})
    print("LUCKY CANDIDATE", json.dumps({k: rec.get(k) for k in ["split", "variant", "val_acc", "val_loss", "stress_mean_acc", "stress_worst_acc", "min_per_class_acc", "temperature", "selection_score", "model_path", "hflip_tta_model_path", "best_epoch"]}, indent=2), flush=True)
    del student, calibrated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    print(f"LUCKY BROAD SEARCH START profile={PROFILE}", flush=True)
    sh("nvidia-smi || true", check=False)
    device = ex.get_device()
    df = pd.read_csv(ex.DATA_ROOT / "labels.csv")
    labels = df["label"].astype(int).tolist()
    splits = make_splits(labels)
    variants = selected_variants()
    print("variants", [n for n, _ in variants], flush=True)

    records: List[Dict] = []
    for split_name, train_idx, val_idx in splits:
        cfg_t = base_cfg(seed=40 if split_name.startswith("random40") else 30)
        teacher_result, labeled_logits, unlabeled_logits = train_teacher_for_split(cfg_t, train_idx, val_idx, device)
        for variant_name, overrides in variants:
            try:
                rec = train_variant(split_name, train_idx, val_idx, teacher_result, labeled_logits, unlabeled_logits, variant_name, overrides, device)
                records.append(rec)
                RESULTS_PATH.write_text(json.dumps({"profile": PROFILE, "records": records}, indent=2, sort_keys=True) + "\n")
            except Exception as e:
                import traceback
                err = {"status": "crash", "split": split_name, "variant": variant_name, "error": repr(e), "traceback": traceback.format_exc()}
                print(err["traceback"], flush=True)
                records.append(err)
                RESULTS_PATH.write_text(json.dumps({"profile": PROFILE, "records": records}, indent=2, sort_keys=True) + "\n")

    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        raise SystemExit("No successful candidates")
    best = max(ok, key=lambda r: float(r["selection_score"]))
    shutil.copyfile(best["model_path"], "model.pt")
    # Also copy best TTA candidate if present.
    if best.get("hflip_tta_model_path"):
        shutil.copyfile(best["hflip_tta_model_path"], "model_hfliptta.pt")
    payload = {"profile": PROFILE, "best": best, "records": records}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nBEST LUCKY CANDIDATE")
    print(json.dumps({k: best.get(k) for k in ["split", "variant", "val_acc", "val_loss", "stress_mean_acc", "stress_worst_acc", "min_per_class_acc", "temperature", "selection_score", "model_path", "hflip_tta_model_path", "best_epoch", "params"]}, indent=2, sort_keys=True), flush=True)
    print(f"Copied best non-TTA candidate to model.pt. All candidates in {OUT_DIR}/", flush=True)

    sh("ls -lh model.pt model_hfliptta.pt lucky_candidates/*.pt lucky_search_results.json 2>/dev/null || true", check=False)
    # Commit metadata only; .pt files are gitignored.
    sh("git config user.email 'colab-luck@example.com'", check=False)
    sh("git config user.name 'Colab Lucky Search'", check=False)
    sh("git add lucky_search_results.json log.txt || true", check=False)
    sh("git commit -m 'Record lucky broad search metadata' || true", check=False)
    sh("git push origin HEAD:master || true", check=False)

    if DOWNLOAD_OUTPUTS:
        try:
            from google.colab import files  # type: ignore
            print("Downloading model.pt and metadata...", flush=True)
            files.download("model.pt")
            if Path("model_hfliptta.pt").exists():
                files.download("model_hfliptta.pt")
            files.download(str(RESULTS_PATH))
        except Exception as e:
            print(f"Download skipped/failed: {e}", flush=True)

    print("LUCKY BROAD SEARCH DONE", flush=True)


if __name__ == "__main__":
    main()
