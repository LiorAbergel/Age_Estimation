"""
Experiment 06: Document Image Transformers (DiT) for Age Estimation

Overview:
This script trains Document Image Transformer models (microsoft/dit-base, dit-large)
for the age estimation task. It treats the problem as a regression task on
bags of patches extracted from high-resolution handwriting images.

Key Features:
- Architecture: DiT Backbone (via Hugging Face) + Linear Regression Head.
- Data: Dynamic patch extraction with intensity filtering.
- Training: Two-stage (Frozen -> Fine-tuning) optimization aligned with Exp 01/03/04.

Pipeline (matched to reference experiments):
- Phase 1: Frozen backbone, Adam(1e-4), up to 50 epochs.
- Phase 2: All layers unfrozen, Adam(1e-5), up to 10 epochs.
- Both phases use ReduceLROnPlateau(factor=0.1, patience=5) and EarlyStopping(patience=10).
- Best checkpoint is carried forward: Phase 2 only overwrites if it beats Phase 1.

Requirements:
pip install transformers torch torchvision pandas numpy scikit-learn
"""

import os
import csv
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModel, BeitImageProcessor
from PIL import Image
from tqdm.auto import tqdm
import itertools

# Set seeds for reproducibility
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- Suppress harmless DataLoader worker teardown traceback on Colab ---
# Colab's forked finalizer triggers "AssertionError: can only test a child
# process" inside _MultiProcessingDataLoaderIter.__del__.  The error is
# cosmetic (printed by Python's GC, not raised) and does not affect training.
# Wrap __del__ to swallow it silently.
import torch.utils.data.dataloader as _dl_mod
_orig_dl_del = _dl_mod._MultiProcessingDataLoaderIter.__del__
def _quiet_dl_del(self):
    try:
        _orig_dl_del(self)
    except Exception:
        pass
_dl_mod._MultiProcessingDataLoaderIter.__del__ = _quiet_dl_del

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from download_dataset import ensure_dataset

EXPERIMENT_DIRNAME = "06_DiT"

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": 400,
    "STRIDE": 200,
    "STANDARD_SIZE": 800,
    # Patch-level batching: the DataLoader batches *patches* (like the CNN/ViT
    # flat_map -> batch(128) pipeline), so the effective optimizer step is 128 patches.
    "TARGET_EFFECTIVE_BATCH_SIZE": 128,  # effective patches per optimizer step
    "BATCH_SIZE_BASE": 128,              # physical patch batch for DiT-Base (accum 1)
    "BATCH_SIZE_LARGE": 16,              # physical patch batch for DiT-Large (accum 8)
    "EVAL_BATCH_SIZE": 128,              # inference patch batch (no effect on results)
    "EPOCHS_PHASE1": 50,
    "EPOCHS_PHASE2": 10,
    "LR_INIT": 1e-3,   # frozen-head LR (same as CNN/ViT aligned pipeline)
    "LR_FT": 1e-4,     # fine-tune LR (same as CNN/ViT aligned pipeline)
    "THR": 0.0054,
    "DATA_DIR": os.path.join(REPO_ROOT, "data"),
    "CSV_PATH": os.path.join(REPO_ROOT, "data", "NewAgeSplit.csv"),
    "MODELS_DIR": os.path.join(REPO_ROOT, "models", "experiment_06"),
    "RESULTS_DIR": os.path.join(REPO_ROOT, "results", "experiment_06"),
    "VAL_PREDICTIONS_CSV": "val_image_level_predictions.csv",
    "TEST_PREDICTIONS_CSV": "test_image_level_predictions.csv",
    "SUMMARY_CSV": "ensemble_evaluation_summary.csv",
}

# Define models to train
# (HuggingFace Model ID)
MODELS_TO_TRAIN = [
    "microsoft/dit-base",
    "microsoft/dit-large",
    "microsoft/dit-base-finetuned-rvlcdip",
    "microsoft/dit-large-finetuned-rvlcdip",
]

# --- Utils ---

def get_batch_size(model_id):
    return CONFIG["BATCH_SIZE_LARGE"] if "large" in model_id else CONFIG["BATCH_SIZE_BASE"]

def get_accum_steps(model_id):
    batch_size = get_batch_size(model_id)
    target = CONFIG["TARGET_EFFECTIVE_BATCH_SIZE"]
    if target % batch_size != 0:
        raise ValueError(f"TARGET_EFFECTIVE_BATCH_SIZE={target} must be divisible by batch_size={batch_size}")
    return target // batch_size

def calculate_resized_dimensions(height, width):
    aspect_ratio = width / height
    if height < width:
        new_height = CONFIG["STANDARD_SIZE"]
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = CONFIG["STANDARD_SIZE"]
        new_height = int(new_width / aspect_ratio)

    def adjust_dimension(dim):
        remainder = (dim - CONFIG["PATCH_SIZE"]) % CONFIG["STRIDE"]
        return dim if remainder == 0 else dim - remainder

    return adjust_dimension(new_height), adjust_dimension(new_width)

class HHDPatchStream(IterableDataset):
    """Streams individual patches (one image at a time) so the DataLoader batches
    *patches*, not images -- mirroring the CNN/ViT ``flat_map(...).batch(N)``
    pipeline (fixed image order, no shuffle). DiT-Large reaches the same effective
    patch batch via gradient accumulation. Memory stays low (one image in flight).
    """
    def __init__(self, df: pd.DataFrame, root: str, processor, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment
        target_size = (processor.size["height"], processor.size["width"])
        # Replaces the per-image processor() call; identical ImageNet/BEiT normalization.
        normalize = transforms.Normalize(processor.image_mean, processor.image_std)

        # Augmentation pipeline (matched to reference experiments)
        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(target_size, scale=(0.9, 1.1), ratio=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.25),
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1)),
            normalize,
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(target_size),
            normalize,
        ])

    def _valid_patches(self, row):
        """Return this image's non-empty 400x400 patches as (N,3,H,W), or None."""
        img_path = self.root / str(row["Set"]) / row["File"]
        if not img_path.exists():
            # Fallback for different folder structures (lowercase/uppercase)
            img_path = self.root / str(row["Set"]).lower() / row["File"]
            if not img_path.exists():
                return None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

        w, h = img.size
        new_h, new_w = calculate_resized_dimensions(h, w)
        # Ensure dimensions are at least patch size
        new_h = max(new_h, CONFIG["PATCH_SIZE"])
        new_w = max(new_w, CONFIG["PATCH_SIZE"])
        img_t = transforms.ToTensor()(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

        # Unfold to (N_Patches, 3, PatchH, PatchW)
        patches = img_t.unfold(1, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])\
                       .unfold(2, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, CONFIG["PATCH_SIZE"], CONFIG["PATCH_SIZE"])

        # Filter empty patches
        valid = patches[patches.mean(dim=[1, 2, 3]) > CONFIG["THR"]]
        return valid if valid.size(0) > 0 else None

    def __iter__(self):
        # Shard images across workers so no patch is yielded twice.
        worker = torch.utils.data.get_worker_info()
        indices = range(len(self.df))
        if worker is not None:
            indices = list(indices)[worker.id::worker.num_workers]

        tfm = self.aug_transforms if self.augment else self.val_transforms
        for idx in indices:
            row = self.df.iloc[idx]
            valid = self._valid_patches(row)
            if valid is None:
                continue
            label = torch.tensor(float(row["Age"]), dtype=torch.float32)
            fid = row["File"]
            for i in range(valid.size(0)):
                yield {"pixel_values": tfm(valid[i]),
                       "label": label, "id": fid}


def collate_patches(batch: List[Any]) -> Dict[str, Any]:
    """Stack individual patch samples into one fixed-size patch batch."""
    batch = [b for b in batch if b]
    if not batch:
        return {}
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "ids": [b["id"] for b in batch],
    }

# --- Model ---

class DiTReg(nn.Module):
    def __init__(self, name: str = "microsoft/dit-base", p: float = 0.5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Sequential(
            nn.Dropout(p), 
            nn.Linear(self.backbone.config.hidden_size, 1)
        )

    def forward(self, pixel_values, labels=None):
        # Forward pass through DiT
        outputs = self.backbone(pixel_values=pixel_values)
        # Global average pooling over patch tokens (exclude CLS at index 0),
        # matching the GlobalAveragePooling head used by the CNN/ViT experiments
        # and the BEiT-recommended mean-pooling (its CLS token is not pretrained).
        pooled = outputs.last_hidden_state[:, 1:].mean(dim=1)
        pred = self.head(pooled).squeeze(1)

        out = {"preds": pred}
        if labels is not None:
            out["loss"] = nn.functional.mse_loss(pred, labels)
        return out

# --- Training & Evaluation ---

def train_one_epoch(model, loader, opt, device, accum_steps):
    model.train()
    total_loss, total_samples = 0.0, 0
    pending_steps = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    opt.zero_grad()

    for batch in pbar:
        if not batch: continue

        px = batch["pixel_values"].to(device)
        lbl = batch["labels"].to(device)

        out = model(pixel_values=px, labels=lbl)
        loss = out["loss"] / accum_steps
        loss.backward()
        pending_steps += 1

        if pending_steps == accum_steps:
            opt.step()
            opt.zero_grad()
            pending_steps = 0

        bs = lbl.size(0)
        total_loss += out["loss"].item() * bs
        total_samples += bs

    if pending_steps > 0:
        opt.step()
        opt.zero_grad()

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate_patch_mae(model, loader, device):
    """Patch-level validation MAE, used for checkpoint selection / early stopping.

    Matches the CNN/ViT experiments, where the best model is chosen on the
    patch-level validation metric. Page-level metrics (the reported numbers) are
    computed separately from the aggregated prediction CSVs.
    """
    model.eval()
    tot_abs, n = 0.0, 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if not batch: continue
        px = batch["pixel_values"].to(device)
        labels = batch["labels"].cpu().numpy().reshape(-1)
        preds = model(pixel_values=px)["preds"].cpu().numpy().reshape(-1)
        tot_abs += float(np.abs(preds - labels).sum())
        n += labels.shape[0]
    return {"MAE": tot_abs / n if n else 0.0}

# --- Ensemble Logic ---

# Human-readable names for display
MODEL_DISPLAY_NAMES = {
    "microsoft/dit-base": "DiT-Base",
    "microsoft/dit-large": "DiT-Large",
    "microsoft/dit-base-finetuned-rvlcdip": "DiT-Base (RVL-CDIP)",
    "microsoft/dit-large-finetuned-rvlcdip": "DiT-Large (RVL-CDIP)",
}

def compute_full_metrics(y_true, y_pred):
    """Compute all evaluation metrics (keys aligned with Exp 03/04/05)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.isnan(mape): mape = 0.0
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "Acc_2yr": float(np.mean(errors <= 2) * 100),
        "Acc_5yr": float(np.mean(errors <= 5) * 100),
        "Acc_10yr": float(np.mean(errors <= 10) * 100),
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }

@torch.no_grad()
def generate_predictions(model, loader, device):
    """Generate image-level predictions. Returns dict {image_id: predicted_age}."""
    model.eval()
    img_preds = defaultdict(list)

    for batch in tqdm(loader, desc="Predicting", leave=False):
        if not batch: continue
        px = batch["pixel_values"].to(device)
        ids = batch["ids"]
        preds = model(pixel_values=px)["preds"].cpu().numpy()
        for fid, p in zip(ids, preds):
            img_preds[fid].append(p)

    return {fid: float(np.mean(plist)) for fid, plist in img_preds.items()}

def select_ensemble_weights(val_predictions, true_age_dict):
    """Select ensemble weights using grid search and MAE-based methods on validation data."""

    # Sort models by val MAE
    model_ids = list(val_predictions.keys())
    model_val_maes = {}
    for model_id in model_ids:
        preds = val_predictions[model_id]
        common = [fid for fid in preds if fid in true_age_dict]
        y_true = [true_age_dict[fid] for fid in common]
        y_pred = [preds[fid] for fid in common]
        model_val_maes[model_id] = mean_absolute_error(y_true, y_pred)

    ranked_models = sorted(model_val_maes.keys(), key=lambda m: model_val_maes[m])
    print("\nModel ranking by Val MAE:")
    for m in ranked_models:
        print(f"  {MODEL_DISPLAY_NAMES.get(m, m)}: {model_val_maes[m]:.3f}")

    # Define ensemble groups based on val MAE ranking
    ensemble_groups = {}
    if len(ranked_models) >= 2:
        ensemble_groups["Best 2"] = ranked_models[:2]
    if len(ranked_models) >= 3:
        ensemble_groups["Best 3"] = ranked_models[:3]
    if len(ranked_models) >= 4:
        ensemble_groups["Full Ensemble"] = ranked_models[:4]

    selected_weights = {}

    for group_name, group_models in ensemble_groups.items():
        common_ids = set.intersection(*[set(val_predictions[m].keys()) for m in group_models])
        common_ids = sorted([fid for fid in common_ids if fid in true_age_dict])

        y_true = np.array([true_age_dict[fid] for fid in common_ids])
        model_preds = {m: np.array([val_predictions[m][fid] for fid in common_ids]) for m in group_models}

        # Grid Search
        grid_step = 0.1
        weight_ranges = [np.arange(0.1, 1.0, grid_step) for _ in group_models]
        best_gs_mae, best_gs_weights = float("inf"), None

        for combo in itertools.product(*weight_ranges):
            if not np.isclose(sum(combo), 1.0, atol=1e-5):
                continue
            weights = dict(zip(group_models, combo))
            ensemble_pred = sum(model_preds[m] * w for m, w in weights.items())
            mae = mean_absolute_error(y_true, ensemble_pred)
            if mae < best_gs_mae:
                best_gs_mae = mae
                best_gs_weights = weights

        # MAE-based (inverse of val MAE)
        inv_maes = {m: 1.0 / model_val_maes[m] for m in group_models}
        total = sum(inv_maes.values())
        mae_weights = {m: inv_maes[m] / total for m in group_models}

        selected_weights[group_name] = {
            "Grid Search": best_gs_weights,
            "MAE-based": mae_weights,
            "models": group_models
        }

        gs_w_str = ", ".join(f"{MODEL_DISPLAY_NAMES.get(m, m)}={best_gs_weights[m]:.2f}" for m in group_models)
        mae_w_str = ", ".join(f"{MODEL_DISPLAY_NAMES.get(m, m)}={mae_weights[m]:.2f}" for m in group_models)
        print(f"\n{group_name} ({', '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in group_models)}):")
        print(f"  Grid Search (Val MAE={best_gs_mae:.3f}): [{gs_w_str}]")
        print(f"  MAE-based: [{mae_w_str}]")

    return selected_weights

def evaluate_ensembles(test_predictions, true_age_dict, selected_weights):
    """Evaluate ensemble configurations on the test set using weights selected on validation."""
    results_summary = []

    print("\n=== Ensemble Evaluation on Test Set ===")
    for group_name, config in selected_weights.items():
        group_models = config["models"]
        common_ids = set.intersection(*[set(test_predictions[m].keys()) for m in group_models])
        common_ids = sorted([fid for fid in common_ids if fid in true_age_dict])

        y_true = np.array([true_age_dict[fid] for fid in common_ids])
        model_preds = {m: np.array([test_predictions[m][fid] for fid in common_ids]) for m in group_models}

        for method in ["Grid Search", "MAE-based"]:
            weights = config[method]
            ensemble_pred = sum(model_preds[m] * w for m, w in weights.items())
            metrics = compute_full_metrics(y_true, ensemble_pred)

            w_dict = {MODEL_DISPLAY_NAMES.get(m, m): round(w, 4) for m, w in weights.items()}
            row = {
                "Ensemble Group": group_name,
                "Method": method,
                "Weights": str(w_dict),
            }
            row.update(metrics)
            results_summary.append(row)
            print(f"  {group_name} ({method}): Test MAE = {metrics['MAE']:.3f}")

    return pd.DataFrame(results_summary)

# --- Colab / output directory handling (matched to experiments 01, 04, 05) ---

def _in_colab():
    """True when running on Google Colab, whose local disk is ephemeral."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def resolve_output_dirs():
    """On Colab, redirect model/result output to mounted Google Drive.

    Everything under /content is wiped on a Colab runtime crash, which would
    destroy the checkpoints mid-run. Persisting to Drive lets a rerun resume
    from the last completed backbone. Local runs keep the repo-relative defaults.
    """
    if not _in_colab():
        return
    drive_root = "/content/drive"
    persist_base = os.path.join(drive_root, "MyDrive", "Age_Estimation", EXPERIMENT_DIRNAME)
    try:
        if not os.path.exists(os.path.join(drive_root, "MyDrive")):
            from google.colab import drive
            print("Colab detected: mounting Google Drive to persist trained models...")
            drive.mount(drive_root)
        if os.path.exists(os.path.join(drive_root, "MyDrive")):
            CONFIG["MODELS_DIR"] = os.path.join(persist_base, "models")
            CONFIG["RESULTS_DIR"] = os.path.join(persist_base, "results")
            print(f"Persisting outputs to Google Drive: {persist_base}")
            return
    except Exception as exc:
        print(f"WARNING: could not mount Google Drive ({exc}).")
    print("WARNING: Drive unavailable; trained models will be LOST if the "
          "Colab runtime crashes.")


# --- Training helpers ---

# Per-epoch training log, mirroring the ViT/CNN experiments' training_log.csv.
LOG_FIELDNAMES = ["model", "phase", "epoch", "train_loss", "val_mae", "lr"]

def _log_epoch(log_path, row, write_header):
    """Append one epoch row to the training-log CSV, flushing immediately.

    Phase 1 ("frozen") starts a fresh file with a header; Phase 2 ("fine_tune")
    appends, so a single CSV spans both phases per model (like exp 04/05).
    """
    if log_path is None:
        return
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES, extrasaction="ignore", restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _train_phase(model, loader, val_loader, optimizer, device, accum_steps, n_epochs, ckpt_path, best_mae,
                 log_path=None, model_name="", phase=""):
    """Run one training phase with ReduceLROnPlateau and EarlyStopping.

    Saves the best model to ``ckpt_path`` when val MAE improves on ``best_mae``.
    Returns the best val MAE achieved (may be unchanged if no improvement).
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10

    for ep in range(1, n_epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, accum_steps)
        metrics = evaluate_patch_mae(model, val_loader, device)
        val_mae = metrics["MAE"]
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mae)

        write_header = log_path is not None and (
            (phase == "frozen" and ep == 1) or not os.path.exists(log_path))
        _log_epoch(log_path,
                   {"model": model_name, "phase": phase, "epoch": ep,
                    "train_loss": loss, "val_mae": val_mae, "lr": current_lr},
                   write_header=write_header)

        print(f"  Ep {ep}: Train Loss {loss:.4f} | Val MAE {val_mae:.3f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"    Val MAE improved to {best_mae:.3f}; saved to {ckpt_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {ep} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    return best_mae


# --- Main Logic ---

def main():
    resolve_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    # 1. Load Data
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return
    df = pd.read_csv(CONFIG["CSV_PATH"])
    true_age_dict = dict(zip(df["File"], df["Age"]))

    # 2. Train models (existing checkpoints in MODELS_DIR are reused per-stage)
    for model_id in MODELS_TO_TRAIN:
        safe_name = model_id.replace("/", "__")
        print(f"\n{'='*40}\nTraining {safe_name}\n{'='*40}")

        model_dir = Path(CONFIG["MODELS_DIR"]) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt = model_dir / f"{safe_name}_best_model.pt"
        done_path = model_dir / f"{safe_name}.done"
        log_path = str(model_dir / f"{safe_name}_training_log.csv")

        # Skip only when training fully completed: the .done marker is written
        # after Phase 2.  A lone checkpoint (no .done) means a crashed run, so we
        # retrain from scratch (same mechanism as the ViT/CNN experiments).
        if done_path.exists() and best_ckpt.exists():
            print(f"Found completed {safe_name}; will load from disk for inference.")
            continue
        if best_ckpt.exists():
            print(f"Found incomplete checkpoint for {safe_name} (no .done marker); retraining from scratch.")

        proc = BeitImageProcessor.from_pretrained(model_id)
        batch_size = get_batch_size(model_id)
        accum_steps = get_accum_steps(model_id)
        print(f"Patch batch size: {batch_size}; gradient accumulation: {accum_steps}; effective patch batch size: {batch_size * accum_steps}")
        loaders = {}
        for split in ["train", "val"]:
            ds_subset = df[df["Set"].str.lower() == split]
            is_train = (split == "train")
            loaders[split] = DataLoader(
                HHDPatchStream(ds_subset, CONFIG["DATA_DIR"], proc, augment=is_train),
                batch_size=batch_size if is_train else CONFIG["EVAL_BATCH_SIZE"],
                num_workers=2,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=collate_patches
            )

        model = DiTReg(name=model_id).to(device)

        # --- Phase 1: Frozen Backbone (Adam @ LR_INIT) ---
        print("Phase 1: Frozen Backbone")
        for p in model.backbone.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(model.head.parameters(), lr=CONFIG["LR_INIT"])
        phase1_best_mae = _train_phase(model, loaders["train"], loaders["val"], optimizer, device,
                     accum_steps, CONFIG["EPOCHS_PHASE1"], best_ckpt, float("inf"),
                     log_path=log_path, model_name=safe_name, phase="frozen")
        print(f"  Phase 1 best Val MAE: {phase1_best_mae:.3f}")

        # --- Phase 2: Fine-Tuning (Adam @ LR_FT) ---
        # Reload Phase 1 best, unfreeze all, and carry the best val MAE forward so the
        # single best-model checkpoint is overwritten only when Phase 2 improves on it.
        print("Phase 2: Fine-Tuning")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        for p in model.backbone.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR_FT"])
        _train_phase(model, loaders["train"], loaders["val"], optimizer, device,
                     accum_steps, CONFIG["EPOCHS_PHASE2"], best_ckpt, phase1_best_mae,
                     log_path=log_path, model_name=safe_name, phase="fine_tune")

        # Mark training as fully complete
        done_path.write_text(f"{safe_name} training complete\n")

        # Free GPU memory before next model
        del model, loaders
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Generate Predictions on Val and Test Sets
    all_predictions = {"val": {}, "test": {}}

    for model_id in MODELS_TO_TRAIN:
        safe_name = model_id.replace("/", "__")
        model_dir = Path(CONFIG["MODELS_DIR"]) / safe_name
        best_ckpt = model_dir / f"{safe_name}_best_model.pt"

        if not best_ckpt.exists():
            print(f"WARNING: No best-model checkpoint for {safe_name}, skipping predictions.")
            continue

        proc = BeitImageProcessor.from_pretrained(model_id)
        model = DiTReg(name=model_id).to(device)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        for split in ["val", "test"]:
            csv_path = os.path.join(CONFIG["RESULTS_DIR"], f"{safe_name}_{split}_preds.csv")

            if os.path.exists(csv_path):
                print(f"Loading existing {split} predictions for {MODEL_DISPLAY_NAMES.get(model_id, model_id)}...")
                pred_df = pd.read_csv(csv_path)
                all_predictions[split][model_id] = dict(zip(pred_df["ImageID"], pred_df["PredAge"]))
            else:
                ds_subset = df[df["Set"].str.lower() == split]
                loader = DataLoader(
                    HHDPatchStream(ds_subset, CONFIG["DATA_DIR"], proc, augment=False),
                    batch_size=CONFIG["EVAL_BATCH_SIZE"],
                    num_workers=2,
                    persistent_workers=True,
                    pin_memory=True,
                    collate_fn=collate_patches
                )
                print(f"Generating {split} predictions for {MODEL_DISPLAY_NAMES.get(model_id, model_id)}...")
                preds = generate_predictions(model, loader, device)
                all_predictions[split][model_id] = preds

                # Save predictions CSV
                rows = [{"ImageID": fid, "PredAge": age} for fid, age in preds.items()]
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  Saved to {csv_path}")

        # Free model after generating predictions (memory management)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not all_predictions["test"]:
        print("No models available for evaluation. Exiting.")
        return

    # 4. Print Individual Model Test Metrics
    print("\n=== Individual Model Test Metrics ===")
    for model_id, preds in all_predictions["test"].items():
        common = [fid for fid in preds if fid in true_age_dict]
        y_true = [true_age_dict[fid] for fid in common]
        y_pred = [preds[fid] for fid in common]
        metrics = compute_full_metrics(y_true, y_pred)
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        print(f"  {name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

    # 5. Select Weights on Validation, Evaluate on Test
    selected_weights = select_ensemble_weights(all_predictions["val"], true_age_dict)
    summary_df = evaluate_ensembles(all_predictions["test"], true_age_dict, selected_weights)

    # 6. Save Summary
    summary_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["SUMMARY_CSV"])
    summary_df.to_csv(summary_path, index=False)

    pd.set_option('display.max_columns', None)
    print("\n=== Final Evaluation Summary ===")
    print(summary_df)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()