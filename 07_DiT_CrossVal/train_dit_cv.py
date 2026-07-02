"""
Experiment 07: Document Image Transformer (DiT) with 5-Fold Stratified Group CV

Overview:
This script performs a 5-fold cross-validation across multiple DiT backbones.
It ensures patches from the same writer are isolated within folds and applies
intensity filtering to remove background noise.

Models Evaluated:
- DiT-Base
- DiT-Large
- DiT-Base (RVL-CDIP Fine-tuned)
- DiT-Large (RVL-CDIP Fine-tuned)

Key Features:
- Mixed Precision (FP16) training for memory efficiency.
- Gradient Accumulation to maintain effective batch sizes on limited VRAM.
- Dynamic Resizing and Patching logic matching SOTA CNN experiments.

Pipeline (matched to reference experiments 03/05):
- Phase 1: Frozen backbone, Adam(1e-4), up to 50 epochs.
- Phase 2: All layers unfrozen, Adam(1e-5), up to 10 epochs.
- Both phases use ReduceLROnPlateau(factor=0.1, patience=5) and EarlyStopping(patience=10).
- Best checkpoint is carried forward: Phase 2 only overwrites if it beats Phase 1.
- Out-of-fold predictions collected for each fold (matched to Exp 03/05).

Requirements:
pip install transformers torch torchvision pandas numpy scikit-learn
"""

import os
import csv
import random
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, BeitImageProcessor
from PIL import Image
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from download_dataset import ensure_dataset

EXPERIMENT_DIRNAME = "07_DiT_CrossVal"

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": 400,
    "STRIDE": 200,
    "STANDARD_SIZE": 800,
    "LR_INIT": 1e-3,   # frozen-head LR (same as CNN/ViT aligned pipeline)
    "LR_FT": 1e-4,     # fine-tune LR (same as CNN/ViT aligned pipeline)
    "THR": 0.0054,  # Empty patch threshold
    "SEED": 42,
    "N_SPLITS": 5,
    # Patch-level batching: the DataLoader batches *patches* (like the CNN/ViT
    # flat_map -> batch(128) pipeline), so the effective optimizer step is 128 patches.
    "TARGET_EFFECTIVE_BATCH_SIZE": 128,  # effective patches per optimizer step
    "BATCH_SIZE_BASE": 128,              # physical patch batch for DiT-Base (accum 1)
    "BATCH_SIZE_LARGE": 16,              # physical patch batch for DiT-Large (accum 8)
    "EVAL_BATCH_SIZE": 128,              # inference patch batch (no effect on results)
    "EPOCHS_PHASE1": 50,
    "EPOCHS_PHASE2": 10,
    "DATA_DIR": os.path.join(REPO_ROOT, "data"),
    "CSV_PATH": os.path.join(REPO_ROOT, "data", "NewAgeSplit.csv"),
    "MODELS_DIR": os.path.join(REPO_ROOT, "models", EXPERIMENT_DIRNAME),
    "RESULTS_DIR": os.path.join(REPO_ROOT, "results", EXPERIMENT_DIRNAME),
}

# Full Model List
MODEL_IDS = [
    "microsoft/dit-base",
    "microsoft/dit-large",
    "microsoft/dit-base-finetuned-rvlcdip",
    "microsoft/dit-large-finetuned-rvlcdip",
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["SEED"])

def seed_worker(worker_id):
    """Seed each DataLoader worker's RNGs so augmentation is reproducible across runs."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Dedicated generator so worker base seeds are stable regardless of prior RNG use
# (e.g. model initialization) in the main process.
DATALOADER_GENERATOR = torch.Generator()
DATALOADER_GENERATOR.manual_seed(CONFIG["SEED"])

def get_batch_size(model_id):
    return CONFIG["BATCH_SIZE_LARGE"] if "large" in model_id else CONFIG["BATCH_SIZE_BASE"]

def get_accum_steps(model_id):
    batch_size = get_batch_size(model_id)
    target = CONFIG["TARGET_EFFECTIVE_BATCH_SIZE"]
    if target % batch_size != 0:
        raise ValueError(f"TARGET_EFFECTIVE_BATCH_SIZE={target} must be divisible by batch_size={batch_size}")
    return target // batch_size

# --- Image Processing ---

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
    def __init__(self, df, root, processor, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment
        target_size = (processor.size["height"], processor.size["width"])
        # Replaces the per-image processor() call; identical ImageNet/BEiT normalization.
        normalize = transforms.Normalize(processor.image_mean, processor.image_std)

        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(target_size, scale=(0.9, 1.1), ratio=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.25),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1)),
            normalize,
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalize,
        ])

    def _valid_patches(self, row):
        """Return this image's non-empty 400x400 patches as (N,3,H,W), or None."""
        img_path = self.root / str(row["Set"]).lower() / row["File"]
        if not img_path.exists():
            return None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        w, h = img.size
        new_h, new_w = calculate_resized_dimensions(h, w)
        if new_h < CONFIG["PATCH_SIZE"] or new_w < CONFIG["PATCH_SIZE"]:
            new_h = max(new_h, CONFIG["PATCH_SIZE"])
            new_w = max(new_w, CONFIG["PATCH_SIZE"])
        img_t = transforms.ToTensor()(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

        # Unfold to (N_Patches, 3, PatchH, PatchW)
        patches_h = img_t.unfold(1, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
        patches = patches_h.unfold(2, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, CONFIG["PATCH_SIZE"], CONFIG["PATCH_SIZE"])

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
                yield {"pixel_values": tfm(transforms.ToPILImage()(valid[i])),
                       "label": label, "id": fid}


def collate_patches(batch):
    """Stack individual patch samples into one fixed-size patch batch."""
    batch = [b for b in batch if b]
    if not batch:
        return {}
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "file_ids": [b["id"] for b in batch],
    }

# --- Model ---



class DiTReg(nn.Module):
    def __init__(self, name="microsoft/dit-base", p=0.5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(self.backbone.config.hidden_size, 1))

    def forward(self, pixel_values, labels=None):
        out = self.backbone(pixel_values=pixel_values)
        # Global average pooling over patch tokens (exclude CLS at index 0),
        # matching the GlobalAveragePooling head used by the CNN/ViT experiments
        # and the BEiT-recommended mean-pooling (its CLS token is not pretrained).
        pooled = out.last_hidden_state[:, 1:].mean(dim=1)
        preds = self.head(pooled).squeeze(1)
        if labels is None: return {"preds": preds}
        return {"loss": nn.functional.mse_loss(preds, labels), "preds": preds}

# --- Training and Evaluation Functions ---

def train_one_epoch(model, loader, opt, device, scaler, accum_steps):
    model.train()
    total_loss, count = 0.0, 0
    pending_steps = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    opt.zero_grad()
    for batch in pbar:
        if not batch: continue
        px, lbl = batch["pixel_values"].to(device), batch["labels"].to(device)
        with autocast('cuda'):
            out = model(pixel_values=px, labels=lbl)
            loss = out["loss"] / accum_steps
        scaler.scale(loss).backward()
        pending_steps += 1
        if pending_steps == accum_steps:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            pending_steps = 0
        total_loss += out["loss"].item() * lbl.size(0)
        count += lbl.size(0)
    if pending_steps > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    return total_loss / count if count > 0 else 0.0

@torch.no_grad()
def _aggregate_preds(model, loader, device):
    """Run inference and average patch predictions to image level."""
    model.eval()
    img_preds, img_gts = defaultdict(list), {}
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if not batch: continue
        px, lbl, ids = batch["pixel_values"].to(device), batch["labels"].cpu().numpy(), batch["file_ids"]
        preds = model(pixel_values=px)["preds"].cpu().numpy()
        for p, l, fid in zip(preds, lbl, ids):
            img_preds[fid].append(p)
            img_gts[fid] = l

    y_true = np.array([img_gts[f] for f in img_preds])
    y_pred = np.array([np.mean(img_preds[f]) for f in img_preds])
    return y_true, y_pred

@torch.no_grad()
def evaluate_patch_mae(model, loader, device):
    """Patch-level validation MAE, used for checkpoint selection / early stopping.

    Matches the CNN/ViT experiments, where the best model is chosen on the
    patch-level validation metric. Page-level metrics (the reported numbers) are
    computed separately from the aggregated OOF predictions.
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

def compute_full_metrics(y_true, y_pred):
    """Extended image-level metrics (keys aligned with Exp 03/05)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        if np.isnan(mape): mape = 0.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape,
        "Acc_2yr": float(np.mean(errors <= 2) * 100),
        "Acc_5yr": float(np.mean(errors <= 5) * 100),
        "Acc_10yr": float(np.mean(errors <= 10) * 100),
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }

# --- Colab / output directory handling (matched to experiments 03, 05) ---

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
    destroy the fold checkpoints mid-run. Persisting to Drive lets a rerun
    resume from the last completed fold. Local runs keep the repo-relative defaults.
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
    appends, so a single CSV spans both phases per fold (like exp 04/05).
    """
    if log_path is None:
        return
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES, extrasaction="ignore", restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _train_phase(model, loader, val_loader, optimizer, device, scaler, accum_steps,
                 n_epochs, ckpt_path, best_mae, log_path=None, model_name="", phase=""):
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
        loss = train_one_epoch(model, loader, optimizer, device, scaler, accum_steps)
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


# --- Results Persistence (matched to experiment 03/05) ---

def save_cv_results(all_fold_metrics, all_oof_records):
    """Persist OOF predictions, per-fold metrics, and the mean +/- std CV summary.

    ``oof_predictions.csv`` is the input consumed by reproduce_results.py's fast
    path; ``cv_metrics_summary.csv`` reproduces the paper's individual-model CV
    table (mean +/- std across folds).
    """
    out_dir = CONFIG["RESULTS_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    oof_path = os.path.join(out_dir, "oof_predictions.csv")
    pd.DataFrame(all_oof_records).to_csv(oof_path, index=False)
    print(f"\nSaved OOF predictions to {oof_path} ({len(all_oof_records)} rows)")

    per_fold_rows = []
    for model_name, fms in all_fold_metrics.items():
        for i, fm in enumerate(fms, start=1):
            row = {"Model": model_name, "Fold": i}
            row.update(fm)
            per_fold_rows.append(row)
    per_fold_path = os.path.join(out_dir, "cv_metrics_per_fold.csv")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)
    print(f"Saved per-fold metrics to {per_fold_path}")

    metric_keys = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr", "Acc_10yr",
                   "Max_Error", "Median_Error", "Min_Error"]
    summary_rows = []
    print("\n────── CV SUMMARY (mean ± std across folds) ──────")
    for model_name, fms in all_fold_metrics.items():
        if not fms:
            continue
        row = {"Model": model_name}
        print(f"\n{model_name}:")
        for k in metric_keys:
            arr = np.array([fm[k] for fm in fms], dtype=float)
            row[f"{k}_mean"], row[f"{k}_std"] = arr.mean(), arr.std()
            print(f"  {k:<13}: {arr.mean():.2f} ± {arr.std():.2f}")
        summary_rows.append(row)
    summary_path = os.path.join(out_dir, "cv_metrics_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved CV summary to {summary_path}")


# --- Main CV Routine ---

def run_full_cv():
    resolve_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    df = pd.read_csv(CONFIG["CSV_PATH"])
    sgkf = StratifiedGroupKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])
    splits = list(sgkf.split(df.index, df["AgeGroup"], df["WriterNumber"]))

    all_fold_metrics = {}
    all_oof_records = []

    for model_id in MODEL_IDS:
        safe_name = model_id.replace("/", "__")
        print(f"\n{'='*40}\nProcessing Model: {model_id}\n{'='*40}")

        proc = BeitImageProcessor.from_pretrained(model_id)
        batch_size = get_batch_size(model_id)
        accum_steps = get_accum_steps(model_id)
        print(f"Patch batch size: {batch_size}; gradient accumulation: {accum_steps}; effective patch batch size: {batch_size * accum_steps}")
        model_root = Path(CONFIG["MODELS_DIR"]) / safe_name
        model_root.mkdir(parents=True, exist_ok=True)

        fold_metrics_list = []

        for fold, (tr_idx, val_idx) in enumerate(splits, 1):
            fold_dir = model_root / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            log_name = f"{safe_name}_fold{fold:02d}"
            best_ckpt = fold_dir / f"{log_name}_best_model.pt"
            done_path = fold_dir / f"{log_name}.done"
            log_path = str(fold_dir / f"{log_name}_training_log.csv")

            print(f"\n── {safe_name} | Fold {fold}/{CONFIG['N_SPLITS']} ──")
            train_df, val_df = df.iloc[tr_idx], df.iloc[val_idx]

            train_loader = DataLoader(HHDPatchStream(train_df, CONFIG["DATA_DIR"], proc, True),
                                      batch_size=batch_size, num_workers=2, persistent_workers=True,
                                      pin_memory=True, worker_init_fn=seed_worker,
                                      generator=DATALOADER_GENERATOR, collate_fn=collate_patches)
            val_loader = DataLoader(HHDPatchStream(val_df, CONFIG["DATA_DIR"], proc, False),
                                    batch_size=CONFIG["EVAL_BATCH_SIZE"], num_workers=2, persistent_workers=True,
                                    pin_memory=True, worker_init_fn=seed_worker,
                                    generator=DATALOADER_GENERATOR, collate_fn=collate_patches)

            # Skip training only when fully completed: the .done marker is written
            # after Phase 2.  A lone checkpoint (no .done) means a crashed run, so we
            # retrain from scratch (same mechanism as the ViT/CNN experiments).
            if done_path.exists() and best_ckpt.exists():
                print(f"Found completed {safe_name} Fold {fold}; loading for OOF evaluation.")
            else:
                if best_ckpt.exists():
                    print(f"Found incomplete checkpoint for {safe_name} Fold {fold} (no .done marker); retraining from scratch.")
                model = DiTReg(name=model_id).to(device)
                scaler = GradScaler('cuda')

                # --- Phase 1: Frozen Backbone (Adam @ LR_INIT) ---
                print("Phase 1: Frozen Backbone")
                for p in model.backbone.parameters():
                    p.requires_grad = False
                opt = torch.optim.Adam(model.head.parameters(), lr=CONFIG["LR_INIT"])
                phase1_best_mae = _train_phase(model, train_loader, val_loader, opt, device, scaler,
                             accum_steps, CONFIG["EPOCHS_PHASE1"], best_ckpt, float("inf"),
                             log_path=log_path, model_name=log_name, phase="frozen")
                print(f"  Phase 1 best Val MAE: {phase1_best_mae:.3f}")

                # --- Phase 2: Fine-Tuning (Adam @ LR_FT) ---
                # Reload Phase 1 best, unfreeze all, and carry the best val MAE forward so the
                # single best-model checkpoint is overwritten only when Phase 2 improves on it.
                print("Phase 2: Fine-Tuning")
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                for p in model.backbone.parameters():
                    p.requires_grad = True
                opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR_FT"])
                _train_phase(model, train_loader, val_loader, opt, device, scaler,
                             accum_steps, CONFIG["EPOCHS_PHASE2"], best_ckpt, phase1_best_mae,
                             log_path=log_path, model_name=log_name, phase="fine_tune")

                # Mark training as fully complete
                done_path.write_text(f"{log_name} training complete\n")

                del model
                torch.cuda.empty_cache()
                gc.collect()

            # --- OOF evaluation (matched to Exp 03/05) ---
            model = DiTReg(name=model_id).to(device)
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

            y_true, y_pred = _aggregate_preds(model, val_loader, device)
            fold_metrics = compute_full_metrics(y_true, y_pred)
            fold_metrics_list.append(fold_metrics)
            print(f"Fold {fold}: MAE={fold_metrics['MAE']:.2f}, RMSE={fold_metrics['RMSE']:.2f}, R2={fold_metrics['R2']:.2f}")

            # Collect per-image OOF records
            model.eval()
            img_preds_oof = defaultdict(list)
            img_gts_oof = {}
            for batch in val_loader:
                if not batch:
                    continue
                px = batch["pixel_values"].to(device)
                ids = batch["file_ids"]
                labels = batch["labels"].cpu().numpy()
                with torch.no_grad():
                    preds = model(pixel_values=px)["preds"].cpu().numpy()
                for p, l, fid in zip(preds, labels, ids):
                    img_preds_oof[fid].append(p)
                    img_gts_oof[fid] = l

            true_map = dict(zip(val_df["File"], val_df["Age"]))
            for iid, plist in img_preds_oof.items():
                if iid in true_map:
                    mean_pred = float(np.mean(plist))
                    all_oof_records.append({
                        "Model": safe_name, "Fold": fold, "ImageID": iid,
                        "Prediction": mean_pred, "TrueAge": float(true_map[iid])
                    })

            # Cleanup
            del model, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()

        all_fold_metrics[safe_name] = fold_metrics_list

    # Save all results
    save_cv_results(all_fold_metrics, all_oof_records)


if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    run_full_cv()