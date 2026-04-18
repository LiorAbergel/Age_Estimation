"""
Experiment 09: Document Image Transformer (DiT) with 5-Fold Stratified Group CV

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
"""

import os
import time
import shutil
import random
import json
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, BeitImageProcessor
from PIL import Image
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": 400,
    "STRIDE": 200,
    "STANDARD_SIZE": 800,
    "LR_BASE": 1e-4,
    "THR": 0.0054,  # Empty patch threshold
    "SEED": 42,
    "N_SPLITS": 5,
    "EPOCHS_STAGE1": 15,
    "EPOCHS_STAGE2": 30,
    "DATA_DIR": "./data",
    "CSV_PATH": os.path.join("data", "NewAgeSplit.csv"),
    "MODELS_DIR": "./models/dit_cv",
    "RESULTS_DIR": "./results/dit_cv"
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

class HHDPatchDataset(Dataset):
    def __init__(self, df, root, processor, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.proc = processor
        self.augment = augment
        self.target_size = (processor.size["height"], processor.size["width"])

        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(self.target_size, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1))
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / str(row["Set"]).lower() / row["File"]
        if not img_path.exists(): return None

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_h, new_w = calculate_resized_dimensions(h, w)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_t = transforms.ToTensor()(img)

            # Unfold to patches
            patches_h = img_t.unfold(1, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
            patches = patches_h.unfold(2, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
            patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, CONFIG["PATCH_SIZE"], CONFIG["PATCH_SIZE"])

            # Filter background patches
            mask = patches.mean(dim=[1, 2, 3]) > CONFIG["THR"]
            valid_patches = patches[mask]
            if valid_patches.size(0) == 0: return None

            processed = []
            for i in range(valid_patches.size(0)):
                p_pil = transforms.ToPILImage()(valid_patches[i])
                processed.append(self.aug_transforms(p_pil) if self.augment else self.val_transforms(p_pil))

            final_patches = torch.stack(processed)
            final_patches = self.proc(images=final_patches, return_tensors="pt", do_rescale=False, do_resize=False)["pixel_values"]
            
            return {"pixel_values": final_patches, "label": torch.tensor(row["Age"], dtype=torch.float32), "id": row["File"]}
        except: return None

def collate_patches(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return {}
    return {
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.cat([item["label"].repeat(item["pixel_values"].size(0)) for item in batch], dim=0),
        "file_ids": [fid for item in batch for fid in [item["id"]] * item["pixel_values"].size(0)]
    }

# --- Model ---



class DiTReg(nn.Module):
    def __init__(self, name="microsoft/dit-base", p=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(self.backbone.config.hidden_size, 1))

    def forward(self, pixel_values, labels=None):
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0]
        preds = self.head(cls_token).squeeze(1)
        if labels is None: return {"preds": preds}
        return {"loss": nn.functional.l1_loss(preds, labels), "preds": preds}

# --- Training and Evaluation Functions ---

def train_one_epoch(model, loader, opt, device, scaler, accum_steps):
    model.train()
    total_loss, count = 0.0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    opt.zero_grad()
    for step, batch in enumerate(pbar):
        if not batch: continue
        px, lbl = batch["pixel_values"].to(device), batch["labels"].to(device)
        with autocast():
            out = model(pixel_values=px, labels=lbl)
            loss = out["loss"] / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        total_loss += loss.item() * accum_steps * lbl.size(0)
        count += lbl.size(0)
    return total_loss / count if count > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
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
    return {"MAE": mean_absolute_error(y_true, y_pred)}

# --- Main CV Routine ---

def run_full_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    df = pd.read_csv(CONFIG["CSV_PATH"])
    sgkf = StratifiedGroupKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])

    for model_id in MODEL_IDS:
        safe_name = model_id.replace("/", "__")
        print(f"\n🚀 Processing Model: {model_id}")
        
        # Memory-safe batch configuration
        if "large" in model_id:
            batch_size = 1
            accum_steps = 8
        else:
            batch_size = 2
            accum_steps = 4

        proc = BeitImageProcessor.from_pretrained(model_id)
        model_root = Path(CONFIG["MODELS_DIR"]) / safe_name
        model_root.mkdir(parents=True, exist_ok=True)

        for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df.index, df["AgeGroup"], df["WriterNumber"]), 1):
            fold_dir = model_root / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            final_ckpt = fold_dir / "final_best.pt"
            if final_ckpt.exists():
                print(f"✅ {safe_name} Fold {fold} already complete. Skipping.")
                continue

            print(f"\n--- {safe_name} | Fold {fold} ---")
            train_df, val_df = df.iloc[tr_idx], df.iloc[val_idx]
            
            train_loader = DataLoader(HHDPatchDataset(train_df, CONFIG["DATA_DIR"], proc, True), 
                                      batch_size=batch_size, shuffle=True, collate_fn=collate_patches)
            val_loader = DataLoader(HHDPatchDataset(val_df, CONFIG["DATA_DIR"], proc, False), 
                                    batch_size=batch_size, shuffle=False, collate_fn=collate_patches)
            
            model = DiTReg(name=model_id).to(device)
            scaler = GradScaler()
            best_mae = float("inf")
            
            # Stage 1: Frozen Backbone
            stage1_ckpt = fold_dir / "stage1_best.pt"
            if not stage1_ckpt.exists():
                print("❄️ Stage 1: Frozen Backbone")
                for p in model.backbone.parameters(): p.requires_grad = False
                opt = torch.optim.AdamW(model.head.parameters(), lr=CONFIG["LR_BASE"])
                for ep in range(1, CONFIG["EPOCHS_STAGE1"] + 1):
                    loss = train_one_epoch(model, train_loader, opt, device, scaler, accum_steps)
                    m = evaluate(model, val_loader, device)
                    if m["MAE"] < best_mae:
                        best_mae = m["MAE"]
                        torch.save(model.state_dict(), stage1_ckpt)
                    print(f"  Ep {ep}: Loss {loss:.4f} | Val MAE {m['MAE']:.3f}")

            # Stage 2: Full Fine-tuning
            print("🔥 Stage 2: Unfrozen Backbone")
            model.load_state_dict(torch.load(stage1_ckpt))
            for p in model.backbone.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_BASE"]/10)
            
            # Reset best_mae for stage 2
            best_mae = evaluate(model, val_loader, device)["MAE"]
            
            for ep in range(1, CONFIG["EPOCHS_STAGE2"] + 1):
                loss = train_one_epoch(model, train_loader, opt, device, scaler, accum_steps)
                m = evaluate(model, val_loader, device)
                if m["MAE"] < best_mae:
                    best_mae = m["MAE"]
                    torch.save(model.state_dict(), final_ckpt)
                print(f"  Ep {ep}: Loss {loss:.4f} | Val MAE {m['MAE']:.3f}")

            # Cleanup
            del model, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    run_full_cv()