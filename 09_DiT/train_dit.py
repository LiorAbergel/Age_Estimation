"""
Experiment 09: Diffusion Transformers (DiT) for Age Estimation

Overview:
This script trains Diffusion Transformer models (microsoft/dit-base, dit-large)
for the age estimation task. It treats the problem as a regression task on
bags of patches extracted from high-resolution handwriting images.

Key Features:
- Architecture: DiT Backbone (via Hugging Face) + Linear Regression Head.
- Data: Dynamic patch extraction with intensity filtering.
- Training: Two-stage (Frozen -> Unfrozen) optimization.

Requirements:
pip install transformers torch torchvision pandas numpy scikit-learn
"""

import os
import gc
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModel, BeitImageProcessor
from PIL import Image
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": 400,
    "STRIDE": 200,
    "STANDARD_SIZE": 800,
    "BATCH_SIZE": 4,  # Adjust based on VRAM (DiT-Base is heavy)
    "EPOCHS_STAGE1": 15,
    "EPOCHS_STAGE2": 30,
    "LR_BASE": 1e-4,
    "THR": 0.0054,
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/dit_experiment",
    "RESULTS_DIR": "./results/dit_predictions"
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
    def __init__(self, df: pd.DataFrame, root: str, processor, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.proc = processor
        self.augment = augment
        self.target_size = (processor.size["height"], processor.size["width"])

        # Augmentation pipeline
        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(self.target_size, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1))
        ])

        # Validation pipeline (Resize only)
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / str(row["Set"]) / row["File"]
        
        if not img_path.exists():
            # Fallback for different folder structures (lowercase/uppercase)
            img_path = self.root / str(row["Set"]).lower() / row["File"]
            if not img_path.exists():
                return None

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_h, new_w = calculate_resized_dimensions(h, w)
            
            # Ensure dimensions are at least patch size
            new_h = max(new_h, CONFIG["PATCH_SIZE"])
            new_w = max(new_w, CONFIG["PATCH_SIZE"])

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_t = transforms.ToTensor()(img)

            # Unfold to patches
            # (Channels, Height, Width) -> (Channels, P_Rows, P_Cols, PatchH, PatchW)
            patches = img_t.unfold(1, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])\
                           .unfold(2, CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
            
            # Reshape to (N_Patches, 3, PatchH, PatchW)
            patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, CONFIG["PATCH_SIZE"], CONFIG["PATCH_SIZE"])

            # Filter empty patches
            patch_means = patches.mean(dim=[1, 2, 3])
            mask = patch_means > CONFIG["THR"]
            valid_patches = patches[mask]

            if valid_patches.size(0) == 0: return None

            # Process patches for DiT (Resize to 224x224 or model native)
            processed_patches = []
            for i in range(valid_patches.size(0)):
                p_pil = transforms.ToPILImage()(valid_patches[i])
                if self.augment:
                    p_trans = self.aug_transforms(p_pil)
                else:
                    p_trans = self.val_transforms(p_pil)
                processed_patches.append(p_trans)

            # Stack and normalize using HuggingFace processor
            final_patches = torch.stack(processed_patches)
            # HF Processor expects numpy or PIL usually, but can handle tensors if configured.
            # Here we manually normalized to [0,1] in transforms, so we might need standard normalization.
            # Using processor for standard ImageNet mean/std
            final_patches = self.proc(images=final_patches, return_tensors="pt", do_rescale=False, do_resize=False)["pixel_values"]

            label = torch.tensor(row["Age"], dtype=torch.float32)
            return {"pixel_values": final_patches, "label": label, "id": row["File"]}

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

def collate_patches(batch: List[Any]) -> Dict[str, Any]:
    """
    Collates a list of image-dictionaries into a single batch of patches.
    Effectively flattens the "Bag of Patches" into a standard batch.
    """
    batch = [b for b in batch if b is not None]
    if not batch: return {}

    all_pixels, all_labels, all_ids = [], [], []
    for item in batch:
        patches = item["pixel_values"]
        N = patches.size(0)
        all_pixels.append(patches)
        all_labels.append(item["label"].repeat(N)) # Replicate label for each patch
        all_ids.extend([item["id"]] * N)

    return {
        "pixel_values": torch.cat(all_pixels, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "ids": all_ids
    }

# --- Model ---

class DiTReg(nn.Module):
    def __init__(self, name: str = "microsoft/dit-base", p: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Sequential(
            nn.Dropout(p), 
            nn.Linear(self.backbone.config.hidden_size, 1)
        )

    def forward(self, pixel_values, labels=None):
        # Forward pass through DiT
        outputs = self.backbone(pixel_values=pixel_values)
        # Use CLS token (index 0)
        cls_token = outputs.last_hidden_state[:, 0]
        pred = self.head(cls_token).squeeze(1)
        
        out = {"preds": pred}
        if labels is not None:
            out["loss"] = nn.functional.l1_loss(pred, labels) # MAE Loss
        return out

# --- Training & Evaluation ---

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    
    # Gradient accumulation for stability if batch size is small
    ACCUM_STEPS = 1 
    
    pbar = tqdm(loader, desc="Training", leave=False)
    opt.zero_grad()
    
    for i, batch in enumerate(pbar):
        if not batch: continue
        
        px = batch["pixel_values"].to(device)
        lbl = batch["labels"].to(device)

        out = model(pixel_values=px, labels=lbl)
        loss = out["loss"] / ACCUM_STEPS
        loss.backward()

        if (i + 1) % ACCUM_STEPS == 0:
            opt.step()
            opt.zero_grad()

        bs = lbl.size(0)
        total_loss += out["loss"].item() * ACCUM_STEPS * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{out['loss'].item()*ACCUM_STEPS:.4f}")

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    img_preds = defaultdict(list)
    img_gts = {}
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if not batch: continue
        px = batch["pixel_values"].to(device)
        ids = batch["ids"]
        labels = batch["labels"].cpu().numpy()

        preds = model(pixel_values=px)["preds"].cpu().numpy()

        for p, l, fid in zip(preds, labels, ids):
            img_preds[fid].append(p)
            img_gts[fid] = l

    # Aggregate patch predictions to image level
    final_preds, final_gts = [], []
    for fid in img_preds:
        final_preds.append(np.mean(img_preds[fid]))
        final_gts.append(img_gts[fid])

    final_gts = np.array(final_gts)
    final_preds = np.array(final_preds)
    
    mae = mean_absolute_error(final_gts, final_preds)
    rmse = np.sqrt(mean_squared_error(final_gts, final_preds))
    
    return {"MAE": mae, "RMSE": rmse}

# --- Main Logic ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    df = pd.read_csv(CONFIG["CSV_PATH"])

    for model_id in MODELS_TO_TRAIN:
        safe_name = model_id.replace("/", "__")
        print(f"\n{'='*40}\nTraining {safe_name}\n{'='*40}")
        
        model_dir = Path(CONFIG["MODELS_DIR"]) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup Data
        proc = BeitImageProcessor.from_pretrained(model_id)
        
        loaders = {}
        for split in ["train", "val", "test"]:
            ds_subset = df[df["Set"].str.lower() == split]
            is_train = (split == "train")
            
            loaders[split] = DataLoader(
                HHDPatchDataset(ds_subset, CONFIG["DATA_DIR"], proc, augment=is_train),
                batch_size=CONFIG["BATCH_SIZE"] if is_train else CONFIG["BATCH_SIZE"] * 2,
                shuffle=is_train,
                num_workers=2, 
                collate_fn=collate_patches
            )

        # 2. Setup Model
        model = DiTReg(name=model_id).to(device)
        
        stage1_ckpt = model_dir / "stage1_best.pt"
        final_ckpt = model_dir / "final_best.pt"
        
        best_mae = float("inf")

        # --- Stage 1: Frozen Backbone ---
        if not stage1_ckpt.exists():
            print("❄️ Stage 1: Frozen Backbone")
            for p in model.backbone.parameters(): p.requires_grad = False
            
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=CONFIG["LR_BASE"])
            
            for ep in range(1, CONFIG["EPOCHS_STAGE1"] + 1):
                loss = train_one_epoch(model, loaders["train"], optimizer, device)
                metrics = evaluate(model, loaders["val"], device)
                print(f"  Ep {ep}: Train Loss {loss:.4f} | Val MAE {metrics['MAE']:.3f}")
                
                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    torch.save(model.state_dict(), stage1_ckpt)
        else:
            print("🔹 Stage 1 already completed. Loading weights.")
            model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
            # Evaluate once to set best_mae
            metrics = evaluate(model, loaders["val"], device)
            best_mae = metrics["MAE"]

        # --- Stage 2: Unfrozen ---
        if not final_ckpt.exists():
            print("🔥 Stage 2: Fine-Tuning")
            for p in model.backbone.parameters(): p.requires_grad = True
            
            # Lower LR for fine-tuning
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_BASE"] / 10)
            
            for ep in range(1, CONFIG["EPOCHS_STAGE2"] + 1):
                loss = train_one_epoch(model, loaders["train"], optimizer, device)
                metrics = evaluate(model, loaders["val"], device)
                print(f"  Ep {ep}: Train Loss {loss:.4f} | Val MAE {metrics['MAE']:.3f}")
                
                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    torch.save(model.state_dict(), final_ckpt)
        
        # --- Final Inference ---
        print("🧪 Generating Test Predictions...")
        if final_ckpt.exists():
            model.load_state_dict(torch.load(final_ckpt, map_location=device))
        
        # Evaluate on Test
        img_preds = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loaders["test"]):
                if not batch: continue
                px = batch["pixel_values"].to(device)
                ids = batch["ids"]
                preds = model(pixel_values=px)["preds"].cpu().numpy()
                for fid, p in zip(ids, preds):
                    img_preds[fid].append(p)
        
        # Save CSV
        rows = []
        for fid, plist in img_preds.items():
            rows.append({"ImageID": fid, "PredAge": float(np.mean(plist))})
            
        pd.DataFrame(rows).to_csv(os.path.join(CONFIG["RESULTS_DIR"], f"{safe_name}_preds.csv"), index=False)
        print(f"✅ Saved predictions to {safe_name}_preds.csv")

if __name__ == "__main__":
    main()