# Experiment 05 — ViT Cross-Validation: Results

> **Role in paper:** Table 2 (middle section) — ViT individual models, 5-fold stratified group CV.
> Results reported as mean ± std across 5 folds. Folds are stratified by AgeGroup and grouped
> by WriterNumber (no writer appears in more than one fold).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 256×256 (SwinV2, MobileViT) or 224×224 (ConvNeXtV2, TinyViT) — after patch extraction |
| Label scaling | No |
| Augmentation | Rotation (±15°), zoom (up to 10%), brightness/contrast, Gaussian noise |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | ImageNet-1K |
| Framework | TensorFlow / `keras_cv_attention_models` (requires `TF_USE_LEGACY_KERAS=1`) |

---

## Results — Original HHD (5-Fold Stratified Group CV)

Values are **mean ± std** across 5 folds.

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-----------|-----------|--------------|
| **MobileViT-XXS** | **4.69 ± 0.22** | 7.82 ± 0.58 | **0.18 ± 0.09** | 21.00 ± 2.05 | 40.75 ± 5.96 | 72.23 ± 4.23 | 34.16 ± 6.69 | 0.02 ± 0.02 | 2.80 ± 0.72 |
| ConvNeXtV2-Tiny | 4.76 ± 0.72 | 7.51 ± 0.44 | 0.24 ± 0.10 | 22.85 ± 5.84 | 37.35 ± 14.26 | 69.08 ± 12.42 | 30.43 ± 6.72 | 0.04 ± 0.07 | 2.98 ± 1.16 |
| TinyViT-11M | 5.60 ± 0.71 | 7.85 ± 0.76 | 0.17 ± 0.09 | 27.39 ± 2.49 | 16.25 ± 7.27 | 59.87 ± 10.28 | 31.06 ± 6.94 | 0.11 ± 0.16 | 4.33 ± 0.76 |
| SwinV2-Tiny | 6.55 ± 0.56 | 8.71 ± 0.46 | −0.02 ± 0.02 | 33.40 ± 3.05 | 10.90 ± 4.29 | 44.36 ± 10.78 | 32.74 ± 7.74 | 0.32 ± 0.30 | 5.88 ± 1.09 |
