# Experiment 05 — ViT Cross-Validation: Original Results

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
| Batch size | 64 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | ImageNet-1K |
| Framework | TensorFlow / `keras_cv_attention_models` |

---

## Results — Original HHD (5-Fold Stratified Group CV)

Values are **mean ± std** across 5 folds.

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-------------|-----------|--------------|
| **MobileViT-XXS** | **6.00 ± 0.76** | 8.27 ± 0.27 | **0.08 ± 0.05** | 30.63 ± 4.35 | 14.08 ± 5.21 | 54.89 ± 13.95 | — | 31.23 ± 7.30 | 4.76 ± 1.01 |
| TinyViT-11M | 6.63 ± 0.47 | 8.71 ± 0.70 | −0.02 ± 0.13 | 35.61 ± 5.96 | 13.49 ± 5.59 | 39.59 ± 6.23 | — | 31.69 ± 7.04 | 5.80 ± 0.69 |
| SwinV2-Tiny | 6.78 ± 0.28 | 8.85 ± 0.44 | −0.05 ± 0.07 | 36.23 ± 4.42 | 12.49 ± 5.80 | 35.66 ± 1.41 | — | 31.98 ± 6.74 | 6.15 ± 0.41 |
| ConvNeXtV2-Tiny | 6.81 ± 0.38 | 8.91 ± 0.51 | −0.07 ± 0.11 | 36.34 ± 6.18 | 11.08 ± 4.87 | 37.08 ± 3.37 | — | 32.04 ± 6.44 | 6.16 ± 0.56 |

> The paper's Table 2 does not report Acc±10 (±10 yrs); that column is shown as "—".
