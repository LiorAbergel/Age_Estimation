# Experiment 06 — Vision Transformers (Initial): Results

> **Role in paper:** Development experiment — not reported in the paper.
> Initial ViT exploration; results are reported as mean ± std (5-fold CV).

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
| Pretrained weights | ImageNet-1K |
| Framework | TensorFlow / `keras_cv_attention_models` (requires `TF_USE_LEGACY_KERAS=1`) |

---

## Results — Original HHD (5-Fold CV)

Values are **mean ± std** across 5 folds.

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-------------|-----------|--------------|
| **MobileViT-XXS** | **7.96 ± 0.53** | 12.30 ± 1.22 | −0.05 ± 0.09 | 32.27 ± 3.38 | 17.16 ± 2.23 | 44.04 ± 11.21 | 83.89 ± 3.25 | 58.39 ± 8.57 | 5.61 ± 1.04 |
| SwinV2-Tiny | 8.35 ± 0.56 | 12.07 ± 0.87 | −0.01 ± 0.01 | 38.02 ± 4.25 | 16.71 ± 3.21 | 37.15 ± 1.64 | 81.41 ± 6.59 | 56.81 ± 6.12 | 6.79 ± 0.48 |
| TinyViT-11M | 10.51 ± 1.57 | 12.86 ± 1.14 | −0.14 ± 0.07 | 55.34 ± 9.38 | 8.23 ± 4.81 | 23.07 ± 8.17 | 50.02 ± 15.14 | 52.35 ± 5.93 | 10.49 ± 1.81 |
| ConvNeXtV2-Tiny | 12.29 ± 2.81 | 14.26 ± 2.16 | −0.43 ± 0.36 | 66.98 ± 22.46 | 5.71 ± 5.26 | 15.66 ± 11.82 | 40.21 ± 21.49 | 50.06 ± 5.35 | 13.02 ± 3.84 |
