# Experiment 04 — ViT Ensemble: Results

> **Role in paper:** Development experiment — not reported in the paper.
> ViT ensemble exploration; single-split results on the official HHD test set.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 256×256 (SwinV2, MobileViT) or 224×224 (ConvNeXtV2, TinyViT) — after patch extraction |
| Patch extraction | 400×400 patches, stride 200 (from 800 px-height images) |
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

## Results — Original HHD (Official Split)

### Individual Models

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| **MobileViT-XXS** | **2.79** | 6.13 | 0.06 | 13.59 | 65.52 | 93.97 | 34.91 | 0.05 | 1.35 |
| ConvNeXtV2-Tiny | 3.86 | 6.08 | 0.08 | 21.87 | 37.07 | 81.03 | 27.99 | 0.13 | 2.98 |
| TinyViT-11M | 4.69 | 6.23 | 0.03 | 28.95 | 18.97 | 70.69 | 25.95 | 0.06 | 3.68 |
| SwinV2-Tiny | 5.46 | 7.46 | −0.39 | 30.87 | 11.21 | 55.17 | 32.41 | 0.16 | 4.83 |

### Ensemble Configurations

> **Note:** Ensemble weights are selected on the **validation set** (grid search or MAE-based formula) and evaluated on the **test set** to avoid data leakage.

| Ensemble | Weights | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|----------|---------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| **Best 3 (Grid Search)** | MobileViT=0.70, ConvNeXtV2=0.10, TinyViT=0.20 | **2.77** | 5.84 | 0.15 | 14.41 | 66.38 | 87.93 | 32.42 | 0.03 | 1.26 |
| Best 2 (Grid Search) | MobileViT=0.90, ConvNeXtV2=0.10 | 2.81 | 6.06 | 0.08 | 13.87 | 65.52 | 93.10 | 34.21 | 0.03 | 1.44 |
| Full Ensemble (Grid Search) | MobileViT=0.10, ConvNeXtV2=0.10, TinyViT=0.50, SwinV2=0.30 | 2.89 | 5.57 | 0.22 | 15.75 | 64.66 | 82.76 | 28.99 | 0.06 | 1.40 |
| Best 3 (MAE-based) | MobileViT=0.37, ConvNeXtV2=0.33, TinyViT=0.30 | 2.89 | 5.68 | 0.19 | 15.78 | 63.79 | 81.90 | 29.89 | 0.01 | 1.24 |
| Full Ensemble (MAE-based) | MobileViT=0.27, ConvNeXtV2=0.26, TinyViT=0.24, SwinV2=0.23 | 3.03 | 5.75 | 0.17 | 16.09 | 53.45 | 85.34 | 30.38 | 0.00 | 1.68 |
| Best 2 (MAE-based) | MobileViT=0.56, ConvNeXtV2=0.44 | 3.04 | 5.93 | 0.12 | 15.80 | 55.17 | 86.21 | 31.84 | 0.01 | 1.77 |
