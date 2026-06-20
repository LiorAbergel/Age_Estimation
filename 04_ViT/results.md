# Experiment 06 — Vision Transformers (Initial): Results

> **Role in paper:** Development experiment — not reported in the paper.
> Initial ViT exploration; single-split results on the official HHD test set.

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

## Results — Original HHD (Single Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **MobileViT-XXS** | **4.42** | 6.54 | −0.07 | 25.90 | 16.38 | 74.14 | 96.55 | 31.44 | 3.38 |
| ConvNeXtV2-Tiny | 6.30 | 7.58 | −0.44 | 38.78 | 1.72 | 41.38 | 96.55 | 30.36 | 5.64 |
| Ensemble (all 4) | 6.09 | 7.42 | −0.38 | 37.51 | 1.72 | 41.38 | 96.55 | 30.02 | 5.33 |
| TinyViT-11M | 6.70 | 7.79 | −0.52 | 41.92 | 1.72 | 35.34 | 96.55 | 28.70 | 5.87 |
| SwinV2-Tiny | 6.98 | 8.04 | −0.62 | 43.58 | 1.72 | 18.10 | 96.55 | 29.58 | 6.42 |
