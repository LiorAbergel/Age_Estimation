# Experiment 02 — SOTA CNN Ensemble (Initial Transfer Learning): Results

> **Role in paper:** Development experiment — not reported in the paper.
> First transfer-learning baseline with patch extraction and minimal augmentation.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | `random_flip_left_right` only |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| ResNet50 | 3.15 | 5.87 | 0.14 | 17.68 | 63.79 | 79.31 | 96.55 | 29.82 | 1.48 |
| Smaller Ensemble (ResNet50 + InceptionV3) | 3.38 | 5.98 | 0.11 | 19.44 | 59.48 | 81.03 | 95.69 | 28.50 | 1.71 |
| InceptionV3 | 3.71 | 6.23 | 0.03 | 21.78 | 46.55 | 80.17 | 91.38 | 27.18 | 2.07 |
| Full Ensemble (all 5 models) | 4.35 | 6.37 | −0.01 | 26.17 | 34.48 | 78.45 | 95.69 | 28.69 | 3.23 |
| DenseNet121 | 4.58 | 6.59 | −0.08 | 27.18 | 14.66 | 72.41 | 96.55 | 30.21 | 3.31 |
| InceptionResNetV2 | 4.97 | 6.60 | −0.09 | 30.62 | 12.07 | 68.10 | 94.83 | 27.10 | 3.64 |
| EfficientNetV2M | 6.92 | 8.02 | −0.61 | 43.22 | 1.72 | 25.00 | 96.55 | 29.15 | 5.96 |
