# Experiment 03 — SOTA CNN Ensemble with Augmentation: Results

> **Role in paper:** Table 3 (top section) — CNN ensembles on the official HHD split.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | Rotation (±15°), zoom (up to 10%), brightness, Gaussian noise |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (Official Split)

### Individual Models

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| ResNet50 | 3.12 | 6.38 | −0.02 | 16.14 | 50.86 | 93.97 | 95.69 | 34.55 | 1.89 |
| InceptionResNetV2 | 4.00 | 6.50 | −0.06 | 22.59 | 26.72 | 81.90 | 96.55 | 32.71 | 2.69 |
| DenseNet121 | 3.80 | 6.45 | −0.04 | 20.70 | 38.79 | 86.21 | 95.69 | 29.61 | 2.82 |
| InceptionV3 | 3.31 | 6.20 | 0.04 | 18.05 | 48.28 | 87.07 | 95.69 | 33.98 | 2.04 |
| EfficientNetV2M | 2.77 | 6.28 | 0.01 | 13.45 | 66.38 | 93.97 | 96.55 | 34.36 | 1.25 |

### Ensemble Configurations

> **Note:** Ensemble weights are selected on the **validation set** (grid search or MAE-based formula) and evaluated on the **test set** to avoid data leakage.

| Ensemble | Weights | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|----------|---------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **Full Ensemble (Grid Search)** | ResNet50=0.10, InceptionV3=0.10, InceptionResNetV2=0.30, DenseNet121=0.30, EfficientNetV2M=0.20 | **2.73** | 6.09 | 0.07 | 13.93 | 72.41 | 88.79 | 96.55 | 32.42 | 1.22 |
| Best 4 (Grid Search) | ResNet50=0.10, InceptionResNetV2=0.40, DenseNet121=0.40, InceptionV3=0.10 | 2.73 | 6.05 | 0.08 | 14.13 | 71.55 | 87.07 | 96.55 | 31.78 | 1.20 |
| Best 3 (Grid Search) | ResNet50=0.10, InceptionResNetV2=0.40, DenseNet121=0.50 | 2.75 | 6.05 | 0.08 | 14.12 | 71.55 | 87.07 | 96.55 | 31.35 | 1.15 |
| Best 3 (MAE-based) | ResNet50=0.35, InceptionResNetV2=0.32, DenseNet121=0.33 | 2.76 | 6.11 | 0.07 | 14.16 | 70.69 | 88.79 | 96.55 | 32.34 | 1.11 |
| Full Ensemble (MAE-based) | ResNet50=0.20, InceptionV3=0.20, InceptionResNetV2=0.19, DenseNet121=0.20, EfficientNetV2M=0.21 | 2.76 | 6.13 | 0.06 | 14.12 | 67.24 | 89.66 | 96.55 | 33.08 | 1.12 |
| Best 4 (MAE-based) | ResNet50=0.26, InceptionResNetV2=0.24, DenseNet121=0.25, InceptionV3=0.25 | 2.82 | 6.10 | 0.07 | 14.66 | 59.48 | 87.07 | 96.55 | 32.75 | 1.13 |
| Best 2 (Grid Search) | ResNet50=0.90, InceptionResNetV2=0.10 | 3.18 | 6.38 | −0.02 | 16.60 | 50.86 | 93.10 | 95.69 | 34.36 | 1.91 |
| Best 2 (MAE-based) | ResNet50=0.55, InceptionResNetV2=0.45 | 3.45 | 6.39 | −0.02 | 18.66 | 44.83 | 89.66 | 96.55 | 33.71 | 2.15 |
