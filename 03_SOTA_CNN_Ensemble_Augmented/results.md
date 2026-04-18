# Experiment 03 — SOTA CNN Ensemble with Augmentation: Results

> **Role in paper:** Table 3 (top section) — CNN ensembles on the official HHD split.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | Rotation (±15°), zoom (up to 10%), brightness/contrast, Gaussian noise |
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
| ResNet50 | 3.23 | 6.03 | 0.09 | 18.13 | 56.90 | 81.90 | 96.55 | 30.60 | 1.59 |
| InceptionResNetV2 | 3.64 | 6.39 | −0.02 | 20.12 | 43.97 | 87.07 | 96.55 | 32.62 | 2.46 |
| DenseNet121 | 3.97 | 6.10 | 0.07 | 22.80 | 33.62 | 82.76 | 93.10 | 26.85 | 2.76 |
| InceptionV3 | 5.06 | 6.55 | −0.07 | 31.29 | 6.90 | 67.24 | 93.97 | 27.78 | 4.12 |
| EfficientNetV2M | 7.17 | 8.12 | −0.65 | 45.19 | 0.86 | 16.38 | 93.10 | 28.06 | 6.39 |

### Ensemble Configurations

| Ensemble | Weights | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|----------|---------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **Best 3 (Grid Search)** | ResNet50=0.10, InceptionResNetV2=0.50, DenseNet121=0.40 | **2.85** | 5.87 | 0.14 | 15.30 | 68.97 | 84.48 | 96.55 | 30.11 | 1.26 |
| Best 3 (MAE-based) | ResNet50=0.35, InceptionResNetV2=0.33, DenseNet121=0.32 | 2.86 | 5.86 | 0.14 | 15.48 | 68.10 | 82.76 | 96.55 | 30.08 | 1.21 |
| Best Weighted (R+D+IRN) | ResNet50=0.25, DenseNet121=0.33, InceptionResNetV2=0.42 | 2.85 | 5.87 | 0.14 | 15.38 | 68.10 | 83.62 | 96.55 | 30.21 | 1.14 |
| Best 2 Weighted | ResNet50=0.70, DenseNet121=0.30 | 2.93 | 5.84 | 0.15 | 16.12 | 66.38 | 81.03 | 96.55 | 29.48 | 1.30 |
| Best 4 (Grid Search) | ResNet50=0.10, InceptionResNetV2=0.40, DenseNet121=0.40, InceptionV3=0.10 | 2.90 | 5.83 | 0.15 | 15.85 | 68.10 | 81.90 | 96.55 | 29.62 | 1.19 |
| Best 4 (MAE-based) | ResNet50=0.27, InceptionResNetV2=0.26, DenseNet121=0.25, InceptionV3=0.23 | 3.14 | 5.88 | 0.14 | 17.71 | 55.17 | 81.03 | 96.55 | 29.54 | 1.36 |
| Full Ensemble (Grid Search) | ResNet50=0.10, InceptionV3=0.10, InceptionResNetV2=0.20, DenseNet121=0.50, EfficientNetV2M=0.10 | 3.01 | 5.81 | 0.16 | 16.88 | 67.24 | 81.03 | 96.55 | 28.59 | 1.31 |
| Full Ensemble (MAE-based) | ResNet50=0.21, InceptionV3=0.20, InceptionResNetV2=0.21, DenseNet121=0.21, EfficientNetV2M=0.17 | 3.75 | 6.08 | 0.08 | 21.97 | 41.38 | 80.17 | 96.55 | 29.26 | 2.30 |
| Weighted (R+I+D, equal) | ResNet50=0.33, InceptionV3=0.33, DenseNet121=0.33 | 3.10 | 5.81 | 0.16 | 17.78 | 57.76 | 81.90 | 96.55 | 28.41 | 1.33 |
| Weighted (ResNet50 + DenseNet121) | ResNet50=0.50, DenseNet121=0.50 | 3.08 | 5.81 | 0.15 | 16.99 | 65.52 | 81.03 | 96.55 | 28.72 | 1.44 |
| Weighted (ResNet50 + InceptionResNetV2) | ResNet50=0.80, InceptionResNetV2=0.20 | 3.26 | 6.06 | 0.08 | 18.24 | 56.90 | 81.90 | 96.55 | 31.01 | 1.55 |
| Full Ensemble (equal weights) | All 5 models, equal weights | 3.88 | 6.13 | 0.06 | 22.89 | 37.93 | 80.17 | 96.55 | 29.18 | 2.46 |
| Best 2 (MAE-based) | ResNet50=0.53, InceptionResNetV2=0.47 | 3.35 | 6.14 | 0.06 | 18.61 | 53.45 | 82.76 | 96.55 | 31.55 | 1.69 |
| Best 2 (Grid Search) | ResNet50=0.90, InceptionResNetV2=0.10 | 3.24 | 6.04 | 0.09 | 18.17 | 56.90 | 81.90 | 96.55 | 30.80 | 1.58 |
