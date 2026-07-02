# Experiment 01 — CNN Ensemble: Original Results

> **Role in paper:** Table 3 (top section) — CNN ensembles on the official HHD split.
>
> **Source:** Values from the original published paper (`paper/main.tex`, Tables `tab:ensembles` and `tab:ensemble_weights_CNN`) and the original official-split individual model results. These reflect the pipeline **before** the misalignment fix.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | Rotation (±10°), zoom (up to 10%), brightness, Gaussian noise |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (Official Split)

### Individual Models

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| ResNet50 | 3.23 | 6.03 | 0.09 | 18.13 | 56.90 | 81.90 | 30.60 | 0.00 | 1.59 |
| InceptionResNetV2 | 3.64 | 6.39 | −0.02 | 20.12 | 43.97 | 87.07 | 32.62 | 0.03 | 2.46 |
| DenseNet121 | 3.97 | 6.10 | 0.07 | 22.80 | 33.62 | 82.76 | 26.85 | 0.14 | 2.76 |
| InceptionV3 | 5.06 | 6.55 | −0.07 | 31.29 | 6.90 | 67.24 | 27.78 | 0.10 | 4.12 |
| EfficientNetV2M | 7.17 | 8.12 | −0.65 | 45.19 | 0.86 | 16.38 | 28.06 | 1.24 | 6.39 |

### Ensemble Configurations

> **Note:** In the original pipeline, ensemble weights were selected on the **test set** (not validation), introducing data leakage. This was corrected in the re-run.

| Ensemble | Weights | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|----------|---------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| **Best 3 (Grid Search)** | ResNet50=0.10, InceptionResNetV2=0.50, DenseNet121=0.40 | **2.85** | 5.87 | 0.14 | 15.30 | 68.97 | 84.48 | 30.11 | 0.03 | 1.26 |
| Best 3 (MAE-based) | ResNet50=0.3510, InceptionResNetV2=0.3321, DenseNet121=0.3169 | 2.86 | 5.86 | 0.14 | 15.48 | 68.10 | 82.76 | 30.08 | 0.01 | 1.21 |
| Best 4 (Grid Search) | ResNet50=0.10, InceptionResNetV2=0.40, DenseNet121=0.40, InceptionV3=0.10 | 2.90 | 5.83 | 0.15 | 15.85 | 68.10 | 81.90 | 29.62 | 0.03 | 1.19 |
| Full Ensemble (Grid Search) | ResNet50=0.10, InceptionResNetV2=0.20, DenseNet121=0.50, InceptionV3=0.10, EfficientNetV2M=0.10 | 3.01 | 5.81 | 0.16 | 16.88 | 67.24 | 81.03 | 28.59 | 0.02 | 1.31 |
| Best 4 (MAE-based) | ResNet50=0.2656, InceptionResNetV2=0.2570, DenseNet121=0.2501, InceptionV3=0.2272 | 3.14 | 5.88 | 0.14 | 17.71 | 55.17 | 81.03 | 29.54 | 0.01 | 1.36 |
| Best 2 (Grid Search) | ResNet50=0.90, InceptionResNetV2=0.10 | 3.24 | 6.04 | 0.09 | 18.17 | 56.90 | 81.90 | 30.80 | 0.00 | 1.58 |
| Best 2 (MAE-based) | ResNet50=0.5298, InceptionResNetV2=0.4701 | 3.35 | 6.14 | 0.06 | 18.61 | 53.45 | 82.76 | 31.55 | 0.02 | 1.69 |
| Full Ensemble (MAE-based) | ResNet50=0.2149, InceptionResNetV2=0.2105, DenseNet121=0.2069, InceptionV3=0.1951, EfficientNetV2M=0.1723 | 3.75 | 6.08 | 0.08 | 21.97 | 41.38 | 80.17 | 29.26 | 0.13 | 2.30 |
