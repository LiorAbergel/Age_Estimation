# Experiment 03 — CNN Cross-Validation: Results

> **Role in paper:** Table 2 (top section) — CNN individual models, 5-fold stratified group CV.
> Results reported as mean ± std across 5 folds. Folds are stratified by AgeGroup and grouped
> by WriterNumber (no writer appears in more than one fold).

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
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (5-Fold Stratified Group CV)

Values are **mean ± std** across 5 folds.

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-----------|-----------|--------------|
| **ResNet50** | **5.02 ± 0.34** | 7.56 ± 0.56 | **0.23 ± 0.10** | 25.01 ± 3.78 | 27.04 ± 7.96 | 69.60 ± 5.94 | 31.23 ± 8.18 | 0.07 ± 0.05 | 3.29 ± 0.57 |
| DenseNet121 | 4.74 ± 0.53 | 7.80 ± 0.47 | 0.18 ± 0.13 | 20.66 ± 3.71 | 35.77 ± 14.41 | 75.62 ± 5.65 | 32.05 ± 8.55 | 0.08 ± 0.09 | 2.78 ± 0.92 |
| InceptionV3 | 4.67 ± 0.48 | 7.69 ± 0.64 | 0.20 ± 0.12 | 21.78 ± 5.89 | 40.72 ± 6.25 | 74.27 ± 6.10 | 32.66 ± 8.16 | 0.03 ± 0.04 | 2.63 ± 0.49 |
| InceptionResNetV2 | 4.99 ± 0.48 | 7.58 ± 0.95 | 0.21 ± 0.20 | 23.12 ± 2.53 | 26.66 ± 9.07 | 69.98 ± 4.84 | 30.94 ± 8.91 | 0.05 ± 0.04 | 3.46 ± 0.53 |
| EfficientNetV2M | 5.56 ± 0.61 | 8.43 ± 0.86 | 0.05 ± 0.11 | 26.11 ± 3.29 | 22.91 ± 6.50 | 61.38 ± 11.44 | 34.73 ± 9.33 | 0.09 ± 0.13 | 4.07 ± 0.87 |
