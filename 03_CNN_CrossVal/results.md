# Experiment 03 — SOTA CNN Cross-Validation: Results

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

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-------------|-----------|--------------|
| **ResNet50** | **5.41 ± 0.78** | 8.17 ± 0.58 | **0.10 ± 0.06** | 25.68 ± 3.73 | 23.72 ± 5.17 | 63.40 ± 11.26 | 90.10 ± 4.42 | 32.99 ± 7.14 | 3.72 ± 0.88 |
| DenseNet121 | 5.46 ± 1.06 | 8.16 ± 0.56 | 0.11 ± 0.06 | 26.25 ± 5.47 | 21.49 ± 16.51 | 61.61 ± 16.41 | 91.18 ± 3.84 | 34.14 ± 6.66 | 3.94 ± 1.33 |
| InceptionResNetV2 | 5.69 ± 0.70 | 7.98 ± 0.68 | 0.17 ± 0.11 | 29.08 ± 3.16 | 16.76 ± 3.99 | 58.32 ± 9.19 | 89.32 ± 4.58 | 32.62 ± 5.80 | 4.37 ± 0.59 |
| InceptionV3 | 6.03 ± 0.64 | 8.41 ± 0.64 | 0.05 ± 0.05 | 29.97 ± 3.17 | 16.40 ± 4.30 | 52.83 ± 9.01 | 90.11 ± 4.84 | 32.70 ± 7.60 | 4.80 ± 0.67 |
| EfficientNetV2M | 7.30 ± 0.28 | 9.03 ± 0.36 | −0.07 ± 0.14 | 41.48 ± 5.73 | 11.58 ± 4.78 | 29.47 ± 2.21 | 84.32 ± 7.19 | 31.44 ± 5.01 | 7.05 ± 0.68 |
