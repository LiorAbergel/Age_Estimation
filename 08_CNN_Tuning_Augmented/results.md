# Experiment 08 — CNN Hyperparameter Tuning: Results

> **Role in paper:** Development experiment — not reported in the paper.
> Grid search over unfreeze ratio, dropout, and fine-tune learning rate.

---

## Training Configuration

| Parameter | Search Space / Value |
|-----------|----------------------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | Rotation (±15°), zoom (up to 10%), brightness/contrast, Gaussian noise |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 (fixed) |
| Fine-tune epochs | 10 |
| **Unfreeze ratio** | {0, 0.25, 0.50, 0.75, 1.0} (fraction of backbone unfrozen) |
| **Dropout** | {0.2, 0.3, 0.4, 0.5, 0.6} |
| **Fine-tune LR** | {1e-3, 5e-4, 1e-4} |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (Official Split)

Best hyperparameters found per model (single evaluation on test set).

| Model | Unfreeze Ratio | Dropout | Fine-tune LR | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|:--------------:|:-------:|:------------:|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **InceptionV3** | 0.75 | 0.2 | 1e-4 | **2.66** | 6.09 | 0.07 | 12.34 | 70.69 | 93.10 | 95.69 | 34.71 | 1.35 |
| EfficientNetV2M | 0.25 | 0.3 | 1e-4 | 3.03 | 6.31 | 0.01 | 15.56 | 56.03 | 93.97 | 95.69 | 34.29 | 1.57 |
| ResNet50 | 1.0 | 0.6 | 1e-3 | 2.80 | 6.34 | −0.01 | 13.20 | 62.93 | 93.97 | 94.83 | 34.96 | 1.40 |
| Full Ensemble (all 5) | — | — | — | 3.08 | 6.15 | 0.05 | 16.37 | 55.17 | 91.38 | 96.55 | 32.98 | 1.67 |
| DenseNet121 | 0.25 | 0.2 | 5e-4 | 3.71 | 6.38 | −0.02 | 20.72 | 41.38 | 82.76 | 96.55 | 31.68 | 2.32 |
| InceptionResNetV2 | 0.50 | 0.5 | 1e-3 | 6.08 | 7.37 | −0.36 | 37.71 | 1.72 | 43.97 | 95.69 | 29.24 | 5.35 |

> **Conclusion:** Per-model hyperparameter search did not consistently outperform the
> unified protocol used in Exp 03/05. The standard protocol (unfreeze ratio=1.0 for backbone
> head, dropout=0.5, fine-tune LR=1e-4) was retained for the main experiments.
