# Experiment 04 — Label Scaling + LR Scheduling: Results

> **Role in paper:** Development experiment — not reported in the paper.
> Investigates the effect of label scaling and ExponentialDecay LR scheduling.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | Yes |
| Augmentation | Rotation, zoom, brightness, Gaussian noise, translation, contrast (more aggressive than Exp 03) |
| Batch size | 128 |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 with ExponentialDecay |
| Fine-tune epochs | 20 |
| Fine-tune LR | 1e-4 with ExponentialDecay |
| Pretrained weights | ImageNet-1K |

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| InceptionV3 | 3.42 | 6.19 | 0.04 | 19.04 | 44.83 | 81.03 | 96.55 | 32.06 | 2.47 |
| Full Ensemble (3 models) | 4.93 | 6.71 | −0.13 | 29.52 | 9.48 | 69.83 | 96.55 | 30.18 | 3.88 |
| DenseNet121 | 5.73 | 7.15 | −0.28 | 34.82 | 2.59 | 55.17 | 96.55 | 29.64 | 4.82 |
| ResNet50 | 5.75 | 7.21 | −0.30 | 35.34 | 1.72 | 50.00 | 95.69 | 28.86 | 5.00 |

> **Conclusion:** Label scaling combined with ExponentialDecay LR did not improve over the
> fixed-LR approach in Exp 03. The standard protocol (no scaling, fixed LR with step decay)
> was adopted for subsequent experiments.
