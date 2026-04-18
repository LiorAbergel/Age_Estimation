# Experiment 01 — Baseline CNN: Results

> **Role in paper:** Development experiment — not reported in the paper.
> This custom baseline CNN establishes a lower-bound reference before transfer learning.

---

## Architecture

```
Sequential CNN
  Conv2D(32,  3×3, ReLU) → MaxPooling(2×2)
  Conv2D(64,  3×3, ReLU) → MaxPooling(2×2)
  Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
  Flatten
  Dense(128, ReLU)
  Dropout(0.5)
  Dense(1, linear)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 128×128 (full images, no patch extraction) |
| Label scaling | Yes (ages divided by max age) |
| Augmentation | rotation\_range=20, width\_shift=0.2, height\_shift=0.2, shear=0.2, zoom=0.2, horizontal\_flip |
| Batch size | 32 |
| Training epochs | 50 |
| Training LR | 1e-3 |
| Fine-tune epochs | — |
| Fine-tune LR | — |

---

## Results — Original HHD (Official Split)

| MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs | ±10 yrs | Max Error | Median Error |
|----:|-----:|---:|---------:|-----------:|-------:|--------:|----------:|-------------:|
| 11.13 | 6.58 | −0.08 | 18.66 | 50.00 | 0.853 | 0.966 | 32.00 | 2.29 |
