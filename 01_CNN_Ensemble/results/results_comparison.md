# Results Comparison: Original (Paper) vs. Re-run (Pipeline-Aligned)

> Comparison of the **original results** reported in the paper with the **new results** after re-running experiments with the corrected pipeline.

---

## Experiment 1 — CNN Ensemble (Official Split)

### 1.1 Individual Models

| Model | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Max Err | Min Err | Med Err |
|-------|:-:|----:|-----:|---:|---------:|----------:|----------:|--------:|--------:|--------:|
| **ResNet50** | Orig | 3.23 | 6.03 | 0.09 | 18.13 | 56.90 | 81.90 | 30.60 | 0.00 | 1.59 |
| | New | 3.12 | 6.38 | −0.02 | 16.14 | 50.86 | 93.97 | 34.55 | 0.01 | 1.89 |
| | **Δ** | **−0.11** | +0.35 | −0.11 | −1.99 | −6.04 | **+12.07** | +3.95 | +0.01 | +0.30 |
| **InceptionV3** | Orig | 5.06 | 6.55 | −0.07 | 31.29 | 6.90 | 67.24 | 27.78 | 0.10 | 4.12 |
| | New | 3.31 | 6.20 | 0.04 | 18.05 | 48.28 | 87.07 | 33.98 | 0.06 | 2.04 |
| | **Δ** | **−1.75** | −0.35 | +0.11 | **−13.24** | **+41.38** | **+19.83** | +6.20 | **−0.04** | **−2.08** |
| **DenseNet121** | Orig | 3.97 | 6.10 | 0.07 | 22.80 | 33.62 | 82.76 | 26.85 | 0.14 | 2.76 |
| | New | 3.80 | 6.45 | −0.04 | 20.70 | 38.79 | 86.21 | 29.61 | 0.04 | 2.82 |
| | **Δ** | **−0.17** | +0.35 | −0.11 | −2.10 | +5.17 | +3.45 | +2.76 | **−0.10** | +0.06 |
| **InceptionResNetV2** | Orig | 3.64 | 6.39 | −0.02 | 20.12 | 43.97 | 87.07 | 32.62 | 0.03 | 2.46 |
| | New | 4.00 | 6.50 | −0.06 | 22.59 | 26.72 | 81.90 | 32.71 | 0.21 | 2.69 |
| | **Δ** | +0.36 | +0.11 | −0.04 | +2.47 | **−17.25** | −5.17 | +0.09 | +0.18 | +0.23 |
| **EfficientNetV2M** | Orig | 7.17 | 8.12 | −0.65 | 45.19 | 0.86 | 16.38 | 28.06 | 1.24 | 6.39 |
| | New | 2.77 | 6.28 | 0.01 | 13.45 | 66.38 | 93.97 | 34.36 | 0.07 | 1.25 |
| | **Δ** | **−4.40** | **−1.84** | **+0.66** | **−31.74** | **+65.52** | **+77.59** | +6.30 | **−1.17** | **−5.14** |

**Highlights:**
- **EfficientNetV2M** had the most dramatic change — MAE dropped from 7.17 → 2.77 (now the best individual CNN). The original pipeline clearly mishandled this model.
- **InceptionV3** also improved substantially (MAE 5.06 → 3.31).
- **InceptionResNetV2** is the only model that worsened (MAE 3.64 → 4.00).
- **Model ranking changed:** Original: ResNet50 > InceptionResNetV2 > DenseNet121 > InceptionV3 > EfficientNetV2M. New: **EfficientNetV2M > ResNet50 > InceptionV3 > DenseNet121 > InceptionResNetV2**.

---

### 1.2 Ensemble Results

| Ensemble | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Max Err | Min Err | Med Err |
|----------|:-:|----:|-----:|---:|---------:|----------:|----------:|--------:|--------:|--------:|
| **Best2 Grid** | Orig | 3.24 | 6.04 | 0.09 | 18.17 | 56.90 | 81.90 | 30.80 | 0.00 | 1.58 |
| | New | 3.18 | 6.38 | −0.02 | 16.60 | 50.86 | 93.10 | 34.36 | 0.07 | 1.91 |
| | **Δ** | −0.06 | +0.34 | −0.11 | −1.57 | −6.04 | **+11.20** | +3.56 | +0.07 | +0.33 |
| **Best2 MAE** | Orig | 3.35 | 6.14 | 0.06 | 18.61 | 53.45 | 82.76 | 31.55 | 0.02 | 1.69 |
| | New | 3.45 | 6.39 | −0.02 | 18.66 | 44.83 | 89.66 | 33.71 | 0.05 | 2.15 |
| | **Δ** | +0.10 | +0.25 | −0.08 | +0.05 | −8.62 | **+6.90** | +2.16 | +0.03 | +0.46 |
| **Best3 Grid** | Orig | **2.85** | 5.87 | 0.14 | 15.30 | 68.97 | 84.48 | 30.11 | 0.03 | 1.26 |
| | New | 2.75 | 6.05 | 0.08 | 14.12 | 71.55 | 87.07 | 31.35 | 0.02 | 1.15 |
| | **Δ** | **−0.10** | +0.18 | −0.06 | −1.18 | +2.58 | +2.59 | +1.24 | −0.01 | −0.11 |
| **Best3 MAE** | Orig | 2.86 | 5.86 | 0.14 | 15.48 | 68.10 | 82.76 | 30.08 | 0.01 | 1.21 |
| | New | 2.76 | 6.11 | 0.07 | 14.16 | 70.69 | 88.79 | 32.34 | 0.01 | 1.11 |
| | **Δ** | **−0.10** | +0.25 | −0.07 | −1.32 | +2.59 | **+6.03** | +2.26 | 0.00 | −0.10 |
| **Best4 Grid** | Orig | 2.90 | 5.83 | 0.15 | 15.85 | 68.10 | 81.90 | 29.62 | 0.03 | 1.19 |
| | New | 2.73 | 6.05 | 0.08 | 14.13 | 71.55 | 87.07 | 31.78 | 0.01 | 1.20 |
| | **Δ** | **−0.17** | +0.22 | −0.07 | −1.72 | +3.45 | **+5.17** | +2.16 | −0.02 | +0.01 |
| **Best4 MAE** | Orig | 3.14 | 5.88 | 0.14 | 17.71 | 55.17 | 81.03 | 29.54 | 0.01 | 1.36 |
| | New | 2.82 | 6.10 | 0.07 | 14.66 | 59.48 | 87.07 | 32.75 | 0.01 | 1.13 |
| | **Δ** | **−0.32** | +0.22 | −0.07 | −3.05 | +4.31 | **+6.04** | +3.21 | 0.00 | −0.23 |
| **Full Grid** | Orig | 3.01 | 5.81 | 0.16 | 16.88 | 67.24 | 81.03 | 28.59 | 0.02 | 1.31 |
| | New | **2.73** | 6.09 | 0.07 | 13.93 | 72.41 | 88.79 | 32.42 | 0.04 | 1.22 |
| | **Δ** | **−0.28** | +0.28 | −0.09 | −2.95 | +5.17 | **+7.76** | +3.83 | +0.02 | −0.09 |
| **Full MAE** | Orig | 3.75 | 6.08 | 0.08 | 21.97 | 41.38 | 80.17 | 29.26 | 0.13 | 2.30 |
| | New | 2.76 | 6.13 | 0.06 | 14.12 | 67.24 | 89.66 | 33.08 | 0.00 | 1.12 |
| | **Δ** | **−0.99** | +0.05 | −0.02 | **−7.85** | **+25.86** | **+9.49** | +3.82 | **−0.13** | **−1.18** |

---

### 1.3 Ensemble Weights Changes

| Ensemble | Paper Weights | New Weights |
|----------|---------------|-------------|
| Full Grid | Res=0.1, IncRN=0.2, Dense=**0.5**, IncV3=0.1, Eff=0.1 | Res=0.1, IncRN=**0.3**, Dense=**0.3**, IncV3=0.1, Eff=**0.2** |
| Best4 Grid | Res=0.1, IncRN=0.4, Dense=0.4, IncV3=0.1 | *(unchanged)* |
| Best3 Grid | Res=0.1, IncRN=**0.5**, Dense=**0.4** | Res=0.1, IncRN=**0.4**, Dense=**0.5** |
| Best2 Grid | Res=0.9, IncRN=0.1 | *(unchanged)* |

All MAE-based weights shifted due to the changed individual-model validation MAEs. Grid-search weights changed for Full and Best3 ensembles.

---

### 1.4 Summary

| Aspect | Change |
|--------|--------|
| **Best individual model** | ResNet50 (MAE 3.23) → **EfficientNetV2M (MAE 2.77)** |
| **Best ensemble** | Best3 Grid (MAE 2.85) → **Full Grid (MAE 2.73)** |
| **Most affected model** | EfficientNetV2M: MAE 7.17 → 2.77 (−4.40) |
| **Most affected ensemble** | Full MAE: MAE 3.75 → 2.76 (−0.99) |

**Consistent patterns across all configurations:**

| Metrics that improved | Metrics that degraded |
|----------------------|----------------------|
| MAE (7/8 configs) | RMSE (8/8 configs, +0.05 to +0.34) |
| MAPE (7/8 configs) | R² (8/8 configs, −0.02 to −0.11) |
| Acc±5 (8/8 configs) | Max Error (8/8 configs, +1.2 to +3.8) |
| Acc±2 (6/8 configs) | |
| Median Error (5/8 configs) | |

The corrected pipeline produces **lower average errors** (MAE/MAPE/median) and **higher threshold accuracy** (Acc±2, Acc±5), but with a **wider error tail** (higher RMSE, max error, lower R²). This is consistent with fixing a pipeline issue that previously suppressed extreme predictions — the corrected models make bolder predictions that are right more often on average, but miss by more on the hardest cases.
