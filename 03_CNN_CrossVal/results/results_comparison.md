# Results Comparison: Original (Paper) vs. Re-run (Pipeline-Aligned)

> Comparison of the **original results** reported in the paper with the **new results** after re-running experiments with the corrected pipeline.

---

## Experiment 3 — CNN Cross-Validation (5-Fold Stratified Group CV)

### 3.1 Individual Models (Mean across 5 folds)

| Model | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Acc±10 (%) | Max Err | Med Err |
|-------|:-:|----:|-----:|---:|---------:|----------:|----------:|-----------:|--------:|--------:|
| **ResNet50** | Orig | 5.41 ± 0.78 | 8.17 ± 0.58 | 0.10 ± 0.06 | 25.68 ± 3.73 | 23.72 ± 5.17 | 63.40 ± 11.26 | 90.10 ± 4.42 | 32.99 ± 7.14 | 3.72 ± 0.88 |
| | New | 5.02 ± 0.30 | 7.56 ± 0.50 | 0.23 ± 0.09 | 25.01 ± 3.38 | 27.04 ± 7.12 | 69.60 ± 5.31 | 89.28 ± 3.74 | 31.23 ± 7.32 | 3.29 ± 0.51 |
| | **Δ** | **−0.39** | −0.61 | **+0.13** | −0.67 | +3.32 | **+6.20** | −0.82 | −1.76 | −0.43 |
| **DenseNet121** | Orig | 5.46 ± 1.06 | 8.16 ± 0.56 | 0.11 ± 0.06 | 26.25 ± 5.47 | 21.49 ± 16.51 | 61.61 ± 16.41 | 91.18 ± 3.84 | 34.14 ± 6.66 | 3.94 ± 1.33 |
| | New | 4.74 ± 0.48 | 7.80 ± 0.42 | 0.18 ± 0.11 | 20.66 ± 3.32 | 35.77 ± 12.88 | 75.62 ± 5.06 | 89.70 ± 3.19 | 32.05 ± 7.65 | 2.78 ± 0.82 |
| | **Δ** | **−0.72** | −0.36 | +0.07 | **−5.59** | **+14.28** | **+14.01** | −1.48 | −2.09 | **−1.16** |
| **InceptionV3** | Orig | 6.03 ± 0.64 | 8.41 ± 0.64 | 0.05 ± 0.05 | 29.97 ± 3.17 | 16.40 ± 4.30 | 52.83 ± 9.01 | 90.11 ± 4.84 | 32.70 ± 7.60 | 4.80 ± 0.67 |
| | New | 4.67 ± 0.43 | 7.69 ± 0.58 | 0.20 ± 0.11 | 21.78 ± 5.27 | 40.72 ± 5.59 | 74.27 ± 5.46 | 88.06 ± 2.85 | 32.66 ± 7.29 | 2.63 ± 0.44 |
| | **Δ** | **−1.36** | **−0.72** | **+0.15** | **−8.19** | **+24.32** | **+21.44** | −2.05 | −0.04 | **−2.17** |
| **InceptionResNetV2** | Orig | 5.69 ± 0.70 | 7.98 ± 0.68 | 0.17 ± 0.11 | 29.08 ± 3.16 | 16.76 ± 3.99 | 58.32 ± 9.19 | 89.32 ± 4.58 | 32.62 ± 5.80 | 4.37 ± 0.59 |
| | New | 4.99 ± 0.43 | 7.58 ± 0.85 | 0.21 ± 0.17 | 23.12 ± 2.26 | 26.66 ± 8.11 | 69.98 ± 4.33 | 90.19 ± 3.86 | 30.94 ± 7.97 | 3.46 ± 0.47 |
| | **Δ** | **−0.70** | −0.40 | +0.04 | **−5.96** | **+9.90** | **+11.66** | +0.87 | −1.68 | −0.91 |
| **EfficientNetV2M** | Orig | 7.30 ± 0.28 | 9.03 ± 0.36 | −0.07 ± 0.14 | 41.48 ± 5.73 | 11.58 ± 4.78 | 29.47 ± 2.21 | 84.32 ± 7.19 | 31.44 ± 5.01 | 7.05 ± 0.68 |
| | New | 5.56 ± 0.55 | 8.43 ± 0.77 | 0.05 ± 0.10 | 26.11 ± 2.94 | 22.91 ± 5.81 | 61.38 ± 10.23 | 90.62 ± 3.92 | 34.73 ± 8.35 | 4.07 ± 0.78 |
| | **Δ** | **−1.74** | −0.60 | **+0.12** | **−15.37** | **+11.33** | **+31.91** | **+6.30** | +3.29 | **−2.98** |

**Highlights:**
- **EfficientNetV2M** had the largest absolute MAE improvement (7.30 → 5.56, Δ = −1.74), and its Acc±5 nearly doubled (29.47% → 61.38%).
- **InceptionV3** showed the most dramatic ranking shift — MAE dropped from 6.03 → 4.67 (now the best individual CNN in CV).
- **All 5 models improved** in MAE, RMSE, and R² — unlike Exp 1 where one model worsened.
- **Model ranking changed:** Original: ResNet50 > DenseNet121 > InceptionResNetV2 > InceptionV3 > EfficientNetV2M. New: **InceptionV3 > DenseNet121 > InceptionResNetV2 > ResNet50 > EfficientNetV2M**.

---

### 3.2 Per-Fold MAE Comparison (New Run)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|-------:|-------:|-------:|-------:|-------:|
| ResNet50 | 5.05 | 5.16 | 5.49 | 4.66 | 4.73 |
| DenseNet121 | 4.57 | 4.95 | 5.54 | 4.13 | 4.50 |
| InceptionV3 | 4.29 | 5.19 | 5.13 | 4.12 | 4.62 |
| InceptionResNetV2 | 4.33 | 5.07 | 5.41 | 4.69 | 5.46 |
| EfficientNetV2M | 6.19 | 5.25 | 6.00 | 4.67 | 5.70 |

---

### 3.3 Test Set Ensemble (New Run)

| Metric | Value |
|--------|------:|
| **Ensemble MAE** | **3.20** |
| **Ensemble RMSE** | **5.68** |

> The new run also performs test-set inference using all 25 fold-models (5 models × 5 folds) averaged. No original test ensemble was reported in the paper for the CV experiment.

---

### 3.4 Summary

| Aspect | Change |
|--------|--------|
| **Best individual model** | ResNet50 (MAE 5.41) → **InceptionV3 (MAE 4.67)** |
| **Most improved model** | EfficientNetV2M: MAE 7.30 → 5.56 (Δ = −1.74) |
| **Largest ranking shift** | InceptionV3: 4th → **1st**; ResNet50: 1st → **4th** |
| **Std reduction** | New results show lower fold-to-fold variance (more stable training) |

**Consistent patterns across all 5 models:**

| Metrics that improved | Metrics that degraded |
|----------------------|----------------------|
| MAE (5/5 models) | Acc±10 (3/5 models, marginal: −0.8 to −2.1 pp) |
| RMSE (5/5 models) | Max Error (2/5 models) |
| R² (5/5 models) | |
| MAPE (5/5 models) | |
| Acc±2 (5/5 models) | |
| Acc±5 (5/5 models) | |
| Median Error (5/5 models) | |

Unlike Experiment 1 where the corrected pipeline showed a trade-off (better MAE but worse RMSE/tail), Experiment 3 shows **near-universal improvement** — all core metrics (MAE, RMSE, R², MAPE, Acc±2, Acc±5, median error) improved for every model. The pipeline fix had a clear positive effect on cross-validation performance with no significant trade-offs.
