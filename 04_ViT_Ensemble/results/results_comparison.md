# Results Comparison: Original vs. Re-run (Pipeline-Aligned)

> Comparison of the **original results** (`original_results.md`) with the **new results** after re-running experiments with the corrected pipeline.

---

## Experiment 4 — ViT Ensemble (Official Split)

### 4.1 Individual Models

| Model | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Max Err | Min Err | Med Err |
|-------|:-:|----:|-----:|---:|---------:|----------:|----------:|--------:|--------:|--------:|
| **MobileViT-XXS** | Orig | 4.42 | 6.54 | −0.07 | 25.90 | 16.38 | 74.14 | 31.44 | — | 3.38 |
| | New | 2.79 | 6.13 | 0.06 | 13.59 | 65.52 | 93.97 | 34.91 | 0.05 | 1.35 |
| | **Δ** | **−1.63** | −0.41 | **+0.13** | **−12.31** | **+49.14** | **+19.83** | +3.47 | — | **−2.03** |
| **ConvNeXtV2-Tiny** | Orig | 6.30 | 7.58 | −0.44 | 38.78 | 1.72 | 41.38 | 30.36 | — | 5.64 |
| | New | 3.86 | 6.08 | 0.08 | 21.87 | 37.07 | 81.03 | 27.99 | 0.13 | 2.98 |
| | **Δ** | **−2.44** | **−1.50** | **+0.52** | **−16.91** | **+35.35** | **+39.65** | −2.37 | — | **−2.66** |
| **TinyViT-11M** | Orig | 6.70 | 7.79 | −0.52 | 41.92 | 1.72 | 35.34 | 28.70 | — | 5.87 |
| | New | 4.69 | 6.23 | 0.03 | 28.95 | 18.97 | 70.69 | 25.95 | 0.06 | 3.68 |
| | **Δ** | **−2.01** | **−1.56** | **+0.55** | **−12.97** | **+17.25** | **+35.35** | −2.75 | — | **−2.19** |
| **SwinV2-Tiny** | Orig | 6.98 | 8.04 | −0.62 | 43.58 | 1.72 | 18.10 | 29.58 | — | 6.42 |
| | New | 5.46 | 7.46 | −0.39 | 30.87 | 11.21 | 55.17 | 32.41 | 0.16 | 4.83 |
| | **Δ** | **−1.52** | −0.58 | **+0.23** | **−12.71** | **+9.49** | **+37.07** | +2.83 | — | **−1.59** |

**Highlights:**
- **All 4 models improved substantially** — every model saw MAE drop by 1.5–2.4 years.
- **MobileViT-XXS** had the most dramatic accuracy jump — Acc±2 went from 16.38% → 65.52% (+49 pp). It is now by far the best individual ViT (MAE 2.79).
- **ConvNeXtV2-Tiny** improved most in MAE terms (6.30 → 3.86, Δ = −2.44) and Acc±5 (+39.65 pp).
- **Model ranking unchanged:** MobileViT-XXS remains best; SwinV2-Tiny remains worst. But the gap between best and worst narrowed significantly (original spread 2.56, new spread 2.67).

---

### 4.2 Ensemble Results

> The original pipeline used a single equal-weight average of all 4 models. The re-run adds grid search and MAE-based weighting (selected on validation, evaluated on test).

| Ensemble | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Max Err | Min Err | Med Err |
|----------|:-:|----:|-----:|---:|---------:|----------:|----------:|--------:|--------:|--------:|
| **Equal-weight (all 4)** | Orig | 6.09 | 7.42 | −0.38 | 37.51 | 1.72 | 41.38 | 30.02 | — | 5.33 |
| | New | 3.06 | 5.76 | 0.17 | 16.27 | 53.45 | 85.34 | 30.31 | 0.00 | 1.68 |
| | **Δ** | **−3.03** | **−1.66** | **+0.55** | **−21.24** | **+51.73** | **+43.96** | +0.29 | — | **−3.65** |
| **Best 3 (Grid Search)** | New | **2.77** | 5.84 | 0.15 | 14.41 | 66.38 | 87.93 | 32.42 | 0.03 | 1.26 |
| Best 2 (Grid Search) | New | 2.81 | 6.06 | 0.08 | 13.87 | 65.52 | 93.10 | 34.21 | 0.03 | 1.44 |
| Full Ensemble (Grid Search) | New | 2.89 | 5.57 | 0.22 | 15.75 | 64.66 | 82.76 | 28.99 | 0.06 | 1.40 |
| Best 3 (MAE-based) | New | 2.89 | 5.68 | 0.19 | 15.78 | 63.79 | 81.90 | 29.89 | 0.01 | 1.24 |
| Full Ensemble (MAE-based) | New | 3.03 | 5.75 | 0.17 | 16.09 | 53.45 | 85.34 | 30.38 | 0.00 | 1.68 |
| Best 2 (MAE-based) | New | 3.04 | 5.93 | 0.12 | 15.80 | 55.17 | 86.21 | 31.84 | 0.01 | 1.77 |

**Highlights:**
- The equal-weight ensemble MAE halved from 6.09 → 3.06 (−3.03).
- **Best 3 (Grid Search)** is the best overall configuration at MAE 2.77, beating the best individual model (MobileViT-XXS, MAE 2.79).
- Acc±2 jumped from 1.72% → 66.38% (+64.66 pp vs original) for the best ensemble.
- R² went from strongly negative (−0.38) to positive (0.22 for Full Grid), indicating the ensembles now outperform the mean baseline.
- Grid search consistently outperforms MAE-based weighting for the same model group.

---

### 4.3 Summary

| Aspect | Change |
|--------|--------|
| **Best individual model** | MobileViT-XXS: MAE 4.42 → **2.79** (−1.63) |
| **Best ensemble** | Equal-weight (MAE 6.09) → **Best 3 Grid Search (MAE 2.77)** (−3.32) |
| **Most improved (individual)** | ConvNeXtV2-Tiny: MAE 6.30 → 3.86 (−2.44) |
| **Most improved (ensemble)** | Equal-weight: MAE 6.09 → 3.06 (−3.03) |

**Consistent patterns across all models:**

| Metrics that improved | Metrics that degraded |
|----------------------|----------------------|
| MAE (4/4 models + ensemble) | Max Error (2/4 models, +2.8 to +3.5) |
| RMSE (4/4 models + ensemble) | |
| R² (4/4 models + ensemble) | |
| MAPE (4/4 models + ensemble) | |
| Acc±2 (4/4 models + ensemble) | |
| Acc±5 (4/4 models + ensemble) | |
| Median Error (4/4 models + ensemble) | |

The corrected pipeline produces **dramatically better results** across nearly all metrics. Unlike Experiment 1 (where RMSE/R² degraded), here **both central and spread metrics improved**. The only minor degradation is in Max Error for 2 models. This suggests the original pipeline was severely misaligned for the ViT models, suppressing their predictive power far more than for the CNNs.
