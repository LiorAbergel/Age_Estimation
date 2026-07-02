# Results Comparison: Original (Paper) vs. Re-run (Pipeline-Aligned)

> Comparison of the **original results** reported in the paper with the **new results** after re-running experiments with the corrected pipeline.

---

## Experiment 5 — ViT Cross-Validation (5-Fold Stratified Group CV)

### 5.1 Individual Models (Mean ± Std across 5 folds)

| Model | | MAE | RMSE | R² | MAPE (%) | Acc±2 (%) | Acc±5 (%) | Max Err | Min Err | Med Err |
|-------|:-:|----:|-----:|---:|---------:|----------:|----------:|--------:|--------:|--------:|
| **MobileViT-XXS** | Orig | 6.00 ± 0.76 | 8.27 ± 0.27 | 0.08 ± 0.05 | 30.63 ± 4.35 | 14.08 ± 5.21 | 54.89 ± 13.95 | 31.23 ± 7.30 | — | 4.76 ± 1.01 |
| | New | 4.69 ± 0.22 | 7.82 ± 0.58 | 0.18 ± 0.09 | 21.00 ± 2.05 | 40.75 ± 5.96 | 72.23 ± 4.23 | 34.16 ± 6.69 | 0.02 ± 0.02 | 2.80 ± 0.72 |
| | **Δ** | **−1.31** | −0.45 | **+0.10** | **−9.63** | **+26.67** | **+17.34** | +2.93 | — | **−1.96** |
| **ConvNeXtV2-Tiny** | Orig | 6.81 ± 0.38 | 8.91 ± 0.51 | −0.07 ± 0.11 | 36.34 ± 6.18 | 11.08 ± 4.87 | 37.08 ± 3.37 | 32.04 ± 6.44 | — | 6.16 ± 0.56 |
| | New | 4.76 ± 0.72 | 7.51 ± 0.44 | 0.24 ± 0.10 | 22.85 ± 5.84 | 37.35 ± 14.26 | 69.08 ± 12.42 | 30.43 ± 6.72 | 0.04 ± 0.07 | 2.98 ± 1.16 |
| | **Δ** | **−2.05** | **−1.40** | **+0.31** | **−13.49** | **+26.27** | **+32.00** | −1.61 | — | **−3.18** |
| **TinyViT-11M** | Orig | 6.63 ± 0.47 | 8.71 ± 0.70 | −0.02 ± 0.13 | 35.61 ± 5.96 | 13.49 ± 5.59 | 39.59 ± 6.23 | 31.69 ± 7.04 | — | 5.80 ± 0.69 |
| | New | 5.60 ± 0.71 | 7.85 ± 0.76 | 0.17 ± 0.09 | 27.39 ± 2.49 | 16.25 ± 7.27 | 59.87 ± 10.28 | 31.06 ± 6.94 | 0.11 ± 0.16 | 4.33 ± 0.76 |
| | **Δ** | **−1.03** | −0.86 | **+0.19** | **−8.22** | +2.76 | **+20.28** | −0.63 | — | **−1.47** |
| **SwinV2-Tiny** | Orig | 6.78 ± 0.28 | 8.85 ± 0.44 | −0.05 ± 0.07 | 36.23 ± 4.42 | 12.49 ± 5.80 | 35.66 ± 1.41 | 31.98 ± 6.74 | — | 6.15 ± 0.41 |
| | New | 6.55 ± 0.56 | 8.71 ± 0.46 | −0.02 ± 0.02 | 33.40 ± 3.05 | 10.90 ± 4.29 | 44.36 ± 10.78 | 32.74 ± 7.74 | 0.32 ± 0.30 | 5.88 ± 1.09 |
| | **Δ** | −0.23 | −0.14 | +0.03 | −2.83 | −1.59 | **+8.70** | +0.76 | — | −0.27 |

**Highlights:**
- **ConvNeXtV2-Tiny** had the most dramatic improvement — MAE dropped from 6.81 → 4.76 (−2.05), jumping from worst to 2nd-best. All metrics improved substantially.
- **MobileViT-XXS** improved significantly (MAE 6.00 → 4.69) and remains the best model.
- **TinyViT-11M** also benefited strongly (MAE 6.63 → 5.60), especially Acc±5 (+20.28 pp).
- **SwinV2-Tiny** improved the least (MAE 6.78 → 6.55) and is now the worst model.
- **Model ranking changed:** Original: MobileViT-XXS > TinyViT-11M > SwinV2-Tiny > ConvNeXtV2-Tiny. New: **MobileViT-XXS > ConvNeXtV2-Tiny > TinyViT-11M > SwinV2-Tiny**.

---

### 5.2 Summary

| Aspect | Change |
|--------|--------|
| **Best model** | MobileViT-XXS (MAE 6.00 → **4.69**) — still best, but much improved |
| **Most affected model** | ConvNeXtV2-Tiny: MAE 6.81 → 4.76 (−2.05) |
| **Least affected model** | SwinV2-Tiny: MAE 6.78 → 6.55 (−0.23) |
| **Ranking shift** | ConvNeXtV2 moved from 4th → 2nd; SwinV2 moved from 3rd → 4th |

**Consistent patterns across all models:**

| Metrics that improved | Metrics that degraded |
|----------------------|----------------------|
| MAE (4/4 models) | Max Error (2/4 models, slight) |
| RMSE (4/4 models) | Acc±2 degraded for SwinV2 only |
| R² (4/4 models) | |
| MAPE (4/4 models) | |
| Acc±5 (4/4 models) | |
| Median Error (4/4 models) | |

All four models improved across nearly every metric after the pipeline correction. Unlike Experiment 1, **both average error (MAE) and error spread (RMSE) improved simultaneously**, indicating a uniformly better fit. The corrected pipeline particularly benefits models that were previously underperforming (ConvNeXtV2-Tiny), suggesting the original batch-size/pipeline mismatch disproportionately impacted certain architectures.
