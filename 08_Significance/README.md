# 08_Significance — Paired Bootstrap Significance Tests

> **Paper reference:** "Ensemble Models" section — the paired bootstrap tests
> that establish which ensemble differences are statistically significant on the
> official HHD test split.

## Overview

The paper compares ensembles across architecture families and reports paired
bootstrap tests to show which MAE differences are statistically significant.
`paired_bootstrap.py` reproduces those tests from the committed per-image
predictions — no GPU, model weights, or training required.

The test resamples the 116 official test pages with replacement (10,000
resamples, paired: the same resampled pages are scored for both systems),
reports a 95% percentile confidence interval and a two-sided bootstrap p-value
for the difference in page-level MAE, and applies a **Holm–Bonferroni**
correction across the family of comparisons.

## Comparisons

| # | System A | System B | Meaning |
|---|----------|----------|---------|
| 1 | ResNet50 | CNN Best4 (Grid) | Ensemble vs. its strongest single backbone |
| 2 | CNN Best4 (Grid) | ViT Best3 (Grid) | CNN vs. ViT at the ensemble level |
| 3 | DiT Best2 (Grid) | CNN Best4 (Grid) | DiT vs. CNN |
| 4 | DiT Best2 (Grid) | ViT Best3 (Grid) | DiT vs. ViT |

All four comparisons appear in the paper: comparison 2 is the null result
(CNN vs. ViT, ΔMAE ≈ 0.04, *p* = 0.59, stated in the ViT-ensembles paragraph),
while comparisons 1, 3, and 4 are the *three significant differences* that
survive the Holm–Bonferroni correction.

The reported quantity is Δ = MAE(A) − MAE(B); a positive Δ means system B has the
lower error. Ensemble members and weights are selected on the **validation**
split (grid search over weights in {0.1, …, 0.9} summing to 1, minimizing
validation MAE) and evaluated **once** on the held-out test split, exactly as in
the experiment folders.

## Inputs

Committed per-image predictions (test + validation) from:

- `01_CNN_Ensemble/predictions/` — CNN Best4 (Grid) and ResNet50
- `04_ViT_Ensemble/predictions/` — ViT Best3 (Grid)
- `06_DiT_Ensemble/predictions/` — DiT Best2 (Grid)

## Running

```bash
# From the repository root (reproduces the paper's numbers)
python 08_Significance/paired_bootstrap.py

# Options
python 08_Significance/paired_bootstrap.py --resamples 10000 --seed 42
python 08_Significance/paired_bootstrap.py --output-dir 08_Significance/output
```

Results are printed and written to `output/paired_bootstrap.csv`.

## Expected Results

| System A | System B | ΔMAE | 95% CI | p | Significant (Holm) |
|----------|----------|-----:|--------|--:|:------------------:|
| ResNet50 | CNN Best4 (Grid) | +0.39 | [+0.16, +0.61] | <0.001 | yes |
| CNN Best4 (Grid) | ViT Best3 (Grid) | −0.04 | [−0.19, +0.11] | 0.59 | no |
| DiT Best2 (Grid) | CNN Best4 (Grid) | +0.44 | [+0.18, +0.70] | 0.002 | yes |
| DiT Best2 (Grid) | ViT Best3 (Grid) | +0.40 | [+0.12, +0.69] | 0.006 | yes |

The CNN Best4 ensemble significantly beats its strongest constituent (ResNet50),
and both the CNN and ViT ensembles significantly beat the best DiT ensemble; all
three remain significant after Holm–Bonferroni correction. The CNN and ViT
ensembles are statistically indistinguishable.
</content>
