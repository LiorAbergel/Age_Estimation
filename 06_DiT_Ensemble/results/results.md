# Experiment 06 — Document Image Transformers (DiT): Results

> **Role in paper:** Table 3 (bottom section) — DiT individual models and
> ensemble results on the official HHD split.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 224×224 (via BeitImageProcessor) |
| Patch extraction | 400×400 patches, stride 200 (from 800 px-height images) |
| Label scaling | No |
| Augmentation | Rotation (±15°), brightness/contrast, Gaussian noise |
| Physical batch size | 128 (DiT-Base) / 16 (DiT-Large) |
| Effective patch batch size | 128 patches/step (via gradient accumulation; matches `07_DiT_CrossVal`) |
| Regression head | Global mean-pooling, Dropout (0.5), Dense (1, linear) |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Loss | MSE |
| Optimizer | Adam (no weight decay) |
| Pretrained weights | IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants) |
| Framework | PyTorch + HuggingFace Transformers |

> The configuration above is the unified protocol shared with experiments
> 01/03/04/05 and `07_DiT_CrossVal` (Adam, 1e-3/1e-4, 50/10 epochs, MSE, 128 patches/step),
> using a global-average-pooling regression head and patch-level streaming. The tables below
> are produced by `train_dit.py` and match the DiT results reported in the paper.

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| **DiT-Base (RVL-CDIP)** | **3.30** | 5.84 | **0.15** | 18.53 | **53.45** | 82.76 | 28.94 | 0.02 | **1.79** |
| DiT-Large (RVL-CDIP) | 3.52 | 5.88 | 0.14 | 20.18 | 47.41 | **84.48** | 26.85 | 0.09 | 2.17 |
| DiT-Base | 4.11 | 6.32 | 0.00 | 25.09 | 45.69 | 74.14 | **25.64** | 0.07 | 2.36 |
| DiT-Large | 4.37 | 6.33 | 0.00 | 26.57 | 28.45 | 71.55 | 26.15 | 0.03 | 2.94 |

> **Key finding:** DiT-Base (RVL-CDIP) achieves the best individual-model MAE (3.30) and leads
> on most metrics (R², MAPE, ±2 yrs, Median Error). Document-domain pretraining
> (RVL-CDIP) clearly helps: both RVL-CDIP variants (MAE 3.30–3.52) outperform their IIT-CDIP
> counterparts (MAE 4.11–4.37) by roughly a full year of MAE, and lift ±2-year accuracy from
> ~28–46% to ~47–53%.

---

## Ensemble Results — Validation-Selected, Test-Evaluated

> **Methodology:** To avoid test-set leakage, ensemble composition and weights are
> selected **only on the validation set**, then evaluated **once** on the held-out test set.
> Models are ranked by validation MAE; ensemble groups are the top-2, top-3, and all 4 models.
> Two weighting schemes are used: **Grid Search** (weights in {0.1, …, 0.9}, step 0.1, summing
> to 1, minimizing validation MAE) and **MAE-based** (weights ∝ 1 / validation MAE).

**Validation MAE ranking** (determines ensemble composition):

1. DiT-Base (RVL-CDIP) — 3.336
2. DiT-Large (RVL-CDIP) — 3.622
3. DiT-Base — 4.326
4. DiT-Large — 4.478

**Test-set ensemble metrics** (weights chosen on validation):

| Ensemble | Method | Weights (val-selected) | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|----------|--------|------------------------|----:|-----:|---:|---------:|-----------:|-----------:|----------:|----------:|-------------:|
| **Best 2** | Grid Search | B-RVL 0.7, L-RVL 0.3 | **3.18** | 5.77 | 0.17 | 17.85 | 56.03 | 83.62 | 27.68 | 0.01 | 1.70 |
| Best 3 | Grid Search | B-RVL 0.6, L-RVL 0.3, B 0.1 | 3.18 | 5.77 | 0.17 | 18.05 | 56.90 | 82.76 | 27.35 | 0.01 | 1.53 |
| Full | Grid Search | B-RVL 0.5, L-RVL 0.3, B 0.1, L 0.1 | 3.20 | 5.78 | 0.17 | 18.27 | 57.76 | 82.76 | 27.07 | 0.00 | 1.53 |
| Best 2 | MAE-based | B-RVL 0.52, L-RVL 0.48 | 3.21 | **5.76** | **0.17** | 18.11 | 56.03 | 83.62 | 26.93 | 0.02 | 1.75 |
| Best 3 | MAE-based | B-RVL 0.37, L-RVL 0.34, B 0.29 | 3.24 | 5.81 | 0.16 | 18.67 | 57.76 | 81.03 | **26.56** | 0.01 | 1.54 |
| Full | MAE-based | B-RVL 0.29, L-RVL 0.27, B 0.22, L 0.22 | 3.35 | 5.87 | 0.14 | 19.54 | 56.90 | 77.59 | 26.47 | 0.01 | 1.55 |

> Model key: **B-RVL** = DiT-Base (RVL-CDIP), **L-RVL** = DiT-Large (RVL-CDIP), **L** = DiT-Large, **B** = DiT-Base.
>
> **Key finding:** The best ensemble (**Best 2, Grid Search, MAE = 3.18**) marginally
> improves on the strongest single model (DiT-Base (RVL-CDIP), MAE = 3.30). Because that
> single model dominates the validation ranking, it receives the largest weight in every
> ensemble, so fusion gains are small. Only the two RVL-CDIP variants carry meaningful weight;
> adding the weaker IIT-CDIP models (Full Ensemble) slightly hurts MAE. All reported numbers
> use validation-selected weights, eliminating the test-set selection bias.
>
> **Statistical context:** The test set contains only ~116 pages, yielding bootstrap 95%
> confidence intervals of roughly ±1 MAE, so differences of a few tenths of a year should be read
> with caution. A paired bootstrap test (see `08_Significance/`) confirms that both the best CNN
> and best ViT ensembles significantly outperform the best DiT ensemble on MAE.

---

## Reproduction

Recompute every number above from the committed per-image predictions (no GPU, no weights):

```bash
python 06_DiT_Ensemble/reproduce_results.py            # fast path (uses predictions/)
python 06_DiT_Ensemble/reproduce_results.py --mode full  # from images + Zenodo weights
```

Outputs are written to `../reproduction_output/` and checked against the tables here.
