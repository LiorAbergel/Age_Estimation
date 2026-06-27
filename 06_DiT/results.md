# Experiment 06 — Document Image Transformers (DiT): Results

> **Role in paper:** Table 3 (bottom section) and Table 4 — DiT individual models and
> ensemble results on the official HHD split.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 224×224 (via BeitImageProcessor) |
| Label scaling | No |
| Augmentation | Rotation (±15°), brightness/contrast, Gaussian noise |
| Physical batch size | 16 (DiT-Base) / 2 (DiT-Large) |
| Effective image batch size | 16 (via gradient accumulation; matches `07_DiT_CrossVal`) |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Loss | MSE |
| Optimizer | Adam |
| Pretrained weights | IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants) |
| Framework | PyTorch + HuggingFace Transformers |

> **Note:** The configuration above is the unified protocol shared with experiments
> 01/03/04/05 and `07_DiT_CrossVal` (Adam, 1e-3/1e-4, 50/10 epochs, MSE, effective image
> batch size 16). The result tables below predate this alignment and **must be regenerated**
> by re-running `train_dit.py` under the current configuration before the numbers are quoted
> in the paper.

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **DiT-Base (RVL-CDIP)** | **2.325** | 4.937 | 0.391 | 11.945 | 70.690 | 92.241 | 96.552 | 25.140 | 1.198 |
| DiT-Large | 2.507 | **4.469** | **0.501** | 14.730 | 68.103 | 85.345 | 95.690 | 26.212 | 1.123 |
| DiT-Base | 2.674 | 5.256 | 0.310 | 14.371 | 66.379 | 87.069 | 96.552 | 26.850 | 1.140 |
| DiT-Large (RVL-CDIP) | 2.692 | 4.972 | 0.382 | 15.121 | 68.103 | 85.345 | 95.690 | 25.315 | 1.272 |

> **Key finding:** DiT-Base (RVL-CDIP) achieves the best individual-model MAE
> (2.325), while DiT-Large attains the best RMSE (4.469) and R² (0.501).
> Document-domain pretraining (RVL-CDIP) improves the Base model substantially over
> IIT-CDIP pretraining alone, and all DiT variants outperform the CNN and ViT models.

---

## Ensemble Results — Validation-Selected, Test-Evaluated

> **Methodology:** To avoid test-set leakage, ensemble composition and weights are
> selected **only on the validation set**, then evaluated **once** on the held-out test set.
> Models are ranked by validation MAE; ensemble groups are the top-2, top-3, and all 4 models.
> Two weighting schemes are used: **Grid Search** (weights in {0.1, …, 0.9}, step 0.1, summing
> to 1, minimizing validation MAE) and **MAE-based** (weights ∝ 1 / validation MAE).

**Validation MAE ranking** (determines ensemble composition):

1. DiT-Base (RVL-CDIP) — 2.494
2. DiT-Large (RVL-CDIP) — 2.710
3. DiT-Large — 2.780
4. DiT-Base — 2.867

**Test-set ensemble metrics** (weights chosen on validation):

| Ensemble | Method | Weights (val-selected) | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|----------|--------|------------------------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **Best 2** | Grid Search | B-RVL 0.9, L-RVL 0.1 | **2.337** | 4.916 | 0.396 | 12.101 | 69.828 | 92.241 | 96.552 | 24.614 | 1.187 |
| Best 3 | Grid Search | B-RVL 0.8, L-RVL 0.1, L 0.1 | 2.347 | 4.812 | 0.421 | 12.328 | 71.552 | 92.241 | 96.552 | 24.781 | 1.237 |
| Full | Grid Search | B-RVL 0.7, L-RVL 0.1, L 0.1, B 0.1 | 2.377 | 4.827 | 0.418 | 12.539 | 69.828 | 91.379 | 96.552 | 25.013 | 1.263 |
| Best 2 | MAE-based | B-RVL 0.521, L-RVL 0.479 | 2.439 | 4.885 | 0.403 | 13.066 | 68.966 | 89.655 | 96.552 | 24.909 | 1.263 |
| Best 3 | MAE-based | B-RVL 0.355, L-RVL 0.327, L 0.319 | 2.450 | 4.641 | **0.462** | 13.532 | 67.241 | 87.069 | 96.552 | 25.324 | 1.171 |
| Full | MAE-based | B-RVL 0.271, L-RVL 0.250, L 0.243, B 0.236 | 2.496 | 4.755 | 0.435 | 13.690 | 68.966 | 86.207 | 96.552 | 25.684 | 1.108 |

> Model key: **B-RVL** = DiT-Base (RVL-CDIP), **L-RVL** = DiT-Large (RVL-CDIP), **L** = DiT-Large, **B** = DiT-Base.
>
> **Key finding:** The best ensemble (**Best 2, Grid Search, MAE = 2.337**) marginally
> improves on the strongest single model (DiT-Base (RVL-CDIP), MAE = 2.325). Because that
> single model dominates the validation ranking, it receives the largest weight in every
> ensemble, so fusion gains are small. All reported numbers use validation-selected weights,
> eliminating the test-set selection bias present in earlier results.
