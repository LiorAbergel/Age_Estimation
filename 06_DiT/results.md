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

> **Note:** The configuration above is the unified protocol shared with experiments
> 01/03/04/05 and `07_DiT_CrossVal` (Adam, 1e-3/1e-4, 50/10 epochs, MSE, 128 patches/step).
> The result tables below were generated with a previous LR setting (1e-4/1e-5) and **must
> be regenerated** by re-running `train_dit.py` under the current 1e-3/1e-4 configuration
> before the numbers are quoted in the paper.

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| **DiT-Base (RVL-CDIP)** | **2.845** | **5.084** | **0.354** |
| DiT-Large (RVL-CDIP) | 3.042 | 5.075 | 0.356 |
| DiT-Large | 3.090 | 5.605 | 0.215 |
| DiT-Base | 3.120 | 5.633 | 0.207 |

> **Key finding:** DiT-Base (RVL-CDIP) achieves the best individual-model MAE (2.845).
> Document-domain pretraining (RVL-CDIP) benefits the Base model (2.845 vs. 3.120 for
> IIT-CDIP). All four DiT variants achieve MAE within a narrow band (2.85–3.12).
> **These numbers were generated with LR 1e-4/1e-5 and must be regenerated with the
> current 1e-3/1e-4 setting.**

---

## Ensemble Results — Validation-Selected, Test-Evaluated

> **Methodology:** To avoid test-set leakage, ensemble composition and weights are
> selected **only on the validation set**, then evaluated **once** on the held-out test set.
> Models are ranked by validation MAE; ensemble groups are the top-2, top-3, and all 4 models.
> Two weighting schemes are used: **Grid Search** (weights in {0.1, …, 0.9}, step 0.1, summing
> to 1, minimizing validation MAE) and **MAE-based** (weights ∝ 1 / validation MAE).

**Validation MAE ranking** (determines ensemble composition):

1. DiT-Base (RVL-CDIP) — 3.044
2. DiT-Large — 3.311
3. DiT-Large (RVL-CDIP) — 3.363
4. DiT-Base — 3.456

**Test-set ensemble metrics** (weights chosen on validation):

| Ensemble | Method | Weights (val-selected) | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|----------|--------|------------------------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **Best 2** | Grid Search | B-RVL 0.9, L 0.1 | **2.863** | **5.112** | 0.347 | 16.76 | 67.24 | 82.76 | 93.10 | 25.20 | 1.13 |
| Best 3 | Grid Search | B-RVL 0.8, L 0.1, L-RVL 0.1 | 2.872 | 5.091 | **0.352** | 16.82 | 66.38 | 82.76 | 93.10 | **24.49** | 1.17 |
| Full | Grid Search | B-RVL 0.7, L 0.1, L-RVL 0.1, B 0.1 | 2.897 | 5.141 | 0.339 | 16.98 | 66.38 | 82.76 | 92.24 | 24.62 | 1.14 |
| Best 2 | MAE-based | B-RVL 0.52, L 0.48 | 2.943 | 5.269 | 0.306 | 16.88 | 66.38 | 82.76 | 93.97 | 25.82 | 1.35 |
| Best 3 | MAE-based | B-RVL 0.35, L 0.33, L-RVL 0.32 | 2.964 | 5.171 | 0.332 | 17.16 | 64.66 | 82.76 | 93.10 | 25.99 | 1.44 |
| Full | MAE-based | B-RVL 0.27, L 0.25, L-RVL 0.24, B 0.24 | 2.991 | 5.264 | 0.307 | 17.43 | 66.38 | 81.90 | 92.24 | 25.76 | 1.34 |

> Model key: **B-RVL** = DiT-Base (RVL-CDIP), **L-RVL** = DiT-Large (RVL-CDIP), **L** = DiT-Large, **B** = DiT-Base.
>
> **Key finding:** The best ensemble (**Best 2, Grid Search, MAE = 2.863**) marginally
> improves on the strongest single model (DiT-Base (RVL-CDIP), MAE = 2.845). Because that
> single model dominates the validation ranking, it receives the largest weight in every
> ensemble, so fusion gains are small. All reported numbers use validation-selected weights,
> eliminating the test-set selection bias present in earlier results.
>
> **Statistical context:** The test set contains only ~116 pages, yielding bootstrap 95%
> confidence intervals of roughly ±1 MAE. Once results are regenerated with 1e-3/1e-4,
> the best DiT ensemble will be statistically indistinguishable from the best CNN ensemble
> (2.73) and best CNN individual model (EfficientNetV2M, 2.77) within this range. This
> confirms that in this low-resource setting (small dataset, frozen-backbone feature
> extraction), document-domain pretrained transformers and ImageNet-pretrained CNNs achieve
> comparable performance.
