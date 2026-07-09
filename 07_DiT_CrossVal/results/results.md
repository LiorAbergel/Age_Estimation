# Experiment 07 — DiT Cross-Validation: Results

> **Role in paper:** Table 2 (bottom section) — DiT individual models, 5-fold stratified group CV.
> Results reported as mean ± std across 5 folds. Folds are stratified by AgeGroup and grouped
> by WriterNumber (no writer appears in more than one fold).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 224×224 (via BeitImageProcessor) |
| Label scaling | No |
| Augmentation | Rotation (±15°), brightness/contrast, Gaussian noise |
| Physical batch size | 128 (DiT-Base) / 16 (DiT-Large) |
| Effective patch batch size | 128 patches/step (via gradient accumulation; matches `06_DiT_Ensemble`) |
| Regression head | Global mean-pooling, Dropout (0.5), Dense (1, linear) |
| Mixed precision | FP16 (if GPU supports it) |
| Training epochs | 50 (frozen backbone) |
| Training LR | 1e-3 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| Loss | MSE |
| Optimizer | Adam (no weight decay) |
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants) |
| Framework | PyTorch + HuggingFace Transformers |

> The configuration above is the unified protocol shared with experiments
> 01/03/04/05 and `06_DiT_Ensemble` (Adam, 1e-3/1e-4, 50/10 epochs, MSE, 128 patches/step),
> using a global-average-pooling regression head and patch-level streaming. The tables below
> are produced by `train_dit_cv.py` and match the DiT cross-validation results reported in
> the paper (Table 2, bottom section).

---

## Results — Original HHD (5-Fold Stratified Group CV)

Values are **mean ± std** across 5 folds (population std, ddof=0).

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | Max Error | Min Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-----------|-----------|--------------|
| **DiT-Base (RVL-CDIP)** | **5.01 ± 0.53** | **7.74 ± 0.77** | 0.19 ± 0.11 | **23.85 ± 1.31** | 31.31 ± 2.83 | **69.29 ± 6.43** | 31.80 ± 7.30 | **0.03 ± 0.02** | 3.26 ± 0.30 |
| DiT-Base | 5.12 ± 0.59 | 7.70 ± 0.53 | **0.20 ± 0.08** | 25.42 ± 4.01 | **31.90 ± 7.88** | 66.36 ± 4.82 | **31.35 ± 7.98** | **0.03 ± 0.03** | **3.24 ± 0.71** |
| DiT-Large (RVL-CDIP) | 5.32 ± 0.53 | 7.87 ± 0.62 | 0.16 ± 0.12 | 25.99 ± 2.71 | 21.58 ± 11.97 | 67.89 ± 6.81 | 32.94 ± 8.89 | 0.06 ± 0.08 | 3.66 ± 0.69 |
| DiT-Large | 5.92 ± 0.93 | 8.24 ± 0.57 | 0.09 ± 0.07 | 30.45 ± 6.65 | 21.70 ± 9.31 | 52.38 ± 20.99 | 33.49 ± 8.38 | 0.05 ± 0.05 | 4.77 ± 1.53 |

> **Key finding:** **DiT-Base (RVL-CDIP)** is the best DiT variant (MAE 5.01 ± 0.53), and
> document-domain pretraining helps — both RVL-CDIP variants beat their IIT-CDIP counterparts
> on MAE. Under the unified protocol the DiT family does not lead in cross-validation: the best
> CNN CV model (InceptionV3, MAE 4.67 ± 0.48, `03_CNN_CrossVal`) and the best ViT CV model
> (MobileViT-XXS, MAE 4.69 ± 0.22, `05_ViT_CrossVal`) both outperform the best DiT.
>
> **Statistical context:** Each fold's validation set contains ~140 pages, yielding bootstrap
> 95% confidence intervals of roughly ±1 MAE, so the ~0.3–0.4-year gap between the best DiT
> and the best CNN/ViT should be read with caution — the small dataset limits the strength of
> cross-architecture comparisons.

---

## Reproduction

Recompute every number above from the committed out-of-fold predictions (no GPU, no weights):

```bash
python 07_DiT_CrossVal/reproduce_results.py            # fast path (uses predictions/oof_predictions.csv)
python 07_DiT_CrossVal/reproduce_results.py --mode full  # from images + Zenodo weights
```

Outputs are written to `../reproduction_output/` and checked against the tables here.
