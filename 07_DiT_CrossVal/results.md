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
| Effective patch batch size | 128 patches/step (via gradient accumulation; matches `06_DiT`) |
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

> **Note:** The configuration above is the unified protocol shared with experiments
> 01/03/04/05 and `06_DiT` (Adam, 1e-3/1e-4, 50/10 epochs, MSE, 128 patches/step).
> The result tables below **have not yet been regenerated** under this aligned configuration
> and should be re-run before the numbers are quoted in the paper.

---

## Results — Original HHD (5-Fold Stratified Group CV)

Values are **mean ± std** across 5 folds.

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|-----|------|-----|----------|------------|------------|-------------|-----------|--------------|
| **DiT-Large (RVL-CDIP)** | **3.47 ± 0.54** | **6.26 ± 0.96** | **0.46 ± 0.15** | 15.33 ± 4.66 | 59.51 ± 5.25 | 78.34 ± 7.02 | — | 29.37 ± 6.95 | 1.41 ± 0.34 |
| DiT-Base (RVL-CDIP) | 3.68 ± 0.47 | 6.41 ± 0.83 | 0.44 ± 0.14 | 16.66 ± 4.37 | 55.83 ± 4.27 | 77.05 ± 6.40 | — | 29.90 ± 8.48 | 1.68 ± 0.18 |
| DiT-Base | 3.74 ± 0.53 | 6.72 ± 0.80 | 0.38 ± 0.16 | 16.52 ± 5.74 | 57.95 ± 7.32 | 78.32 ± 5.20 | — | 31.43 ± 8.26 | 1.63 ± 0.41 |
| DiT-Large | 3.83 ± 0.49 | 6.78 ± 0.94 | 0.38 ± 0.13 | 16.98 ± 4.92 | 55.78 ± 3.94 | 78.47 ± 5.48 | — | 31.59 ± 8.01 | 1.73 ± 0.21 |

> The paper's Table 2 does not report Acc±10 (±10 yrs); that column is shown as "—".

> **Key finding:** DiT-Large (RVL-CDIP) achieves the best cross-validated MAE (3.47 ± 0.54)
> and R² (0.46 ± 0.15) among all models, confirming the advantage of document-domain
> pretraining across folds. DiT variants consistently outperform CNN and ViT architectures.
>
> **Statistical context:** The cross-validation folds contain ~140 pages each, yielding
> bootstrap confidence intervals of roughly ±1 MAE. The top DiT models (3.47–3.83) are
> statistically indistinguishable from each other within this range. The best DiT CV result
> (3.47 ± 0.54) is comparable to the best CNN CV result (ResNet50, 5.41 ± 0.78) and best
> ViT CV result (MobileViT-XXS, 4.69 ± 0.22) — DiT's lower MAE is consistent across folds
> but the small dataset limits the strength of cross-architecture comparisons.
>
> **Honest finding:** Under the aligned pipeline (frozen-backbone feature extraction with
> 50+10 epochs, MSE, Adam 1e-3/1e-4), document-domain pretrained transformers are expected
> to be competitive but not dramatically outperform other architectures. The original DiT
> recipe (L1 loss, AdamW, 15+30 epochs, CLS-token pooling) produced lower MAE numbers, but
> used a different optimization protocol. Both `06_DiT` and `07_DiT_CrossVal` results need
> to be regenerated under the current aligned configuration for fair cross-architecture
> comparison.
