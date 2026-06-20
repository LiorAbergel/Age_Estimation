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
| Batch size | 1 image/step, gradient accumulation ×4 (effective ≈ 4 images) |
| Mixed precision | FP16 (if GPU supports it) |
| Training epochs | 15 (frozen backbone) |
| Training LR | 1e-4 |
| Fine-tune epochs | 30 |
| Fine-tune LR | 1e-5 |
| Optimizer | AdamW |
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants) |
| Framework | PyTorch + HuggingFace Transformers |

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
