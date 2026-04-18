# Experiment 10 — DiT Cross-Validation: Results

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
| Batch size | Dynamic (gradient accumulation to simulate larger batches) |
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
| **DiT-Base (RVL-CDIP)** | **5.85 ± 0.84** | 9.96 ± 1.42 | **0.32 ± 0.12** | 22.30 ± 3.94 | 39.01 ± 4.85 | 65.00 ± 4.64 | 83.94 ± 3.39 | 52.67 ± 13.70 | 2.90 ± 0.55 |
| DiT-Large (RVL-CDIP) | 6.09 ± 1.10 | 10.38 ± 1.65 | 0.26 ± 0.15 | 22.53 ± 4.19 | 38.42 ± 7.28 | 64.03 ± 7.52 | 83.78 ± 5.09 | 53.91 ± 12.69 | 3.15 ± 0.88 |
| DiT-Large | 6.46 ± 0.93 | 11.18 ± 1.03 | 0.14 ± 0.04 | 23.38 ± 5.29 | 36.99 ± 5.87 | 65.70 ± 7.99 | 82.95 ± 3.66 | 56.38 ± 7.29 | 3.03 ± 0.78 |
| DiT-Base | 6.50 ± 0.87 | 11.34 ± 1.16 | 0.11 ± 0.07 | 22.85 ± 4.08 | 39.63 ± 6.54 | 65.45 ± 7.17 | 82.85 ± 3.95 | 57.59 ± 7.17 | 2.95 ± 0.77 |

> **Key finding:** DiT-Base (RVL-CDIP) achieves the best cross-validated MAE (5.85 ± 0.84)
> and R² (0.32 ± 0.12) among all models, confirming the advantage of document-domain
> pretraining across folds. DiT variants consistently outperform CNN and ViT architectures.
