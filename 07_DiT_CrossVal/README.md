# 07_DiT_CrossVal — DiT Cross-Validation

> **Paper reference:** Table 2 (bottom section) — DiT individual models, 5-fold stratified group CV.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_dit_cv.py` | Training pipeline — 5-fold stratified group CV with 4 DiT backbones |
| `reproduce_results.py` | Reproduce results from committed OOF predictions (fast path) or from Zenodo weights (full path) |
| `predictions/` | Out-of-fold predictions (`oof_predictions.csv`), used by `reproduce_results.py` |
| `results/results.md` | Training configuration and full evaluation metrics (matches the paper) |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (CV summary, verification) |

## Models

| Model | Backbone | Pretrained Weights |
|-------|----------|--------------------|
| DiT-Base | microsoft/dit-base | IIT-CDIP |
| DiT-Large | microsoft/dit-large | IIT-CDIP |
| DiT-Base (RVL-CDIP) | microsoft/dit-base-finetuned-rvlcdip | RVL-CDIP fine-tuned |
| DiT-Large (RVL-CDIP) | microsoft/dit-large-finetuned-rvlcdip | RVL-CDIP fine-tuned |

## Running

```bash
# Fast path — recompute metrics from committed OOF predictions (no GPU needed)
python 07_DiT_CrossVal/reproduce_results.py

# Full path — retrain from scratch (requires GPU + Zenodo weights download)
python 07_DiT_CrossVal/reproduce_results.py --mode full
```

## Key Results

Aligned-pipeline results, 5-fold stratified group CV (mean ± std):

| Configuration | MAE (years) | R² |
|--------------|------------:|---:|
| Best individual model (DiT-Base RVL-CDIP) | 5.01 ± 0.53 | 0.19 ± 0.11 |

Under the unified pipeline the DiT family does not lead in cross-validation: the best CNN
(InceptionV3, 4.67) and best ViT (MobileViT-XXS, 4.69) both outperform the best DiT.
