# 06_DiT_Ensemble — Document Image Transformer Ensembles

> **Paper reference:** Table 3 (bottom section) — DiT individual models and ensembles on the official HHD split.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_dit.py` | Training pipeline — trains 4 DiT backbones and evaluates ensemble configurations |
| `reproduce_results.py` | Reproduce results from committed predictions (fast path) or from Zenodo weights (full path) |
| `predictions/` | Per-image predictions (test & val) for each model, used by `reproduce_results.py` |
| `results/results.md` | Training configuration and full evaluation metrics (matches the paper) |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (ensemble metrics, individual metrics, verification) |

## Models

| Model | Backbone | Pretrained Weights |
|-------|----------|--------------------|
| DiT-Base | microsoft/dit-base | IIT-CDIP |
| DiT-Large | microsoft/dit-large | IIT-CDIP |
| DiT-Base (RVL-CDIP) | microsoft/dit-base-finetuned-rvlcdip | RVL-CDIP fine-tuned |
| DiT-Large (RVL-CDIP) | microsoft/dit-large-finetuned-rvlcdip | RVL-CDIP fine-tuned |

## Running

```bash
# Fast path — recompute metrics from committed predictions (no GPU needed)
python 06_DiT_Ensemble/reproduce_results.py

# Full path — retrain from scratch (requires GPU + Zenodo weights download)
python 06_DiT_Ensemble/reproduce_results.py --mode full
```

## Key Results

| Configuration | Best MAE (years) |
|--------------|----------------:|
| Best individual model (DiT-Base RVL-CDIP) | 3.295 |
| Best ensemble (Best 2, Grid Search) | 3.175 |
