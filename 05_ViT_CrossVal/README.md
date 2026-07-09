# Experiment 05 — ViT Cross-Validation

> **Role in paper:** Table 2 (middle section) — ViT individual models, 5-fold stratified group CV.

## Overview

This experiment evaluates four Vision Transformer (ViT) backbones using 5-fold
stratified group cross-validation on the original HHD dataset. It is the ViT
counterpart of Experiment 03 (CNN Cross-Validation).

**Key correction:** The original Colab notebook used batch size 64 and had a
backbone-freezing bug (`model.layers[1].trainable = False` only froze one layer).
The repo script fixes both issues — batch size is aligned to 128 (matching all
CNN/ViT experiments) and the freeze logic correctly freezes all backbone layers
(`model.layers[:-2]`).

## Contents

| Path | Description |
|------|-------------|
| `train_vit_cv.py` | Training script — trains 4 ViT models x 5 folds with StratifiedGroupKFold |
| `reproduce_results.py` | Reproduction script — recomputes all metrics from committed OOF predictions (fast) or from Zenodo weights (full) |
| `predictions/oof_predictions.csv` | Committed out-of-fold predictions (one row per model/fold/image) |
| `results/results.md` | Training configuration and full evaluation metrics (matches the paper) |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (gitignored) |

## Models

| Model | Input Size | Architecture |
|-------|-----------|--------------|
| SwinV2-Tiny | 256x256 | `keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2Tiny_window8` |
| MobileViT-XXS | 256x256 | `keras_cv_attention_models.mobilevit.MobileViT_XXS` |
| ConvNeXtV2-Tiny | 224x224 | `keras_cv_attention_models.convnext.ConvNeXtTiny` |
| TinyViT-11M | 224x224 | `keras_cv_attention_models.tinyvit.TinyViT_11M` |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 400x400 patches, stride 200 (resized to model-specific resolution) |
| Batch size | 128 |
| Frozen epochs | 50 @ LR 1e-3 |
| Fine-tune epochs | 10 @ LR 1e-4 |
| CV strategy | StratifiedGroupKFold, k=5 (stratify=AgeGroup, group=WriterNumber) |
| Pretrained weights | ImageNet-1K |
| Framework | TensorFlow / `keras_cv_attention_models` (`TF_USE_LEGACY_KERAS=1`) |

## Running

### Train from scratch (requires GPU)

```bash
python 05_ViT_CrossVal/train_vit_cv.py
```

### Reproduce results (no GPU needed)

```bash
python 05_ViT_CrossVal/reproduce_results.py                # fast: from committed predictions
python 05_ViT_CrossVal/reproduce_results.py --mode full    # full: from Zenodo weights
```

The fast mode recomputes all per-fold and mean+/-std metrics from the committed
OOF predictions in `predictions/oof_predictions.csv`. No GPU, model weights, or
TensorFlow is required. The full mode downloads fine-tuned checkpoints from
Zenodo and reruns the OOF inference pipeline.

Outputs are written to `reproduction_output/`:
- `cv_metrics_summary.csv` — per-model mean+/-std for all metrics
- `ensemble_metrics.csv` — OOF ensemble (simple average of all models)
- `verification.csv` — PASS/FAIL check against `results/results.md`

## Key Results

Best individual model: **MobileViT-XXS** (MAE 4.69 +/- 0.22), matching the paper
(Table 2, middle section). See `results/results.md` for all four models and the
full metric set.
