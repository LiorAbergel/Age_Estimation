# 04_ViT_Ensemble — ViT Ensemble

> **Paper reference:** Table 3 (middle section) — ViT ensembles on the official
> HHD split. Its committed predictions are also reused by the hybrid-ensemble
> analysis (Table 4) and by the significance tests in `08_Significance/`.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_vit.py` | Training pipeline — trains 4 ViT backbones and evaluates ensemble configurations |
| `reproduce_results.py` | Reproduce results from committed predictions (fast path) or from Zenodo weights (full path) |
| `reproduce_hybrid.py` | Reproduce the hybrid InceptionV3 + MobileViT-XXS ensemble (paper Table 4) from committed predictions |
| `predictions/` | Per-image predictions (test & val) for each model, used by `reproduce_results.py` |
| `results/results.md` | Training configuration and full evaluation metrics |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (ensemble metrics, individual metrics, verification) |

## Models

| Model | Input Size | Pretrained Weights |
|-------|-----------|--------------------|
| SwinV2-Tiny | 256×256 | ImageNet-1K |
| MobileViT-XXS | 256×256 | ImageNet-1K |
| ConvNeXtV2-Tiny | 224×224 | ImageNet-1K |
| TinyViT-11M | 224×224 | ImageNet-1K |

Models operate on 400×400 patches (stride 200) resized to each backbone's native
resolution. Framework: TensorFlow / `keras_cv_attention_models` (sets
`TF_USE_LEGACY_KERAS=1` internally).

## Running

```bash
# Fast path — recompute metrics from committed predictions (no GPU needed)
python 04_ViT_Ensemble/reproduce_results.py

# Full path — regenerate predictions from model weights (requires GPU + Zenodo weights)
python 04_ViT_Ensemble/reproduce_results.py --mode full

# Hybrid CNN+ViT ensemble (paper Table 4): InceptionV3 + MobileViT-XXS mean
python 04_ViT_Ensemble/reproduce_hybrid.py
```

### Predictions file format

`predictions/test_image_predictions.csv` has one row per test image:

| Column | Meaning |
|--------|---------|
| `ImageID` | Test-image filename (matches `File` in `data/NewAgeSplit.csv`) |
| `SwinV2_Tiny` | Image-level predicted age (mean over patches) |
| `MobileViT_XXS` | Image-level predicted age |
| `ConvNeXtV2_Tiny` | Image-level predicted age |
| `TinyViT_11M` | Image-level predicted age |

Ensemble configurations (grid search and MAE-based weighting) are computed by
`reproduce_results.py`; true ages are joined from `data/NewAgeSplit.csv`.

## Key Results

| Configuration | Best MAE (years) |
|--------------|----------------:|
| Best individual model (MobileViT-XXS) | 2.79 |
| Best ensemble (Best 3, Grid Search) | 2.77 |
</content>
