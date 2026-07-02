# Committed predictions — Experiment 04 (ViT Ensemble)

`reproduce_results.py --mode fast` reads **`test_image_predictions.csv`** from
this folder. It has one row per test image:

| Column | Meaning |
|--------|---------|
| `ImageID` | Test-image filename (matches `File` in `data/NewAgeSplit.csv`) |
| `SwinV2_Tiny` | Image-level predicted age (mean over patches) |
| `MobileViT_XXS` | Image-level predicted age |
| `ConvNeXtV2_Tiny` | Image-level predicted age |
| `TinyViT_11M` | Image-level predicted age |

Ensemble configurations (grid search and MAE-based weighting) are computed
by `reproduce_results.py`; true ages are joined from `data/NewAgeSplit.csv`.

## How to (re)generate

This file is produced by the evaluation block of `../train_vit.py` (saved to
`RESULTS_DIR/test_image_predictions.csv`). Run that script on Colab/GPU, then
copy the resulting CSV here:

```
04_ViT_Ensemble/predictions/test_image_predictions.csv
```

Once present, `reproduce_results.py` runs in fast mode with no GPU, weights or
TensorFlow required. Without it, use `reproduce_results.py --mode full` to
regenerate predictions from the Zenodo-hosted checkpoints.
