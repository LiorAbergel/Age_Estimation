# 03_CNN_CrossVal — CNN Cross-Validation

> **Paper reference:** Table 2 (top section) — CNN individual models, 5-fold stratified group CV.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_cnn_cv.py` | Training pipeline — 5-fold stratified group CV with 5 CNN backbones |
| `reproduce_results.py` | Reproduce results from committed OOF predictions (fast path) or from Zenodo weights (full path) |
| `predictions/` | Out-of-fold predictions (`oof_predictions.csv`), used by `reproduce_results.py` |
| `results/` | Results documentation |
| `results/results.md` | New results (after pipeline alignment fix) — training config + full evaluation metrics |
| `results/original_results.md` | Original results (from the paper, before the pipeline fix) |
| `results/results_comparison.md` | Side-by-side comparison of original vs. new results |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (CV summary, ensemble metrics, verification) |

## Models

| Model | Backbone | Pretrained Weights |
|-------|----------|--------------------|
| ResNet50 | ResNet-50 | ImageNet-1K |
| InceptionV3 | Inception-v3 | ImageNet-1K |
| DenseNet121 | DenseNet-121 | ImageNet-1K |
| InceptionResNetV2 | Inception-ResNet-v2 | ImageNet-1K |
| EfficientNetV2M | EfficientNetV2-M | ImageNet-1K |

## Running

```bash
# Fast path — recompute metrics from committed OOF predictions (no GPU needed)
python 03_CNN_CrossVal/reproduce_results.py

# Full path — retrain from scratch (requires GPU + Zenodo weights download)
python 03_CNN_CrossVal/reproduce_results.py --full
```

## Key Results

| Configuration | Best MAE (years) |
|--------------|----------------:|
| Best individual model (InceptionV3) | 4.67 ± 0.48 |
| OOF Ensemble | 4.51 |
