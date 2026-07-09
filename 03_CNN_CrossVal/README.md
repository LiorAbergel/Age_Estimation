# 03_CNN_CrossVal — CNN Cross-Validation

> **Paper reference:** Table 2 (top section) — CNN individual models, 5-fold stratified group CV.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_cnn_cv.py` | Training pipeline — 5-fold stratified group CV with 5 CNN backbones |
| `reproduce_results.py` | Reproduce results from committed OOF predictions (fast path) or from Zenodo weights (full path) |
| `predictions/` | Out-of-fold predictions (`oof_predictions.csv`), used by `reproduce_results.py` |
| `results/results.md` | Training configuration and full evaluation metrics (matches the paper) |
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

# Full path — regenerate predictions from model weights (requires GPU + Zenodo weights)
python 03_CNN_CrossVal/reproduce_results.py --mode full
```

## Key Results

| Configuration | Best MAE (years) |
|--------------|----------------:|
| Best individual model (InceptionV3) | 4.67 ± 0.48 |
| OOF Ensemble | 4.51 |
