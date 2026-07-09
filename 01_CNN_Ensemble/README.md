# 01_CNN_Ensemble — CNN Ensemble

> **Paper reference:** Table 3 (top section) — CNN ensembles on the official HHD split.

## Contents

| File / Directory | Description |
|------------------|-------------|
| `train_cnn_ensemble.py` | Training pipeline — trains 5 CNN backbones and evaluates ensemble configurations |
| `reproduce_results.py` | Reproduce results from committed predictions (fast path) or from Zenodo weights (full path) |
| `predictions/` | Per-image predictions (test & val) for each model, used by `reproduce_results.py` |
| `results/results.md` | Training configuration and full evaluation metrics (matches the paper) |
| `reproduction_output/` | Generated CSVs from `reproduce_results.py` (ensemble metrics, individual metrics, verification) |

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
# Fast path — recompute metrics from committed predictions (no GPU needed)
python 01_CNN_Ensemble/reproduce_results.py

# Full path — regenerate predictions from model weights (requires GPU + Zenodo weights)
python 01_CNN_Ensemble/reproduce_results.py --mode full
```

## Key Results

| Configuration | Best MAE (years) |
|--------------|----------------:|
| Best individual model (EfficientNetV2M) | 2.77 |
| Best ensemble (Full Ensemble, Grid Search) | 2.73 |
