# Experiment 09 — Document Image Transformers (DiT): Results

> **Role in paper:** Table 3 (bottom section) and Table 4 — DiT individual models and
> ensemble results on the official HHD split.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Original HHD |
| Input size | 224×224 (via BeitImageProcessor) |
| Label scaling | No |
| Augmentation | Rotation (±15°), brightness/contrast, Gaussian noise |
| Batch size | 4 |
| Training epochs | 15 (frozen backbone) |
| Training LR | 1e-4 |
| Fine-tune epochs | 30 |
| Fine-tune LR | 1e-5 |
| Optimizer | AdamW |
| Pretrained weights | IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants) |
| Framework | PyTorch + HuggingFace Transformers |

---

## Results — Original HHD (Official Split)

| Model | MAE | RMSE | R² | MAPE (%) | ±2 yrs (%) | ±5 yrs (%) | ±10 yrs (%) | Max Error | Median Error |
|-------|----:|-----:|---:|---------:|-----------:|-----------:|------------:|----------:|-------------:|
| **DiT-Base (RVL-CDIP)** | **2.345** | 4.432 | **0.509** | 13.229 | 64.655 | 91.379 | 97.414 | 31.618 | 1.465 |
| DiT-Base | 2.784 | 5.063 | 0.359 | 14.960 | 58.621 | 89.655 | 95.690 | 32.343 | 1.681 |
| DiT-Large (RVL-CDIP) | 3.124 | 5.141 | 0.339 | 17.324 | 51.724 | 86.207 | 94.828 | 33.642 | 1.919 |
| DiT-Large | 3.379 | 5.653 | 0.201 | 18.330 | 45.690 | 86.207 | 94.828 | 32.807 | 2.288 |

> **Key finding:** DiT-Base (RVL-CDIP) achieves the best individual-model performance
> (MAE = 2.35, R² = 0.51), substantially outperforming all CNN and ViT variants.
> Document-domain pretraining (RVL-CDIP) consistently improves over IIT-CDIP pretraining alone.
