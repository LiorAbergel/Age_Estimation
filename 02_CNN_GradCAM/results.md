# Experiment — Grad-CAM Visualizations: Results

> **Role in paper:** Figure 5, Section "Visual Interpretation" — visual interpretation via Grad-CAM.

---

## Overview

This experiment applies Gradient-weighted Class Activation Mapping (Grad-CAM) to
**InceptionV3**, the best-performing individual CNN in cross-validation
(Experiment 03), to visualize which regions of a handwriting patch the model
attends to when predicting age. The predicted scalar age is used as the target
for gradient computation.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | InceptionV3 (trained in Experiment 01) |
| Target layer | `mixed10` (final inception block) |
| Gradient target | Predicted scalar age |
| Preprocessing | Normalization to [0, 1] (matches training) |
| Input | 400×400 patches from the test set |

Run from the repository root:

```bash
python 02_CNN_GradCAM/visualize_gradcam.py            # InceptionV3, mixed10 (paper default)
python 02_CNN_GradCAM/visualize_gradcam.py --model all  # all CNN backbones, for comparison
```

---

## Key Findings

- The model concentrates its attention on coherent stroke patterns rather than
  irrelevant background noise.
- Attention highlights meaningful visual features such as stroke continuity and
  pressure, confirming the model learns age-relevant handwriting cues.
- Both correctly and incorrectly predicted samples show consistent attention to
  handwriting strokes.
- Visualizations are useful for identifying failure cases and guiding data
  quality improvements.

## Paper Reference

**Figure 5** shows two examples for InceptionV3, each with the original input
patch, the Grad-CAM heatmap, and the overlay:

- **Top (small error):** writer aged 18, patch prediction 17.9.
- **Bottom (large error):** writer aged 51, patch prediction 17.0.
</content>
</invoke>
