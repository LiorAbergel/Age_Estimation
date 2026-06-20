# Experiment — Grad-CAM Visualizations: Results

> **Role in paper:** Figure 5, Section 4.3 — Visual interpretation via Grad-CAM.

---

## Overview

This experiment applies Gradient-weighted Class Activation Mapping (Grad-CAM) to the best-performing CNN model from Experiment 03 (ResNet50) to visualize which regions of handwriting patches the model attends to when predicting age.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ResNet50 (from Experiment 03) |
| Target layer | `conv5_block3_3_conv` (last convolutional layer) |
| Gradient target | Predicted scalar age |
| Input | 400×400 patches from test set |

---

## Key Findings

- The model focuses on coherent stroke patterns rather than irrelevant background noise.
- Attention highlights meaningful visual features such as stroke continuity and pressure.
- Both correctly and incorrectly predicted samples show consistent attention to handwriting strokes, confirming the model learns relevant visual patterns.
- Visualizations are useful for identifying failure cases and guiding data quality improvements.

## Paper Reference

- **Figure 5** shows two examples: a correctly predicted patch (writer aged 20, predicted 20.1) and an incorrectly predicted patch (writer aged 51, predicted 20.7), each with original input, Grad-CAM heatmap, and overlay.
