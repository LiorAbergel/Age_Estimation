# Pipeline Comparison: Original Notebooks vs. Current Implementation

## Why we changed anything

The paper describes **one** training protocol shared by every model family. In practice
each experiment was first written as its own Colab notebook (`v2.5`, `v4`, `v5`, `v6`,
`v8`, `v9`), and they drifted apart over time. Two kinds of problems resulted:

1. **Inconsistencies** — batch size, callbacks, augmentation, and freezing differed
   between notebooks, so a "CNN vs. ViT vs. DiT" comparison was not apples-to-apples.
2. **Correctness / leakage bugs** — a few notebooks did something that *contradicts the
   paper* (e.g. tuning ensemble weights on the test set, a backbone that never loaded, a
   "frozen" stage that wasn't frozen, the wrong loss function).

We therefore re-implemented every experiment against a **single unified protocol** and
fixed the bugs. This document lists what changed per experiment and why it mattered.

Folder ↔ original notebook mapping:

| Exp | Folder | Original notebook |
|----|--------|-------------------|
| 1 | `01_CNN_Ensemble` | `v2.5` |
| 2 | `02_CNN_GradCAM` | *(unchanged — visualization only)* |
| 3 | `03_CNN_CrossVal` | `v4` |
| 4 | `04_ViT_Ensemble` | `v5` |
| 5 | `05_ViT_CrossVal` | `v6` |
| 6 | `06_DiT` | `v8` |
| 7 | `07_DiT_CrossVal` | `v9` |

---

## The unified protocol (the target everything was aligned to)

| Stage | Setting |
|-------|---------|
| Preprocess | Resize page to 800 px (keep aspect ratio) → 400×400 patches, stride 200 → pixels /255 |
| Empty-patch filter | Drop near-blank patches (mean intensity ≤ 0.0054) |
| Augmentation | Rotation ±15°, zoom 0.9–1.1, brightness, contrast, Gaussian noise (σ=0.05) |
| Head | GlobalAveragePool → Dropout 0.5 → Dense(1, linear)  *(DiT: global mean-pool over patch tokens → Dropout 0.5 → Dense)* |
| Loss / optimizer | **MSE**, **Adam** |
| Schedule | 50 epochs frozen backbone + 10 epochs fine-tuning |
| Callbacks | Checkpoint on `val_mae`, `ReduceLROnPlateau(factor=0.1, patience=5)`, `EarlyStopping(patience=10, restore_best)`, **carry-forward** (Phase 2 keeps the best Phase 1 weights if it doesn't improve) |
| Image score | Mean of its patch predictions |
| Ensemble | Weights selected on **validation**, evaluated **once** on **test** (leakage-free) |
| Reproducibility | Global seed = 42 |

**DiT-specific (architecture-required, not deviations):** PyTorch + HuggingFace, global
mean-pooling over patch tokens (the BEiT CLS token is not pretrained for downstream
tasks), `BeitImageProcessor` normalization, FP16, gradient accumulation to reach the same
effective patch batch, and empty-patch filtering.

---

## Experiment 1 — CNN Ensemble (`01_CNN_Ensemble`, vs `v2.5`)

| Aspect | Original (`v2.5`) | Current | Why it mattered |
|--------|-------------------|---------|-----------------|
| EfficientNetV2M | `load_model(EfficientNetV2M)` — loads a *class*, not a built model | Built with `build_sota_model` like every other backbone | **Bug.** The model was never trained correctly (paper MAE 7.17). Fixed → **MAE 2.77**, now the best CNN. |
| Ensemble weights | Grid search run **on the test set** | Grid search **on validation**, evaluated on test | **Test-set leakage** — contradicts the paper's protocol. |
| Empty-patch filter | None | Added (THR 0.0054) | Consistency with all other experiments. |
| Augmentation | brightness only | brightness **+ contrast** | Match unified augmentation. |
| Reproducibility | unseeded | seed 42 | Deterministic runs. |
| Checkpoint monitor | `val_loss` | `val_mae` | Consistency (selects on the reported metric). |

## Experiment 3 — CNN Cross-Validation (`03_CNN_CrossVal`, vs `v4`)

| Aspect | Original (`v4`) | Current | Why it mattered |
|--------|-----------------|---------|-----------------|
| Batch size | 64 | 128 | Match Exp 1 / unified protocol. |
| `ReduceLROnPlateau` | factor 0.2, patience 4 | factor 0.1, patience 5 | Unify LR schedule. |
| `EarlyStopping` | patience 8 | patience 10 | Unify stopping rule. |
| Augmentation | contrast only | brightness **+ contrast** | Match unified augmentation. |
| Reproducibility | only the fold split was seeded | global seed 42 (numpy + TF) | Deterministic training, not just splitting. |
| Outputs | aggregate metrics | per-image OOF predictions exported (`oof_predictions.csv`) + `.done` crash-safe markers | Enables fast, exact reproduction. |

*(Unchanged and already correct: StratifiedGroupKFold k=5 grouped by writer, empty-patch
filter, whole-backbone freeze, Adam 1e-3 → 1e-4, 50+10 epochs.)*

## Experiment 4 — ViT Ensemble (`04_ViT_Ensemble`, vs `v5`)

> Development experiment (not reported in the paper), but aligned for consistency.

| Aspect | Original (`v5`) | Current | Why it mattered |
|--------|-----------------|---------|-----------------|
| Frozen stage | froze `layers[1]` only | freeze `layers[:-2]` | **Bug.** The ViT backbone is *inlined*, so `layers[1]` is a single stem layer — the "frozen" warm-up actually trained almost the whole backbone. `layers[:-2]` truly freezes the backbone and trains only the head. |
| Empty-patch filter | None | Added (THR 0.0054) | Consistency. |
| Augmentation | brightness only | brightness **+ contrast** | Match unified augmentation. |
| Inference resize | bilinear (training used bicubic) | bicubic everywhere | Train/inference mismatch removed. |
| Reproducibility | unseeded | seed 42 | Deterministic runs. |

*(Unchanged: batch 128, Adam 1e-3 → 1e-4, MSE, 50+10 epochs, `ReduceLROnPlateau(0.1,5)` +
`EarlyStopping(10)` — `v5` already matched these.)*

## Experiment 5 — ViT Cross-Validation (`05_ViT_CrossVal`, vs `v6`)

| Aspect | Original (`v6`) | Current | Why it mattered |
|--------|-----------------|---------|-----------------|
| Frozen stage | froze `layers[1]` only | freeze `layers[:-2]` | Same backbone-freeze bug as Exp 4 — fixed. |
| Batch size | 64 | 128 | Match the rest of the pipeline. |
| `ReduceLROnPlateau` | factor 0.2, patience 4 | factor 0.1, patience 5 | Unify LR schedule. |
| `EarlyStopping` | patience 8 | patience 10 | Unify stopping rule. |
| Augmentation | contrast only | brightness **+ contrast** | Match unified augmentation. |
| Inference resize | bilinear (training used bicubic) | bicubic everywhere | Train/inference mismatch removed. |
| Reproducibility | only the fold split was seeded | global seed 42 | Deterministic training. |
| Outputs | aggregate metrics | per-image OOF export + `.done` markers + carry-forward | Reproducibility + correct model selection. |

## Experiment 6 — DiT, official split (`06_DiT`, vs `v8`)

| Aspect | Original (`v8`) | Current | Why it mattered |
|--------|-----------------|---------|-----------------|
| Loss | **L1 (MAE)** | **MSE** | The paper/protocol optimizes MSE; L1 changes the optimization entirely. |
| Dropout | 0.1 | 0.5 | 5× less regularization than every other experiment. |
| Optimizer | `AdamW` + weight decay | `Adam` (no weight decay) | Match the reference optimizer. |
| Epochs | 15 + 30 | 50 + 10 | Match the protocol's frozen→fine-tune schedule. |
| LR schedule / early stop | none | `ReduceLROnPlateau(0.1,5)` + `EarlyStopping(10)` | Same model-selection rule as everything else. |
| Augmentation | `ColorJitter(contrast=0.1)`, `RandomResizedCrop` with default ratio | contrast 0.25, `ratio=(1.0,1.0)` (no aspect distortion) | Match augmentation strength and avoid silently warping patches. |
| Effective batch | grad-accumulation defined but **never applied** (≈ a couple of images of patches) | true patch-level accumulation → **128 patches/step** | Made the effective batch well-defined and comparable. |
| Phase-2 model selection | if Phase 2 never beat Phase 1, the final checkpoint was missing | carry-forward: copy best Phase 1 → final | **Bug.** Could crash or keep the wrong weights. |
| Ensemble weights | selected on test | selected on validation, evaluated on test | Remove test-set leakage. |

*(Unchanged, architecture-required: PyTorch, global mean-pooling, Beit normalization, FP16,
empty-patch filter.)*

## Experiment 7 — DiT Cross-Validation (`07_DiT_CrossVal`, vs `v9`)

All Exp 6 fixes apply identically (loss, dropout, optimizer, epochs, LR schedule,
augmentation, carry-forward), plus the CV-specific items:

| Aspect | Original (`v9`) | Current | Why it mattered |
|--------|-----------------|---------|-----------------|
| Effective batch | image batch 1 × accum 4 ≈ **4 images** | **128 patches/step** | `v9` was *not comparable* to `06_DiT`; now both share the same effective batch. |
| OOF predictions | not collected | per-image OOF records exported (matches Exp 3 / 5) | Required for the paper's CV table + reproduction. |
| Results saving | aggregate only | `oof_predictions.csv`, per-fold and mean±std summaries | Reproducibility. |

---

## The fixes that actually move numbers

- **Exp 1 — EfficientNetV2M never built:** MAE **7.17 → 2.77** (became the best individual CNN).
- **Leakage (Exp 1 & 6 ensembles):** weights were tuned on the **test** set → now tuned on
  **validation**. This is the change most likely to be questioned in review.
- **ViT "frozen" stage wasn't frozen (Exp 4 & 5):** `layers[1]` → `layers[:-2]`.
- **DiT loss & regularization (Exp 6 & 7):** L1 → MSE, dropout 0.1 → 0.5, AdamW → Adam,
  15+30 → 50+10 epochs, plus a real LR schedule, a well-defined effective batch, and a
  fixed best-checkpoint carry-forward.

All result tables were regenerated under the corrected pipeline (see each folder's
`results.md`; Exp 1 deltas are in `results_comparison.md`), except `06_DiT` and
`07_DiT_CrossVal` which still need to be re-run under the aligned 1e-3/1e-4 LR.
