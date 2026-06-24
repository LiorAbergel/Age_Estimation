# DiT Pipeline Alignment — Implementation Plan

## Background

Experiments 6 (DiT single-split) and 7 (DiT cross-validation) use **PyTorch + HuggingFace Transformers** — a different framework from the TF/Keras reference experiments (1, 3, 4, 5). This is architecturally required since DiT models (`microsoft/dit-*`) are HuggingFace PyTorch models.

While framework differences are expected, the **training pipeline logic** (loss, optimizer, hyperparameters, augmentation, model selection, output format) must be equivalent. The analysis found **11 CRITICAL** and **5 MODERATE** misalignments.

---

## User Review Required

> [!IMPORTANT]
> **Optimizer**: Exp 1–5 (TF/Keras) all use `Adam`. Exp 6/7 use PyTorch `AdamW` (with weight decay). AdamW is the modern standard for transformers, but switching to plain `Adam` would better match the reference experiments. **My recommendation**: use `torch.optim.Adam` (no weight decay) to match. Please confirm.

> [!IMPORTANT]
> **Epoch counts**: Exp 6/7 use 15+30 (Stage 1+2) vs 50+10 in all other experiments. Changing to 50+10 aligns the pipeline but significantly increases Stage 1 training time. DiT models are larger and slower per epoch. **My recommendation**: align to 50+10 for consistency. Please confirm.

> [!WARNING]
> **Phase 2 carry-forward bug**: In both Exp 6 and 7, if Phase 2 never beats Phase 1, the `final_best.pt` checkpoint is never created. The code then uses the **last-epoch model** (worst!) instead of the best Phase 1 model. This is a correctness bug that must be fixed.

---

## Proposed Changes

### Exp 6 — [train_dit.py](file:///c:/Users/liora/OneDrive/Documents/Important/School/Bachelors/Age%20Estimation%20Project/Age_Estimation/06_DiT/train_dit.py)

---

#### Fix 1 — Loss function: L1 → MSE `[CRITICAL]`
All reference experiments use MSE loss. Exp 6 uses `nn.functional.l1_loss` (MAE). This fundamentally changes the optimization landscape.

```diff
-            out["loss"] = nn.functional.l1_loss(pred, labels)
+            out["loss"] = nn.functional.mse_loss(pred, labels)
```

#### Fix 2 — Dropout: 0.1 → 0.5 `[CRITICAL]`
All reference experiments use Dropout(0.5). Exp 6 uses 0.1 (5× less regularization).

```diff
-    class DiTReg(nn.Module):
-        def __init__(self, name="microsoft/dit-base", p=0.1):
+    class DiTReg(nn.Module):
+        def __init__(self, name="microsoft/dit-base", p=0.5):
```

#### Fix 3 — Optimizer & LR alignment `[CRITICAL]`
Reference experiments use `Adam(1e-3)` for Phase 1 and `Adam(1e-4)` for Phase 2.

```diff
 # Phase 1:
-optimizer = torch.optim.AdamW(model.head.parameters(), lr=CONFIG["LR_BASE"], weight_decay=CONFIG["WEIGHT_DECAY_STAGE1"])
+optimizer = torch.optim.Adam(model.head.parameters(), lr=CONFIG["LR_INIT"])

 # Phase 2:
-optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_BASE"] / 10, weight_decay=CONFIG["WEIGHT_DECAY_STAGE2"])
+optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR_FT"])
```

Config changes:
```diff
-    "EPOCHS_STAGE1": 15,
-    "EPOCHS_STAGE2": 30,
-    "LR_BASE": 1e-4,
-    "WEIGHT_DECAY_STAGE1": 1e-4,
-    "WEIGHT_DECAY_STAGE2": 1e-5,
+    "EPOCHS_PHASE1": 50,
+    "EPOCHS_PHASE2": 10,
+    "LR_INIT": 1e-3,
+    "LR_FT": 1e-4,
```

#### Fix 4 — Contrast augmentation range `[CRITICAL]`
Reference: `contrast in [0.75, 1.25]`. Exp 6: `contrast=0.1` → `[0.9, 1.1]` (much narrower).

```diff
-ColorJitter(brightness=0.1, contrast=0.1),
+ColorJitter(brightness=0.1, contrast=0.25),
```

#### Fix 5 — Add ReduceLROnPlateau + EarlyStopping `[CRITICAL]`
Reference experiments use `ReduceLROnPlateau(patience=5, factor=0.1)` and `EarlyStopping(patience=10)`.

Add `torch.optim.lr_scheduler.ReduceLROnPlateau` and manual early-stopping logic to both training phases.

#### Fix 6 — Fix Phase 2 carry-forward bug `[CRITICAL]`
If Phase 2 never improves on Phase 1, the code must fall back to `stage1_ckpt`, not the last-epoch model.

```diff
+            # If Phase 2 never improved, copy stage1 as final
+            if not final_ckpt.exists():
+                shutil.copy2(stage1_ckpt, final_ckpt)
+
             if final_ckpt.exists():
                 model.load_state_dict(torch.load(final_ckpt, map_location=device))
```

#### Fix 7 — Metric key alignment + Min_Error `[MODERATE]`
Align with `Acc_2yr`/`Acc_5yr`/`Acc_10yr` naming and add `Min_Error`.

```diff
-    "MAPE (%)": mape,
-    "Within ±2 Years (%)": within_2, "Within ±5 Years (%)": within_5,
-    "Within ±10 Years (%)": within_10, "Max Error": max_err, "Median Error": median_err
+    "MAPE": mape,
+    "Acc_2yr": within_2, "Acc_5yr": within_5, "Acc_10yr": within_10,
+    "Max_Error": max_err, "Median_Error": median_err, "Min_Error": float(np.min(errors))
```

#### Fix 8 — Docstring: "Experiment 09" → "Experiment 06" `[LOW]`

#### Fix 9 — Add Colab/Drive support `[MODERATE]`
Port `_in_colab()` and `resolve_output_dirs()` from Exp 4.

#### Fix 10 — Memory management `[MODERATE]`
Free each model after generating predictions instead of keeping all 4 in `trained_models` dict.

#### Fix 11 — Remove dead `json` import, add `shutil` usage `[LOW]`

---

### Exp 7 — [train_dit_cv.py](file:///c:/Users/liora/OneDrive/Documents/Important/School/Bachelors/Age%20Estimation%20Project/Age_Estimation/07_DiT_CrossVal/train_dit_cv.py)

All training-pipeline fixes from Exp 6 (Fixes 1–5) apply identically, plus these CV-specific fixes:

---

#### Fix 12 — Fix Phase 2 carry-forward bug (CV version) `[CRITICAL]`
Same bug as Exp 6 — if Phase 2 never improves, `final_ckpt` doesn't exist. Must fall back to `stage1_ckpt`.

```diff
+                # If Phase 2 never improved, copy stage1 as final
+                if not final_ckpt.exists():
+                    import shutil
+                    shutil.copy2(stage1_ckpt, final_ckpt)
```

#### Fix 13 — Add OOF record collection `[CRITICAL]`
Exp 3/5 collect per-image `{Model, Fold, ImageID, Prediction, TrueAge}` records during `run_cv()`. Exp 7 only saves aggregate metrics. Must add OOF record collection.

#### Fix 14 — Add `save_cv_results()` `[CRITICAL]`
Port from Exp 5: saves `oof_predictions.csv`, `cv_metrics_per_fold.csv`, `cv_metrics_summary.csv`.

#### Fix 15 — Metric key alignment + Min_Error `[MODERATE]`
Same as Fix 7 for Exp 6.

#### Fix 16 — Add Colab/Drive support `[MODERATE]`
Port `_in_colab()` and `resolve_output_dirs()`.

#### Fix 17 — Remove dead imports (`time`, `shutil` if not needed) `[LOW]`

---

## What Stays Unchanged (Architecture-Specific)

| Aspect | Reason |
|--------|--------|
| PyTorch framework | DiT models are HuggingFace PyTorch only |
| CLS token pooling (vs GAP) | Standard for ViT/DiT architectures |
| BeitImageProcessor normalization | Required for pretrained DiT models |
| Mixed precision (FP16) | Memory optimization for large models |
| Gradient accumulation | Memory constraint |
| Batch sizes (16/2 for Exp 6, 1 for Exp 7) | DiT memory requirements |
| `build + load_state_dict` model loading | PyTorch standard |
| `torch.cuda.empty_cache()` (vs `clear_session()`) | Framework equivalent |

---

## Additional Fixes (identified during implementation review)

| # | Fix | Rationale |
|---|-----|-----------|
| 18 | Relative paths → REPO_ROOT-based absolute paths | Scripts broke if run from a directory other than repo root |
| 19 | `RandomResizedCrop` add `ratio=(1.0, 1.0)` | Reference uses uniform zoom (no aspect ratio change); PyTorch default ratio=(0.75, 1.33) was silently modifying aspect ratio |
| 20 | `.done` marker pattern for training completion | Matches Exp 03/05 robustness pattern; prevents incomplete checkpoints from being mistaken as complete |

---

## Verification Plan

### Syntax Verification (PASSED)
- `python -m py_compile 06_DiT/train_dit.py` ✓
- `python -m py_compile 07_DiT_CrossVal/train_dit_cv.py` ✓

### Structural Verification (PASSED)
- Grep for old metric keys (`Within ±`, `MAPE (%)`) → **0 matches in .py files**
- Grep for `l1_loss` → **0 matches in .py files**
- Grep for `AdamW` in DiT .py files → **0 matches**
- Grep for `shutil.copy2` → confirms carry-forward fallback in both scripts
- Grep for `ReduceLROnPlateau` → present in both scripts
- Grep for `save_cv_results` → present in Exp 07, matches Exp 03/05
- Grep for `_in_colab` → present in all experiment scripts
