# DiT Training — Situation Summary & Options

*Prepared for review with instructors. Scope: the Document Image Transformer (DiT) experiments (exp. 06 official split / 07 5-fold CV) within the handwriting age-estimation project.*

---

## 1. The core tension

The accepted paper makes two claims that are in tension:

1. **"DiT is the best-performing architecture"** — the best individual model is DiT-Base (RVL-CDIP), MAE ≈ **2.33** on the official split.
2. **"All experiments were performed under a unified framework to ensure fair comparison across architectures."**

The problem: the DiT numbers in the paper were produced by a training recipe that **differs from the one used for the CNN and ViT experiments**. When we rebuild DiT to run under the *same* pipeline as the CNNs/ViTs (to make claim #2 literally true), its performance drops below the paper's numbers — and below the best CNN.

So we cannot simultaneously (a) reproduce the paper's DiT numbers and (b) use an identical training recipe for all architectures. This document lays out how the original worked, what we tried, and the choices available.

---

## 2. How the original script trained (produced the paper's results)

The paper's DiT numbers came from the standalone script **`Age_Recognition_..._V8_DiT`**. Its recipe:

| Setting | Original DiT script (V8) |
|---|---|
| Loss | **L1 / MAE** (directly optimizes the reported metric) |
| Optimizer | **AdamW** with weight decay |
| Learning rate | **1e-4** (frozen stage) / **1e-5** (fine-tune stage) |
| Pooling head | **CLS token** |
| Dropout | **0.1** |
| Epochs | **15** frozen / **30** fine-tune |
| Model selection | **Page-level** aggregated MAE |
| LR scheduler / early stopping | **None** |
| Result (DiT-Base RVL-CDIP, official split) | **MAE ≈ 2.33** (best individual model in the paper) |

This recipe was tuned specifically for the DiT/transformer backbone and was **not** the same recipe used for the CNN and ViT families.

---

## 3. The aligned pipeline (what we built for a fair comparison)

To honor claim #2, we rebuilt DiT (exp. 06/07) to match the CNN/ViT pipeline exactly:

| Setting | Aligned pipeline (matches CNN & ViT) |
|---|---|
| Loss | MSE |
| Optimizer | Adam (no weight decay) |
| Pooling head | Global mean-pooling (the GAP analog the paper describes) |
| Dropout | 0.5 |
| Epochs | 50 frozen / 10 fine-tune |
| Model selection | Patch-level, best checkpoint across **both** stages, chosen on **validation** |
| LR scheduler / early stopping | ReduceLROnPlateau (×0.1, patience 5) + EarlyStopping (patience 10) |
| Batching | 128 patches per step |

Everything else (data, splits, augmentation, patch extraction, patch→page aggregation, evaluation) is already shared across all experiments.

---

## 4. What we tried — and the results

DiT-Base, official-split test MAE:

| Configuration | Test MAE | Notes |
|---|---:|---|
| **Original V8 recipe** (L1, AdamW, CLS, 1e-4/1e-5, 15/30) | **≈ 2.33** | Paper's number. Different recipe from CNN/ViT. |
| Aligned pipeline, LR **1e-3 / 1e-4**, mean-pool | ≈ **3.16** | Best aligned result; fine-tune unstable then partially recovers. |
| Aligned pipeline, LR **1e-4 / 1e-5**, mean-pool | ≈ **3.90** | Stable but **undertrained** — see §5. |

**For reference, the best CNN (EfficientNetV2M) on the same split ≈ 2.77.**

Bottom line: **no configuration under the aligned pipeline has matched the original recipe.** The aligned DiT lands in the ~3.1–3.9 range.

---

## 5. Why the aligned version underperforms (diagnoses)

1. **Undertraining (confirmed in code).** In the aligned pipeline the fine-tune stage is capped at 10 epochs while EarlyStopping patience is also 10 — so **early stopping can never trigger in the fine-tune stage**; every model runs exactly 10 fine-tune epochs. The DiT's validation MAE was **still decreasing at epoch 10** (5.41 → 5.36 → 5.30). It simply never finished converging. (The same cap exists for CNNs but is harmless there — they converge in a few epochs.)
2. **Loss function.** The original uses **L1**, which directly optimizes MAE; the aligned pipeline uses **MSE**.
3. **Learning-rate sensitivity.** Fine-tuning the transformer at 1e-4 destabilizes the pretrained weights; at 1e-5 it is stable but too slow for a 10-epoch budget.
4. **Recipe differences compound** (optimizer, dropout, pooling, epoch budget) — the original was tuned for the backbone; the aligned settings were tuned for CNNs.

**Takeaway:** the aligned DiT's weakness looks like a *pipeline/training-budget artifact*, not evidence that the DiT architecture is weak.

---

## 6. The critical caveat: statistical noise

The official test set has only **~116 pages**. Bootstrap confidence intervals for MAE are roughly **±1** (e.g. CNN ensemble MAE 2.75, 95% CI ≈ [1.88, 3.83]).

That means the difference between **DiT 2.33**, **EfficientNet 2.77**, and the CNN ensemble **2.75** is **within statistical noise** — the top models are *not* statistically distinguishable. This reframes the whole question: **the paper does not need DiT to "win"; it needs the comparison to be fair and honestly reported with uncertainty.**

---

## 7. Our choices

| | **Choice A — Reproduce the original, disclose** | **Choice B — Strict alignment, fix & retrain** | **Choice C — Keep aligned as-is, report honestly** |
|---|---|---|---|
| **What we do** | Keep DiT on its own tuned recipe (L1/AdamW/CLS/0.1/1e-4-1e-5); report a per-architecture hyperparameter table; reword "unified framework" as *shared protocol with per-architecture optimization*. | Fix the undertraining bug (raise fine-tune epochs so early stopping can act), then **retrain the whole suite** under one identical pipeline; report whatever results emerge. | Keep the current aligned DiT (~3.16); report it as-is, framed as within statistical noise; add the honest finding that feature-extraction/frozen-backbone is competitive in this low-resource setting. |
| **DiT result** | ≈ 2.33 (paper stands) | Unknown; likely competitive-but-tied | ≈ 3.16 |
| **"DiT is best"** | Survives (as best *point estimate*, within CI overlap) | Likely becomes "tied within noise" | Does not survive |
| **Fairness of comparison** | Defensible *if disclosed* — per-architecture tuning is standard practice, but must be stated, not implied identical | Strongest — literally one pipeline | Strong — one pipeline, honest result |
| **Cost / risk** | Low (writing + config disclosure) | **High** — retrain all models (incl. DiT-Large + CV folds); rewrite conclusions | Low |
| **Main drawback** | Slight weakening of the "unified framework" wording | Expensive and risky close to the camera-ready deadline | Contradicts the accepted paper's headline finding |

---

## 8. Recommendation (for discussion)

Given (a) the ~116-page statistical-noise reality and (b) the camera-ready timeline, the pragmatic and defensible path is a **combination of A + honest reporting**:

- **Keep the paper's reported numbers**, but add **bootstrap confidence intervals** so all comparisons are shown as within statistical noise (this is also what the reviewers asked for).
- **Disclose a per-architecture hyperparameter table** and reword "unified framework" to mean the *shared experimental protocol* (data, splits, augmentation, patch→page aggregation, evaluation, validation-based model selection) — **not** identical optimizer/loss/LR. This is standard and honest.
- Treat the **undertraining fix + full re-alignment (Choice B)** as the principled follow-up, but as **future work / post-deadline**, not a rushed suite-wide retrain — optionally probed on DiT-Base only if compute allows.

**One-line summary for the instructors:** *The DiT's paper-best result came from a transformer-specific recipe; forcing it onto the CNN/ViT recipe underperforms — largely because of an epoch-budget/undertraining artifact — but with only ~116 test pages the top models are statistically indistinguishable, so the decision is about how to fairly describe and disclose the setup, not about who truly "wins."*
