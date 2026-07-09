"""Paired bootstrap significance tests for the ensemble comparisons in the paper.

Reproduces the paired bootstrap results reported in the "Ensemble Models"
section of the paper, on the official HHD test split (116 pages):

  1. CNN Best4 (Grid)  vs. ResNet50                 (ensemble vs. strongest backbone)
  2. CNN Best4 (Grid)  vs. ViT Best3 (Grid)         (CNN vs. ViT at the ensemble level)
  3. CNN Best4 (Grid)  vs. DiT Best2 (Grid)         (CNN vs. DiT)
  4. ViT Best3 (Grid)  vs. DiT Best2 (Grid)         (ViT vs. DiT)

For each comparison we bootstrap the difference in page-level MAE by resampling
the 116 test pages with replacement (paired: the same resampled pages are used
for both systems), report a 95 percentile confidence interval and a two-sided
bootstrap p-value, and apply a Holm--Bonferroni correction across the family of
tests.

Method notes (kept identical to how the numbers were produced for the paper):

* Ensembles are reconstructed exactly as in the experiment folders: the member
  models and their weights are selected on the **validation** split (grid search
  over weights in {0.1, ..., 0.9} summing to 1, minimizing validation MAE) and
  then evaluated **once** on the held-out test split. This avoids test-set
  selection bias.
      - CNN Best4 (Grid): ResNet50 + InceptionResNetV2 + DenseNet121 + InceptionV3
      - ViT Best3 (Grid): MobileViT-XXS + ConvNeXtV2-Tiny + TinyViT-11M
      - DiT Best2 (Grid): the two DiT variants with the lowest validation MAE
        (DiT-Base (RVL-CDIP) + DiT-Large (RVL-CDIP))
* Predictions are the committed per-image CSVs under each experiment's
  ``predictions/`` folder; no GPU, model weights, or training is required.
* ``--resamples 10000`` and ``--seed 42`` reproduce the paper's numbers.

Usage (run from the repository root):
    python 08_Significance/paired_bootstrap.py
    python 08_Significance/paired_bootstrap.py --resamples 10000 --seed 42
    python 08_Significance/paired_bootstrap.py --output-dir 08_Significance/output
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DEFAULT_RESAMPLES = 10000
GRID_STEP = 0.1

CNN_BEST4 = ["ResNet50", "InceptionResNetV2", "DenseNet121", "InceptionV3"]
VIT_BEST3 = ["MobileViT_XXS", "ConvNeXtV2_Tiny", "TinyViT_11M"]
DIT_MODELS = ["DiT-Base", "DiT-Large", "DiT-Base (RVL-CDIP)", "DiT-Large (RVL-CDIP)"]

# Comparisons reported in the paper, as (name_A, name_B). The reported quantity is
# delta = MAE(A) - MAE(B); a positive delta means B is the better (lower-MAE) system.
COMPARISONS = [
    ("ResNet50", "CNN_Best4_Grid"),
    ("CNN_Best4_Grid", "ViT_Best3_Grid"),
    ("DiT_Best2_Grid", "CNN_Best4_Grid"),
    ("DiT_Best2_Grid", "ViT_Best3_Grid"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn-test", type=Path, default=root / "01_CNN_Ensemble" / "predictions" / "test_image_predictions.csv")
    p.add_argument("--cnn-val", type=Path, default=root / "01_CNN_Ensemble" / "predictions" / "val_image_predictions.csv")
    p.add_argument("--vit-test", type=Path, default=root / "04_ViT_Ensemble" / "predictions" / "test_image_predictions.csv")
    p.add_argument("--vit-val", type=Path, default=root / "04_ViT_Ensemble" / "predictions" / "val_image_predictions.csv")
    p.add_argument("--dit-test", type=Path, default=root / "06_DiT_Ensemble" / "predictions" / "test_image_predictions.csv")
    p.add_argument("--dit-val", type=Path, default=root / "06_DiT_Ensemble" / "predictions" / "val_image_predictions.csv")
    p.add_argument("--output-dir", type=Path, default=root / "08_Significance" / "output")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, help="Bootstrap resamples (default: 10000).")
    p.add_argument("--seed", type=int, default=SEED, help="RNG seed (default: 42).")
    p.add_argument("--alpha", type=float, default=0.05, help="Family-wise significance level (default: 0.05).")
    return p.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing prediction file: {path}")
    df = pd.read_csv(path)
    for col in ("ImageID", "TrueAge"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: {path} is missing required column '{col}'")
    df["ImageID"] = df["ImageID"].astype(str)
    return df


def weighted_pred(df: pd.DataFrame, models: list[str], weights: dict[str, float]) -> pd.Series:
    total = np.zeros(len(df), dtype=float)
    for m in models:
        total += pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float) * weights[m]
    return pd.Series(total / sum(weights.values()), index=df.index)


def grid_search_weights(val_df: pd.DataFrame, models: list[str], step: float = GRID_STEP) -> dict[str, float]:
    """Grid-search ensemble weights on the validation split, minimizing MAE."""
    candidates = np.round(np.arange(step, 1.0, step), 8)
    y_true = pd.to_numeric(val_df["TrueAge"], errors="coerce").to_numpy(dtype=float)
    best_weights, best_mae = None, np.inf
    for combo in itertools.product(candidates, repeat=len(models)):
        if not np.isclose(sum(combo), 1.0, atol=1e-8):
            continue
        weights = dict(zip(models, (float(c) for c in combo)))
        y_pred = weighted_pred(val_df, models, weights).to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        if mae < best_mae:
            best_mae, best_weights = mae, weights
    if best_weights is None:
        raise SystemExit(f"ERROR: no valid grid weights found for {models}")
    return best_weights


def individual_mae(df: pd.DataFrame, model: str) -> float:
    y_true = pd.to_numeric(df["TrueAge"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[model], errors="coerce").to_numpy(dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def system_frame(image_id: pd.Series, true_age: pd.Series, pred: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "ImageID": image_id.astype(str).to_numpy(),
        "true_age": pd.to_numeric(true_age, errors="coerce").to_numpy(dtype=float),
        "pred_age": pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float),
    }).dropna().sort_values("ImageID").reset_index(drop=True)


def build_systems(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    cnn_test, cnn_val = load_predictions(args.cnn_test), load_predictions(args.cnn_val)
    vit_test, vit_val = load_predictions(args.vit_test), load_predictions(args.vit_val)
    dit_test, dit_val = load_predictions(args.dit_test), load_predictions(args.dit_val)

    systems: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}

    # ResNet50 individual (strongest single CNN backbone on the official split).
    systems["ResNet50"] = system_frame(cnn_test["ImageID"], cnn_test["TrueAge"], cnn_test["ResNet50"])

    # CNN Best4 (Grid): weights chosen on validation.
    cnn_w = grid_search_weights(cnn_val, CNN_BEST4)
    systems["CNN_Best4_Grid"] = system_frame(cnn_test["ImageID"], cnn_test["TrueAge"], weighted_pred(cnn_test, CNN_BEST4, cnn_w))
    meta["CNN_Best4_Grid"] = {"models": CNN_BEST4, "weights": cnn_w}

    # ViT Best3 (Grid): weights chosen on validation.
    vit_w = grid_search_weights(vit_val, VIT_BEST3)
    systems["ViT_Best3_Grid"] = system_frame(vit_test["ImageID"], vit_test["TrueAge"], weighted_pred(vit_test, VIT_BEST3, vit_w))
    meta["ViT_Best3_Grid"] = {"models": VIT_BEST3, "weights": vit_w}

    # DiT Best2 (Grid): the two DiT variants with lowest validation MAE, weights chosen on validation.
    dit_ranked = sorted(DIT_MODELS, key=lambda m: individual_mae(dit_val, m))
    dit_group = dit_ranked[:2]
    dit_w = grid_search_weights(dit_val, dit_group)
    systems["DiT_Best2_Grid"] = system_frame(dit_test["ImageID"], dit_test["TrueAge"], weighted_pred(dit_test, dit_group, dit_w))
    meta["DiT_Best2_Grid"] = {"models": dit_group, "weights": dit_w}

    return systems, meta


def aligned_abs_errors(sys_a: pd.DataFrame, sys_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    merged = sys_a.merge(sys_b, on=["ImageID", "true_age"], how="inner", suffixes=("_a", "_b")).sort_values("ImageID")
    if merged.empty:
        raise SystemExit("ERROR: two systems share no aligned test pages")
    err_a = np.abs(merged["true_age"].to_numpy(dtype=float) - merged["pred_age_a"].to_numpy(dtype=float))
    err_b = np.abs(merged["true_age"].to_numpy(dtype=float) - merged["pred_age_b"].to_numpy(dtype=float))
    return err_a, err_b


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, resamples: int, seed: int) -> dict:
    """Paired bootstrap of delta = MAE(A) - MAE(B) over resampled test pages."""
    rng = np.random.default_rng(seed)
    n = err_a.size
    diffs = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = err_a[idx].mean() - err_b[idx].mean()
    point = float(err_a.mean() - err_b.mean())
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    # Two-sided bootstrap p-value: proportion of resamples on the opposite side of 0.
    p = min(1.0, 2.0 * min(float((diffs >= 0).mean()), float((diffs <= 0).mean())))
    return {"delta_mae": point, "ci_low": lo, "ci_high": hi, "p_value": p, "n_pages": int(n)}


def holm_bonferroni(p_values: list[float], alpha: float) -> list[bool]:
    """Return per-test significance under Holm--Bonferroni at family-wise level alpha."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[i] <= threshold:
            significant[i] = True
        else:
            break  # once a test fails, all higher p-values also fail
    return significant


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems, meta = build_systems(args)

    print(f"Paired bootstrap: {args.resamples} resamples, seed {args.seed}\n")
    print("Reconstructed systems (weights selected on validation, evaluated on test):")
    for name in ("CNN_Best4_Grid", "ViT_Best3_Grid", "DiT_Best2_Grid"):
        info = meta[name]
        weights = ", ".join(f"{m}={info['weights'][m]:.2f}" for m in info["models"])
        test_mae = float(np.abs(systems[name]["true_age"] - systems[name]["pred_age"]).mean())
        print(f"  {name:16s} test MAE {test_mae:5.3f}  [{weights}]")
    print(f"  {'ResNet50':16s} test MAE {float(np.abs(systems['ResNet50']['true_age'] - systems['ResNet50']['pred_age']).mean()):5.3f}\n")

    rows = []
    for name_a, name_b in COMPARISONS:
        err_a, err_b = aligned_abs_errors(systems[name_a], systems[name_b])
        res = paired_bootstrap(err_a, err_b, args.resamples, args.seed)
        rows.append({"system_A": name_a, "system_B": name_b, **res})

    p_values = [r["p_value"] for r in rows]
    holm = holm_bonferroni(p_values, args.alpha)
    for r, sig in zip(rows, holm):
        r["significant_holm"] = bool(sig)

    result_df = pd.DataFrame(rows, columns=[
        "system_A", "system_B", "delta_mae", "ci_low", "ci_high", "p_value", "significant_holm", "n_pages",
    ])
    out_csv = args.output_dir / "paired_bootstrap.csv"
    result_df.to_csv(out_csv, index=False)

    print("Paired bootstrap of delta = MAE(A) - MAE(B)  (positive => B is better):")
    print("-" * 92)
    print(f"{'A':16s} {'B':16s} {'dMAE':>7s} {'95% CI':>18s} {'p':>8s}  Holm-sig")
    print("-" * 92)
    for r in rows:
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        print(f"{r['system_A']:16s} {r['system_B']:16s} {r['delta_mae']:+7.3f} {ci:>18s} {r['p_value']:8.4f}  {'yes' if r['significant_holm'] else 'no'}")
    print("-" * 92)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
