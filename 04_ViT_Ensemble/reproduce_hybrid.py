"""Reproduce the hybrid CNN+ViT ensemble of the paper (Table 4).

The paper's hybrid ensemble is a simple mean of the best individual CNN
(InceptionV3, from ``01_CNN_Ensemble``) and the best individual ViT
(MobileViT-XXS, from ``04_ViT_Ensemble``) on the official HHD test split. This
script recomputes InceptionV3, MobileViT-XXS, and their mean ensemble directly
from the committed per-image predictions — no GPU, model weights, or training
required — and checks them against the paper's Table 4.

Usage (run from the repository root):
    python 04_ViT_Ensemble/reproduce_hybrid.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Paper Table 4 (MAE, RMSE, R2, MAPE, Acc+/-2, Acc+/-5, Err_Max, Err_Med).
EXPECTED = {
    "InceptionV3":   {"MAE": 3.31, "RMSE": 6.20, "R2": 0.04, "MAPE": 18.05, "Acc_2yr": 48.28, "Acc_5yr": 87.07, "Max_Error": 33.98, "Min_Error": 0.06, "Median_Error": 2.04},
    "MobileViT-XXS": {"MAE": 2.79, "RMSE": 6.13, "R2": 0.06, "MAPE": 13.59, "Acc_2yr": 65.52, "Acc_5yr": 93.97, "Max_Error": 34.91, "Min_Error": 0.05, "Median_Error": 1.35},
    "Ensemble":      {"MAE": 2.81, "RMSE": 6.09, "R2": 0.07, "MAPE": 14.33, "Acc_2yr": 62.93, "Acc_5yr": 90.52, "Max_Error": 34.44, "Min_Error": 0.02, "Median_Error": 1.29},
}
METRICS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr", "Max_Error", "Min_Error", "Median_Error"]
TOL = 0.01  # metrics are reported to 2 decimals


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn-test", type=Path, default=root / "01_CNN_Ensemble" / "predictions" / "test_image_predictions.csv")
    p.add_argument("--vit-test", type=Path, default=root / "04_ViT_Ensemble" / "predictions" / "test_image_predictions.csv")
    p.add_argument("--output-dir", type=Path, default=root / "04_ViT_Ensemble" / "reproduction_output")
    return p.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(err)),
        "RMSE": rmse,
        "R2": r2,
        "MAPE": float(np.mean(err / y_true) * 100),
        "Acc_2yr": float(np.mean(err <= 2) * 100),
        "Acc_5yr": float(np.mean(err <= 5) * 100),
        "Max_Error": float(np.max(err)),
        "Min_Error": float(np.min(err)),
        "Median_Error": float(np.median(err)),
    }


def main() -> None:
    args = parse_args()
    for path in (args.cnn_test, args.vit_test):
        if not path.is_file():
            raise SystemExit(f"ERROR: missing prediction file: {path}")

    cnn = pd.read_csv(args.cnn_test)
    vit = pd.read_csv(args.vit_test)
    if "InceptionV3" not in cnn.columns:
        raise SystemExit(f"ERROR: {args.cnn_test} has no 'InceptionV3' column")
    if "MobileViT_XXS" not in vit.columns:
        raise SystemExit(f"ERROR: {args.vit_test} has no 'MobileViT_XXS' column")

    merged = cnn[["ImageID", "TrueAge", "InceptionV3"]].merge(
        vit[["ImageID", "MobileViT_XXS"]], on="ImageID", how="inner"
    )
    y = merged["TrueAge"].to_numpy(dtype=float)
    inc = merged["InceptionV3"].to_numpy(dtype=float)
    mob = merged["MobileViT_XXS"].to_numpy(dtype=float)
    ens = (inc + mob) / 2.0

    results = {
        "InceptionV3": compute_metrics(y, inc),
        "MobileViT-XXS": compute_metrics(y, mob),
        "Ensemble": compute_metrics(y, ens),
    }

    print(f"Hybrid ensemble (Table 4) on the official HHD test split ({len(merged)} pages)\n")
    header = f"{'System':16s}" + "".join(f"{m:>12s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name, mets in results.items():
        print(f"{name:16s}" + "".join(f"{mets[m]:12.2f}" for m in METRICS))

    # Verify against the paper.
    failures = []
    for name, mets in results.items():
        for m in METRICS:
            if abs(mets[m] - EXPECTED[name][m]) > TOL + 1e-9:
                failures.append(f"{name}.{m}: got {mets[m]:.2f}, expected {EXPECTED[name][m]:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "hybrid_metrics.csv"
    pd.DataFrame(results).T.reset_index(names="system").to_csv(out_csv, index=False)

    print()
    if failures:
        print("FAILED checks:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print(f"ALL CHECKS PASSED - matches paper Table 4. Wrote {out_csv}")


if __name__ == "__main__":
    main()
