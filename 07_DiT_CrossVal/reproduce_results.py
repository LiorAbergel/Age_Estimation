"""Reproduce the DiT cross-validation results of Experiment 07 (paper Table 2, bottom).

This is the cross-validation counterpart of ``06_DiT_Ensemble/reproduce_results.py`` and the
DiT (PyTorch + HuggingFace) analogue of ``05_ViT_CrossVal/reproduce_results.py``.

It offers two modes:

    fast (default)
        Recompute every per-fold and mean+/-std metric from the *committed*
        out-of-fold predictions in ``predictions/oof_predictions.csv``. Requires
        no GPU, no model weights and no PyTorch/Transformers -- only the
        ground-truth ages in ``data/NewAgeSplit.csv`` (downloaded automatically
        if absent).

    full
        Download the fine-tuned per-fold checkpoints from Zenodo, rerun the
        out-of-fold inference pipeline (StratifiedGroupKFold with the same seed
        and identical preprocessing, reusing ``train_dit_cv.py``), then perform
        the same aggregation.

In both modes every model's mean+/-std is compared against the numbers reported
in results.md and the results are written to CSV files in ``reproduction_output/``.

Note: ``results.md`` flags its current tables as pending regeneration under the
aligned 1e-3/1e-4 protocol. Once ``train_dit_cv.py`` is re-run, commit the fresh
``oof_predictions.csv`` and update EXPECTED_CV below so this self-check stays in
sync with the paper.

Examples:
    python 07_DiT_CrossVal/reproduce_results.py                 # fast path
    python 07_DiT_CrossVal/reproduce_results.py --mode full     # from weights
    python 07_DiT_CrossVal/reproduce_results.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import download_dataset`

DEFAULT_DATA_DIR = REPO_ROOT / "data"
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
OOF_PREDICTIONS_CSV = PREDICTIONS_DIR / "oof_predictions.csv"
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR / "weights"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reproduction_output"
LABELS_CSV_NAME = "NewAgeSplit.csv"

# ===========================================================================
# Experiment definition
# ===========================================================================
# HuggingFace model id -> human-readable name (used for display / verification).
# ``safe_name`` (id with '/' -> '__') is the canonical key stored in the OOF CSV,
# matching train_dit_cv.py.
MODEL_DISPLAY = {
    "microsoft/dit-base": "DiT-Base",
    "microsoft/dit-large": "DiT-Large",
    "microsoft/dit-base-finetuned-rvlcdip": "DiT-Base (RVL-CDIP)",
    "microsoft/dit-large-finetuned-rvlcdip": "DiT-Large (RVL-CDIP)",
}
MODEL_IDS = list(MODEL_DISPLAY)
SAFE_NAMES = [hf_id.replace("/", "__") for hf_id in MODEL_IDS]
SAFE_TO_DISPLAY = {hf_id.replace("/", "__"): disp for hf_id, disp in MODEL_DISPLAY.items()}

N_FOLDS = 5
SEED = 42

# Patch extraction / inference (only used by the full, image-based pipeline).
EVAL_BATCH_SIZE = 128  # inference patch batch; does not affect predictions

# Fine-tuned CV checkpoints hosted on Zenodo (anonymous, DOI-citable download).
# File layout on the record: {safe_name}_fold{fold:02d}_best_model.pt
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_DIT_CV_ZENODO_RECORD", "21244620")
WEIGHT_MD5 = {
    "microsoft__dit-base_fold01_best_model.pt": "3d88c7c16949ac53ab2560efe502dff5",
    "microsoft__dit-base_fold02_best_model.pt": "d0d437b2b1a66aefb4288639ab7b1e2d",
    "microsoft__dit-base_fold03_best_model.pt": "bf48c5dd19a7f50a937e3e3fe599c1eb",
    "microsoft__dit-base_fold04_best_model.pt": "26db42ec81220102fce01f22858fd1fd",
    "microsoft__dit-base_fold05_best_model.pt": "da56daafe0da08c73b7528e48e4b2d79",
    "microsoft__dit-base-finetuned-rvlcdip_fold01_best_model.pt": "c596ba58e52c7ad1627bc5caacc25197",
    "microsoft__dit-base-finetuned-rvlcdip_fold02_best_model.pt": "dd5755cef98105a9ab4a711786de01b2",
    "microsoft__dit-base-finetuned-rvlcdip_fold03_best_model.pt": "7db3c3e0cdee71f6266432cdcdc5f568",
    "microsoft__dit-base-finetuned-rvlcdip_fold04_best_model.pt": "67207b624555d745601c2b3d3e8a1677",
    "microsoft__dit-base-finetuned-rvlcdip_fold05_best_model.pt": "ad039d932221d4eace38cf0a96f125c2",
    "microsoft__dit-large_fold01_best_model.pt": "1e0e212e98e92102c56d08adf1468a10",
    "microsoft__dit-large_fold02_best_model.pt": "4cf231c0fef210dc140db40f663fda44",
    "microsoft__dit-large_fold03_best_model.pt": "99aa14aac4cca9f65379baa8624c11a9",
    "microsoft__dit-large_fold04_best_model.pt": "80fac3b850a237415570bf168a44f86e",
    "microsoft__dit-large_fold05_best_model.pt": "25b74f9ecbc6defc5950af786d5284d4",
    "microsoft__dit-large-finetuned-rvlcdip_fold01_best_model.pt": "957ae3def3e1ae266b8fa5291750e0ef",
    "microsoft__dit-large-finetuned-rvlcdip_fold02_best_model.pt": "797aa37dd8452eaa42d6f90056dd2d07",
    "microsoft__dit-large-finetuned-rvlcdip_fold03_best_model.pt": "3d8e2f235e9754b7197261882311667e",
    "microsoft__dit-large-finetuned-rvlcdip_fold04_best_model.pt": "14dcf2efa7c39c6b2fb8b3c0dafbc386",
    "microsoft__dit-large-finetuned-rvlcdip_fold05_best_model.pt": "e2144b733a059cf21cb0569a6511efc4",
}

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr",
               "Max_Error", "Min_Error", "Median_Error"]

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check.
# Keyed by safe_name; each metric is (mean, std) across the 5 folds.
# ===========================================================================
METRIC_TOLERANCES = {
    "MAE":          0.15,
    "RMSE":         0.20,
    "R2":           0.05,
    "MAPE":         0.50,
    "Acc_2yr":      0.50,
    "Acc_5yr":      0.50,
    "Max_Error":    0.50,
    "Min_Error":    0.05,
    "Median_Error": 0.15,
}

EXPECTED_CV = {
    "microsoft__dit-base-finetuned-rvlcdip": {
        "MAE": (5.01, 0.53), "RMSE": (7.74, 0.77), "R2": (0.19, 0.11), "MAPE": (23.85, 1.31),
        "Acc_2yr": (31.31, 2.83), "Acc_5yr": (69.29, 6.43),
        "Max_Error": (31.80, 7.30), "Min_Error": (0.03, 0.02), "Median_Error": (3.26, 0.30)},
    "microsoft__dit-base": {
        "MAE": (5.12, 0.59), "RMSE": (7.70, 0.53), "R2": (0.20, 0.08), "MAPE": (25.42, 4.01),
        "Acc_2yr": (31.90, 7.88), "Acc_5yr": (66.36, 4.82),
        "Max_Error": (31.35, 7.98), "Min_Error": (0.03, 0.03), "Median_Error": (3.24, 0.71)},
    "microsoft__dit-large-finetuned-rvlcdip": {
        "MAE": (5.32, 0.53), "RMSE": (7.87, 0.62), "R2": (0.16, 0.12), "MAPE": (25.99, 2.71),
        "Acc_2yr": (21.58, 11.97), "Acc_5yr": (67.89, 6.81),
        "Max_Error": (32.94, 8.89), "Min_Error": (0.06, 0.08), "Median_Error": (3.66, 0.69)},
    "microsoft__dit-large": {
        "MAE": (5.92, 0.93), "RMSE": (8.24, 0.57), "R2": (0.09, 0.07), "MAPE": (30.45, 6.65),
        "Acc_2yr": (21.70, 9.31), "Acc_5yr": (52.38, 20.99),
        "Max_Error": (33.49, 8.38), "Min_Error": (0.05, 0.05), "Median_Error": (4.77, 1.53)},
}


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """Full metric suite (per fold) reported in the paper for one prediction set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = 0.0 if np.isnan(mape) else mape
    return {
        "MAE":          mean_absolute_error(y_true, y_pred),
        "RMSE":         float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":           r2_score(y_true, y_pred),
        "MAPE":         mape,
        "Acc_2yr":      float(np.mean(errors <= 2) * 100),
        "Acc_5yr":      float(np.mean(errors <= 5) * 100),
        "Max_Error":    float(np.max(errors)),
        "Min_Error":    float(np.min(errors)),
        "Median_Error": float(np.median(errors)),
    }


# ===========================================================================
# Aggregation (shared by both modes)
# ===========================================================================
def compute_per_fold(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per (model, fold) and all metrics as columns."""
    rows = []
    for safe_name in SAFE_NAMES:
        sub = oof_df[oof_df["Model"] == safe_name]
        if sub.empty:
            continue
        for fold_id, fold_df in sub.groupby("Fold"):
            m = compute_metrics(fold_df["TrueAge"], fold_df["Prediction"])
            rows.append({"Model": safe_name, "Fold": int(fold_id), **m})
    return pd.DataFrame(rows)


def summarize_folds(fold_df: pd.DataFrame) -> dict:
    """Return {safe_name: {metric: (mean, std)}} aggregated across folds.

    Uses population std (ddof=0) to match ``train_dit_cv.py`` save_cv_results.
    """
    summary = {}
    for safe_name in SAFE_NAMES:
        sub = fold_df[fold_df["Model"] == safe_name]
        if sub.empty:
            continue
        summary[safe_name] = {
            k: (float(sub[k].mean()), float(sub[k].std(ddof=0)))
            for k in METRIC_KEYS if k in sub.columns
        }
    return summary


# ===========================================================================
# Verification (computed vs. results.md)
# ===========================================================================
def build_verification(summary: dict):
    """Return (all_pass, verification_df) comparing every metric against results.md."""
    rows = []
    all_pass = True
    for safe_name in SAFE_NAMES:
        if safe_name not in summary:
            continue
        expected_model = EXPECTED_CV.get(safe_name, {})
        for metric, tol in METRIC_TOLERANCES.items():
            if metric not in summary[safe_name]:
                continue
            mean_val, std_val = summary[safe_name][metric]
            expected = expected_model.get(metric)
            if expected is None:
                continue
            exp_mean, exp_std = expected
            ok = abs(mean_val - exp_mean) <= tol
            all_pass &= ok
            rows.append({
                "Type": "Model", "Name": SAFE_TO_DISPLAY.get(safe_name, safe_name),
                "Metric": metric,
                "Computed_Mean": round(mean_val, 2), "Computed_Std": round(std_val, 2),
                "Reported_Mean": exp_mean, "Reported_Std": exp_std,
                "Status": "PASS" if ok else "FAIL",
            })
    return all_pass, pd.DataFrame(rows)


# ===========================================================================
# Output helpers
# ===========================================================================
def _write_summary_csv(summary: dict, path: Path) -> None:
    """Wide-format summary: one row per model, separate _mean and _std columns per metric."""
    rows = []
    for safe_name in SAFE_NAMES:
        if safe_name not in summary:
            continue
        row = {"Model": SAFE_TO_DISPLAY.get(safe_name, safe_name)}
        for k in METRIC_KEYS:
            if k in summary[safe_name]:
                mean_val, std_val = summary[safe_name][k]
                row[f"{k}_mean"] = round(mean_val, 2)
                row[f"{k}_std"] = round(std_val, 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary_readable(summary: dict, path: Path) -> None:
    """Long-format readable summary: one row per (model, metric) with a 'X.XX ± Y.YY' column."""
    rows = []
    for safe_name in SAFE_NAMES:
        if safe_name not in summary:
            continue
        for k in METRIC_KEYS:
            if k in summary[safe_name]:
                mean_val, std_val = summary[safe_name][k]
                rows.append({
                    "Model": SAFE_TO_DISPLAY.get(safe_name, safe_name),
                    "Metric": k,
                    "Mean": round(mean_val, 2),
                    "Std": round(std_val, 2),
                    "Mean_Std": f"{mean_val:.2f} ± {std_val:.2f}",
                })
    pd.DataFrame(rows).to_csv(path, index=False)


# ===========================================================================
# Ground-truth labels
# ===========================================================================
def load_labels_df(data_dir) -> pd.DataFrame:
    """Load NewAgeSplit.csv, downloading the dataset from Zenodo if absent."""
    csv_path = Path(data_dir) / LABELS_CSV_NAME
    if not csv_path.is_file():
        try:
            from download_dataset import ensure_dataset
            ensure_dataset(str(data_dir))
        except Exception as exc:  # pragma: no cover - environment dependent
            raise FileNotFoundError(
                f"Ground-truth labels not found at {csv_path} and automatic download "
                f"failed ({exc}). See the README for dataset setup instructions."
            ) from exc
    return pd.read_csv(csv_path)


# ===========================================================================
# Weight download (Zenodo)
# ===========================================================================
def _md5(path, chunk=1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, dest, expected_md5=None) -> Path:
    dest = Path(dest)
    if dest.is_file() and (expected_md5 is None or _md5(dest) == expected_md5):
        print(f"  [cached] {dest.name}")
        return dest

    print(f"  downloading {dest.name} ...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            sys.stdout.write(f"\r    {pct:3d}%")
            sys.stdout.flush()

    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, _progress)
    sys.stdout.write("\r")
    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}; download may be corrupt.")
    tmp.replace(dest)
    return dest


def _weight_name(safe_name, fold):
    return f"{safe_name}_fold{fold:02d}_best_model.pt"


def ensure_weights(weights_dir) -> Path:
    """Ensure all fold checkpoints exist locally, downloading from Zenodo if needed."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    expected = [(s, f, _weight_name(s, f))
                for s in SAFE_NAMES for f in range(1, N_FOLDS + 1)]
    missing = [(s, f, fn) for (s, f, fn) in expected if not (weights_dir / s / fn).is_file()]
    if not missing:
        return weights_dir

    if ZENODO_RECORD_ID == "REPLACE_WITH_ZENODO_RECORD_ID":
        raise RuntimeError(
            "CV weights are missing and no Zenodo record id is configured.\n"
            f"Either place the fold checkpoints under "
            f"{weights_dir}/<safe_name>/<safe_name>_fold<KK>_best_model.pt\n"
            "or set ZENODO_RECORD_ID in this script (or the HHD_AGE_DIT_CV_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )

    print(f"Downloading {len(missing)} weight file(s) from Zenodo record {ZENODO_RECORD_ID}:")
    for s, _f, fn in missing:
        (weights_dir / s).mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fn}?download=1"
        _download(url, weights_dir / s / fn, WEIGHT_MD5.get(fn))
    return weights_dir


# ===========================================================================
# Full mode: image -> patch -> OOF prediction pipeline (PyTorch)
# ===========================================================================
def run_full_oof(labels_df, data_dir, weights_dir) -> pd.DataFrame:
    """Rerun out-of-fold inference for every fold checkpoint; return an OOF DataFrame.

    Reuses the exact preprocessing / model definition / CV splits from
    ``train_dit_cv.py`` so the reproduction matches training.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import BeitImageProcessor

    sys.path.insert(0, str(SCRIPT_DIR))  # allow `import train_dit_cv`
    import train_dit_cv as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights_dir = Path(weights_dir)

    # Reproduce the exact CV splits used in training.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(labels_df.index, labels_df["AgeGroup"], labels_df["WriterNumber"]))

    records = []
    for hf_id in MODEL_IDS:
        safe_name = hf_id.replace("/", "__")
        display = MODEL_DISPLAY[hf_id]
        proc = BeitImageProcessor.from_pretrained(hf_id)
        for fold, (_tr_idx, val_idx) in enumerate(splits, start=1):
            ckpt = weights_dir / safe_name / _weight_name(safe_name, fold)
            if not ckpt.is_file():
                print(f"  {display} fold {fold}: checkpoint missing -- skipping")
                continue
            print(f"  {display} fold {fold}: loading {ckpt.name}")
            val_df = labels_df.iloc[val_idx].reset_index(drop=True)
            loader = DataLoader(
                T.HHDPatchStream(val_df, str(data_dir), proc, augment=False),
                batch_size=EVAL_BATCH_SIZE, num_workers=0,
                pin_memory=True, collate_fn=T.collate_patches,
            )
            model = T.DiTReg(name=hf_id).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            img_preds = defaultdict(list)
            for batch in loader:
                if not batch:
                    continue
                px = batch["pixel_values"].to(device)
                ids = batch["file_ids"]
                with torch.no_grad():
                    preds = model(pixel_values=px)["preds"].cpu().numpy()
                for p, fid in zip(preds, ids):
                    img_preds[fid].append(p)

            true_map = dict(zip(val_df["File"], val_df["Age"]))
            for iid, plist in img_preds.items():
                if iid in true_map:
                    records.append({
                        "Model": safe_name, "Fold": fold, "ImageID": iid,
                        "Prediction": float(np.mean(plist)), "TrueAge": float(true_map[iid]),
                    })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(records)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the DiT cross-validation results of Experiment 07.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["auto", "fast", "full"], default="auto",
        help="'fast' uses the committed OOF prediction CSV (no GPU/weights); "
             "'full' downloads weights from Zenodo and reruns OOF inference; "
             "'auto' picks 'fast' when the CSV is present.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory containing NewAgeSplit.csv (and images for full mode).")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="Where fold checkpoints are cached/downloaded (full mode).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where output CSVs are written.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "fast" if OOF_PREDICTIONS_CSV.is_file() else "full"
    print(f"Mode: {mode}")

    labels_df = load_labels_df(args.data_dir)

    if mode == "full":
        weights_dir = ensure_weights(args.weights_dir)
        oof_df = run_full_oof(labels_df, args.data_dir, weights_dir)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(OOF_PREDICTIONS_CSV, index=False)
    else:
        if not OOF_PREDICTIONS_CSV.is_file():
            raise FileNotFoundError(
                f"Fast mode needs {OOF_PREDICTIONS_CSV}. Run with --mode full to "
                "regenerate it from the model weights."
            )
        oof_df = pd.read_csv(OOF_PREDICTIONS_CSV)

    # --- Per-fold metrics (used to compute summary) ---
    fold_metrics_df = compute_per_fold(oof_df)
    fold_metrics_df.to_csv(output_dir / "cv_metrics_per_fold.csv", index=False)

    # --- Summary: mean ± std across folds ---
    summary = summarize_folds(fold_metrics_df)
    _write_summary_csv(summary, output_dir / "cv_metrics_summary.csv")
    _write_summary_readable(summary, output_dir / "cv_metrics_summary_readable.csv")

    # --- Verification (computed vs. results.md) ---
    all_pass, verification_df = build_verification(summary)
    verification_df.to_csv(output_dir / "verification.csv", index=False)

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("DiT CV REPRODUCTION SUMMARY  vs  results.md  (mean ± std across folds)")
    print("=" * 72)
    for safe_name in SAFE_NAMES:
        if safe_name not in summary:
            continue
        disp = SAFE_TO_DISPLAY.get(safe_name, safe_name)
        m = summary[safe_name]
        print(f"\n  {disp}:")
        for k in ("MAE", "RMSE", "R2"):
            if k in m:
                print(f"    {k:<5}: {m[k][0]:.2f} ± {m[k][1]:.2f}")

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n{status} — outputs in {output_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
