"""Reproduce the DiT Ensemble results of Experiment 06 (paper Table 3 bottom / Table 4).

This is the DiT (PyTorch + HuggingFace) counterpart of
``04_ViT_Ensemble/reproduce_results.py``: four Document Image Transformer
backbones evaluated on the single official HHD test split, plus ensemble
configurations (grid search and inverse-MAE weighting, weights selected on the
validation split). It offers two modes:

    fast (default)
        Recompute the full ensemble selection and every reported metric from
        the *committed* per-image predictions in ``predictions/``. Requires no
        GPU, no model weights and no PyTorch/Transformers -- only the
        ground-truth ages in ``data/NewAgeSplit.csv`` (downloaded automatically
        if absent).

    full
        Download the trained DiT checkpoints from Zenodo, rerun the patch-level
        inference pipeline (reusing ``train_dit.py``) on the validation and test
        splits, then perform the identical ensemble selection. Reproduces the
        numbers end-to-end from the raw images.

In both modes every computed value is compared against the numbers reported in
``results.md`` and a PASS/FAIL summary is printed.

Note: the EXPECTED_* tables and ``results/results.md`` reflect the aligned
1e-3/1e-4 protocol (``train_dit.py`` re-run). If the pipeline changes again,
recompute the committed prediction CSVs and update both in lock-step so this
self-check stays in sync with the paper.

Examples:
    python 06_DiT_Ensemble/reproduce_results.py                 # fast path
    python 06_DiT_Ensemble/reproduce_results.py --mode full     # from images
    python 06_DiT_Ensemble/reproduce_results.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import download_dataset`

DEFAULT_DATA_DIR = REPO_ROOT / "data"
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR / "weights"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reproduction_output"
LABELS_CSV_NAME = "NewAgeSplit.csv"
VAL_PREDICTIONS_CSV = PREDICTIONS_DIR / "val_image_predictions.csv"
TEST_PREDICTIONS_CSV = PREDICTIONS_DIR / "test_image_predictions.csv"

# ===========================================================================
# Experiment definition
# ===========================================================================
# HuggingFace model id -> human-readable name used as the CSV column / result key.
MODEL_DISPLAY = {
    "microsoft/dit-base": "DiT-Base",
    "microsoft/dit-large": "DiT-Large",
    "microsoft/dit-base-finetuned-rvlcdip": "DiT-Base (RVL-CDIP)",
    "microsoft/dit-large-finetuned-rvlcdip": "DiT-Large (RVL-CDIP)",
}
MODEL_IDS = list(MODEL_DISPLAY)
MODEL_NAMES = list(MODEL_DISPLAY.values())  # canonical column / key order

# Ensemble groups are formed from the top-k models ranked by VALIDATION MAE, so
# their membership is decided at run time (see select_and_evaluate). The keys
# below only fix the group sizes and display order.
ENSEMBLE_GROUP_SIZES = {"Best 2": 2, "Best 3": 3, "Full Ensemble": 4}
GRID_STEP = 0.1  # weight resolution for the grid search

# Patch extraction / inference (only used by the full, image-based pipeline).
EVAL_BATCH_SIZE = 128  # inference patch batch; does not affect predictions

# ---------------------------------------------------------------------------
# Pretrained weights hosted on Zenodo (anonymous, DOI-citable download).
# After uploading the four ``<safe_name>_best_model.pt`` files to a Zenodo
# record, set the integer record id below (visible in the record URL, e.g.
# https://zenodo.org/records/1234567) or export HHD_AGE_DIT_ZENODO_RECORD.
# ``safe_name`` is the HF id with '/' replaced by '__' (matches train_dit.py).
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_DIT_ZENODO_RECORD", "21244620")
WEIGHT_FILES = {hf_id: f"{hf_id.replace('/', '__')}_best_model.pt" for hf_id in MODEL_IDS}
# Optional integrity check: fill in {filename: md5_hex} once the files are uploaded.
WEIGHT_MD5 = {
    "microsoft__dit-base_best_model.pt": "91b1fc2b746752d812026e16a0cc9ff0",
    "microsoft__dit-base-finetuned-rvlcdip_best_model.pt": "92e9e585c6dcfab550fb07594ca266cc",
    "microsoft__dit-large_best_model.pt": "ff150d2ff9b1bd8f5e7fa087bdc6cdf2",
    "microsoft__dit-large-finetuned-rvlcdip_best_model.pt": "17cbcc767f88b6bf43d1a6d6077f57ab",
}

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check
# ===========================================================================
METRIC_TOLERANCES = {
    "MAE":          0.02,
    "RMSE":         0.02,
    "R2":           0.02,
    "MAPE":         0.10,
    "Acc_2yr":      1.00,  # step metric: absorbs one ±-band boundary flip (1/116 = 0.86 pp) under full-mode inference nondeterminism
    "Acc_5yr":      1.00,
    "Max_Error":    0.10,
    "Median_Error": 0.05,
}

# Individual models: results.md only reports MAE / RMSE / R2 for the four DiTs.
EXPECTED_INDIVIDUAL_METRICS = {
    "DiT-Base (RVL-CDIP)":  {"MAE": 3.30, "RMSE": 5.84, "R2": 0.15},
    "DiT-Large (RVL-CDIP)": {"MAE": 3.52, "RMSE": 5.88, "R2": 0.14},
    "DiT-Base":             {"MAE": 4.11, "RMSE": 6.32, "R2": 0.00},
    "DiT-Large":            {"MAE": 4.37, "RMSE": 6.33, "R2": 0.00},
}

# Ensemble configs (weights selected on val, evaluated on test). Keys are the
# internal group names ("Full Ensemble" is shown as "Full" in results.md).
EXPECTED_ENSEMBLE_METRICS = {
    ("Best 2",        "Grid Search"): {"MAE": 3.18, "RMSE": 5.77, "R2": 0.17, "MAPE": 17.85, "Acc_2yr": 56.03, "Acc_5yr": 83.62, "Max_Error": 27.68, "Min_Error": 0.01, "Median_Error": 1.70},
    ("Best 3",        "Grid Search"): {"MAE": 3.18, "RMSE": 5.77, "R2": 0.17, "MAPE": 18.05, "Acc_2yr": 56.90, "Acc_5yr": 82.76, "Max_Error": 27.35, "Min_Error": 0.01, "Median_Error": 1.53},
    ("Full Ensemble", "Grid Search"): {"MAE": 3.20, "RMSE": 5.78, "R2": 0.17, "MAPE": 18.27, "Acc_2yr": 57.76, "Acc_5yr": 82.76, "Max_Error": 27.07, "Min_Error": 0.00, "Median_Error": 1.53},
    ("Best 2",        "MAE-based"):   {"MAE": 3.21, "RMSE": 5.76, "R2": 0.17, "MAPE": 18.11, "Acc_2yr": 56.03, "Acc_5yr": 83.62, "Max_Error": 26.93, "Min_Error": 0.02, "Median_Error": 1.75},
    ("Best 3",        "MAE-based"):   {"MAE": 3.24, "RMSE": 5.81, "R2": 0.16, "MAPE": 18.67, "Acc_2yr": 57.76, "Acc_5yr": 81.03, "Max_Error": 26.56, "Min_Error": 0.01, "Median_Error": 1.54},
    ("Full Ensemble", "MAE-based"):   {"MAE": 3.35, "RMSE": 5.87, "R2": 0.14, "MAPE": 19.54, "Acc_2yr": 56.90, "Acc_5yr": 77.59, "Max_Error": 26.47, "Min_Error": 0.01, "Median_Error": 1.55},
}

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr",
               "Max_Error", "Min_Error", "Median_Error"]


# ===========================================================================
# Metrics (keys aligned with train_dit.py compute_full_metrics)
# ===========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """Return the full metric suite reported in the paper for one prediction set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = 0.0 if np.isnan(mape) else mape
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape,
        "Acc_2yr": float(np.mean(errors <= 2) * 100),
        "Acc_5yr": float(np.mean(errors <= 5) * 100),
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }


def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    return df.assign(**{c: df[c].round(decimals) for c in numeric_cols})


# ===========================================================================
# Ensemble selection (mirrors train_dit.py: grid search + inverse-MAE weights)
# ===========================================================================
def _pivot_true(pivot: pd.DataFrame) -> dict:
    """{ImageID: TrueAge} from a prediction pivot."""
    return dict(zip(pivot["ImageID"], pivot["TrueAge"].astype(float)))


def individual_maes(pivot: pd.DataFrame, models=MODEL_NAMES) -> dict:
    """Per-model MAE over a pivot table (used to rank models and weight them)."""
    maes = {}
    for model in models:
        if model not in pivot.columns:
            continue
        sub = pivot[["TrueAge", model]].dropna()
        if not sub.empty:
            maes[model] = mean_absolute_error(sub["TrueAge"], sub[model])
    return maes


def _ensemble_arrays(pivot: pd.DataFrame, weights: dict, group_models):
    """Return aligned (y_true, y_pred) arrays for a weighted ensemble.

    Only rows where every group model has a prediction are used (matches the
    ``set.intersection`` behaviour of train_dit.py). Weights sum to 1.
    """
    sub = pivot[["TrueAge", *group_models]].dropna()
    y_true = sub["TrueAge"].to_numpy(dtype=float)
    y_pred = sum(sub[m].to_numpy(dtype=float) * weights[m] for m in group_models)
    return y_true, np.asarray(y_pred, dtype=float)


def grid_search_weights(group_models, val_pivot, step=GRID_STEP):
    """Search weights in {0.1..0.9} summing to 1.0 minimising VALIDATION MAE."""
    ranges = [np.arange(step, 1.0, step) for _ in group_models]
    best_weights, best_mae = None, float("inf")
    for combo in itertools.product(*ranges):
        if not np.isclose(sum(combo), 1.0, atol=1e-5):
            continue
        weights = dict(zip(group_models, combo))
        y_true, y_pred = _ensemble_arrays(val_pivot, weights, group_models)
        mae = mean_absolute_error(y_true, y_pred)
        if mae < best_mae:
            best_mae, best_weights = mae, weights
    return best_weights, best_mae


def inverse_mae_weights(group_models, model_maes) -> dict:
    """Weights proportional to 1 / validation MAE (matches train_dit.py)."""
    inv = {m: 1.0 / model_maes[m] for m in group_models}
    total = sum(inv.values())
    return {m: inv[m] / total for m in group_models}


def select_and_evaluate(val_pivot, test_pivot):
    """Select ensemble weights on validation and evaluate on test (leakage-free).

    Returns (summary_df, val_maes). ``summary_df`` has one row per (group,
    method) sorted by test MAE; ``val_maes`` is a {model: val_MAE} dict.
    """
    val_maes = individual_maes(val_pivot)
    ranked = sorted(val_maes, key=lambda m: val_maes[m])

    groups = {}
    for name, size in ENSEMBLE_GROUP_SIZES.items():
        if len(ranked) >= size:
            groups[name] = ranked[:size]

    rows = []
    for group, models in groups.items():
        grid_w, grid_val_mae = grid_search_weights(models, val_pivot)
        mae_w = inverse_mae_weights(models, val_maes)
        val_true_m, val_pred_m = _ensemble_arrays(val_pivot, mae_w, models)
        mae_based_val_mae = mean_absolute_error(val_true_m, val_pred_m)
        methods = (
            ("Grid Search", grid_w, grid_val_mae),
            ("MAE-based", mae_w, mae_based_val_mae),
        )
        for method, weights, val_mae in methods:
            y_true, y_pred = _ensemble_arrays(test_pivot, weights, models)
            rows.append({
                "Ensemble Group": group,
                "Method": method,
                "Val_MAE": val_mae,
                "Weights": json.dumps({m: round(float(weights[m]), 2) for m in models}),
                **compute_metrics(y_true, y_pred),
            })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), val_maes


def individual_metrics_from_pivot(test_pivot) -> dict:
    """Per-model full metric suite on the test pivot."""
    out = {}
    for model in MODEL_NAMES:
        if model not in test_pivot.columns:
            continue
        sub = test_pivot[["TrueAge", model]].dropna()
        if not sub.empty:
            out[model] = compute_metrics(sub["TrueAge"], sub[model])
    return out


# ===========================================================================
# Ground-truth labels
# ===========================================================================
def load_labels_df(data_dir) -> pd.DataFrame:
    csv_path = Path(data_dir) / LABELS_CSV_NAME
    if not csv_path.is_file():
        print(f"Labels not found at {csv_path}; attempting dataset download...")
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
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}; download may be corrupt.")
    tmp.replace(dest)
    return dest


def ensure_weights(weights_dir) -> Path:
    weights_dir = Path(weights_dir)
    missing = [(hf_id, fn) for hf_id, fn in WEIGHT_FILES.items()
               if not (weights_dir / fn).is_file()]
    if not missing:
        return weights_dir
    if ZENODO_RECORD_ID == "REPLACE_WITH_ZENODO_RECORD_ID":
        raise RuntimeError(
            "DiT weights are missing and no Zenodo record id is configured.\n"
            f"Either place the checkpoints at {weights_dir}/<safe_name>_best_model.pt\n"
            "or set ZENODO_RECORD_ID in this script (or the HHD_AGE_DIT_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )
    print(f"Downloading {len(missing)} checkpoint(s) from Zenodo record {ZENODO_RECORD_ID}:")
    weights_dir.mkdir(parents=True, exist_ok=True)
    for _hf_id, fn in missing:
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fn}?download=1"
        _download(url, weights_dir / fn, WEIGHT_MD5.get(fn))
    return weights_dir


# ===========================================================================
# Full mode: image -> patch -> per-image prediction pipeline (PyTorch)
# ===========================================================================
def _preds_to_pivot(per_model_preds: dict, true_age_dict: dict) -> pd.DataFrame:
    """Assemble {model: {imageid: pred}} into a wide pivot with a TrueAge column."""
    all_ids = sorted(set().union(*[set(d) for d in per_model_preds.values()])) if per_model_preds else []
    all_ids = [i for i in all_ids if i in true_age_dict]
    out = pd.DataFrame({"ImageID": all_ids})
    out["TrueAge"] = [float(true_age_dict[i]) for i in all_ids]
    for model in MODEL_NAMES:
        if model in per_model_preds:
            out[model] = [per_model_preds[model].get(i, np.nan) for i in all_ids]
    return out


def run_full_inference(labels_df, data_dir, weights_dir):
    """Run patch-level inference for every DiT checkpoint on val + test splits.

    Reuses the exact preprocessing / model definition from ``train_dit.py`` so
    the reproduction matches training. Returns (val_pivot, test_pivot).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import BeitImageProcessor

    sys.path.insert(0, str(SCRIPT_DIR))  # allow `import train_dit`
    import train_dit as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights_dir = Path(weights_dir)
    true_age_dict = dict(zip(labels_df["File"], labels_df["Age"].astype(float)))

    val_preds: dict = {}
    test_preds: dict = {}
    for hf_id in MODEL_IDS:
        display = MODEL_DISPLAY[hf_id]
        ckpt = weights_dir / WEIGHT_FILES[hf_id]
        if not ckpt.is_file():
            print(f"  {display}: checkpoint missing -- skipping")
            continue
        print(f"  {display}: loading {ckpt.name}")
        proc = BeitImageProcessor.from_pretrained(hf_id)
        model = T.DiTReg(name=hf_id).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        for split, store in (("val", val_preds), ("test", test_preds)):
            subset = labels_df[labels_df["Set"].str.lower() == split]
            loader = DataLoader(
                T.HHDPatchStream(subset, str(data_dir), proc, augment=False),
                batch_size=EVAL_BATCH_SIZE, num_workers=0,
                pin_memory=True, collate_fn=T.collate_patches,
            )
            print(f"    inferring on {split} split...")
            store[display] = T.generate_predictions(model, loader, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_pivot = _preds_to_pivot(val_preds, true_age_dict)
    test_pivot = _preds_to_pivot(test_preds, true_age_dict)
    return val_pivot, test_pivot


# ===========================================================================
# Verification (computed vs. results.md)
# ===========================================================================
def build_verification(summary, individual_metrics):
    """Return (all_pass, verification_df) comparing every metric against results.md."""
    rows = []
    all_pass = True

    for model in MODEL_NAMES:
        if model not in individual_metrics:
            continue
        expected_model = EXPECTED_INDIVIDUAL_METRICS.get(model, {})
        for metric, tol in METRIC_TOLERANCES.items():
            expected_val = expected_model.get(metric)
            computed_val = individual_metrics[model].get(metric)
            if expected_val is None or computed_val is None:
                continue
            ok = abs(computed_val - expected_val) <= tol
            all_pass &= ok
            rows.append({
                "Type": "Model", "Name": model, "Method": "-", "Metric": metric,
                "Computed": round(computed_val, 2), "Reported": expected_val,
                "Status": "PASS" if ok else "FAIL",
            })

    for _, row in summary.iterrows():
        key = (row["Ensemble Group"], row["Method"])
        expected_ens = EXPECTED_ENSEMBLE_METRICS.get(key, {})
        for metric, tol in METRIC_TOLERANCES.items():
            expected_val = expected_ens.get(metric)
            if expected_val is None or metric not in row:
                continue
            computed_val = float(row[metric])
            ok = abs(computed_val - expected_val) <= tol
            all_pass &= ok
            rows.append({
                "Type": "Ensemble", "Name": row["Ensemble Group"], "Method": row["Method"],
                "Metric": metric, "Computed": round(computed_val, 2), "Reported": expected_val,
                "Status": "PASS" if ok else "FAIL",
            })

    return all_pass, pd.DataFrame(rows)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the DiT Ensemble results of Experiment 06.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["auto", "fast", "full"], default="auto",
        help="'fast' uses the committed per-image prediction CSVs (no GPU/weights); "
             "'full' downloads weights from Zenodo and reruns inference; "
             "'auto' picks 'fast' when the CSVs are present.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory containing NewAgeSplit.csv (and images for full mode).")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="Where DiT checkpoints are cached/downloaded (full mode).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where reproduction output CSVs are written.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs_available = VAL_PREDICTIONS_CSV.is_file() and TEST_PREDICTIONS_CSV.is_file()
    mode = args.mode
    if mode == "auto":
        mode = "fast" if csvs_available else "full"
    print(f"Mode: {mode}")

    labels_df = load_labels_df(args.data_dir)

    if mode == "full":
        weights_dir = ensure_weights(args.weights_dir)
        val_pivot, test_pivot = run_full_inference(labels_df, args.data_dir, weights_dir)
        # Persist freshly computed pivots so future runs can use the fast path.
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        val_pivot.to_csv(VAL_PREDICTIONS_CSV, index=False)
        test_pivot.to_csv(TEST_PREDICTIONS_CSV, index=False)
    else:
        if not csvs_available:
            raise FileNotFoundError(
                f"Fast mode needs {VAL_PREDICTIONS_CSV} and {TEST_PREDICTIONS_CSV}. "
                "Run with --mode full to regenerate them from the model weights."
            )
        val_pivot = pd.read_csv(VAL_PREDICTIONS_CSV)
        test_pivot = pd.read_csv(TEST_PREDICTIONS_CSV)

    individual_metrics = individual_metrics_from_pivot(test_pivot)
    summary, val_maes = select_and_evaluate(val_pivot, test_pivot)

    # --- Individual model metrics CSV (test set) ---
    ind_rows = [{"Model": m, **{k: round(v, 2) for k, v in individual_metrics[m].items()}}
                for m in MODEL_NAMES if m in individual_metrics]
    pd.DataFrame(ind_rows).to_csv(output_dir / "individual_model_metrics.csv", index=False)

    # --- Validation MAE per model ---
    pd.DataFrame([{"Model": m, "Val_MAE": round(val_maes[m], 2)}
                  for m in MODEL_NAMES if m in val_maes]).to_csv(
        output_dir / "val_mae_per_model.csv", index=False)

    # --- Ensemble metrics CSV ---
    _round_df(summary).to_csv(output_dir / "ensemble_metrics.csv", index=False)

    # --- Verification CSV (computed vs. results.md) ---
    all_pass, verification_df = build_verification(summary, individual_metrics)
    verification_df.to_csv(output_dir / "verification.csv", index=False)

    # --- Print summary ---
    print("\n" + "=" * 82)
    print("DiT REPRODUCTION SUMMARY  vs  results.md")
    print("=" * 82)

    print("\n  Validation MAE ranking (determines ensemble composition):")
    for i, m in enumerate(sorted(val_maes, key=lambda k: val_maes[k]), start=1):
        print(f"    {i}. {m:<22} {val_maes[m]:.2f}")

    print("\n  Individual Models (test set):")
    print(f"    {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    for m in MODEL_NAMES:
        if m in individual_metrics:
            mt = individual_metrics[m]
            print(f"    {m:<22} {mt['MAE']:7.2f} {mt['RMSE']:7.2f} {mt['R2']:7.2f}")

    print("\n  Ensemble Configurations (weights selected on val, evaluated on test):")
    print(f"    {'Group':<16} {'Method':<12} {'Val_MAE':>8} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'MAPE':>7} {'±2%':>7} {'±5%':>7}")
    for _, row in summary.iterrows():
        print(f"    {row['Ensemble Group']:<16} {row['Method']:<12} {row['Val_MAE']:8.2f} "
              f"{row['MAE']:7.2f} {row['RMSE']:7.2f} {row['R2']:7.2f} {row['MAPE']:7.2f} "
              f"{row['Acc_2yr']:7.2f} {row['Acc_5yr']:7.2f}")

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n{status} — outputs in {output_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
