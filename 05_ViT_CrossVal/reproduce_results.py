"""Reproduce the ViT cross-validation results of Experiment 05 (paper Table 2, middle).

This is the cross-validation counterpart of 04_ViT/reproduce_results.py and the
ViT analogue of 03_CNN_CrossVal/reproduce_results.py.

It offers two modes:

    fast (default)
        Recompute every per-fold and mean+/-std metric from the *committed*
        out-of-fold predictions in ``predictions/oof_predictions.csv``. Requires
        no GPU, no model weights and no TensorFlow -- only the ground-truth ages
        in ``data/NewAgeSplit.csv`` (downloaded automatically if absent).

    full
        Download the fine-tuned per-fold checkpoints from Zenodo, rerun the
        out-of-fold inference pipeline (StratifiedGroupKFold with the same seed
        and identical preprocessing), then perform the same aggregation.

In both modes every model's mean+/-std is compared against the numbers reported
in results.md and the results are written to CSV files in ``reproduction_output/``.

Examples:
    python 05_ViT_CrossVal/reproduce_results.py                 # fast path
    python 05_ViT_CrossVal/reproduce_results.py --mode full     # from weights
    python 05_ViT_CrossVal/reproduce_results.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
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
MODEL_NAMES = ["SwinV2_Tiny", "MobileViT_XXS", "ConvNeXtV2_Tiny", "TinyViT_11M"]
N_FOLDS = 5
SEED = 42

# ViT model input resolutions (must match training)
VIT_MODEL_INPUT_SIZE = {
    "SwinV2_Tiny": 256,
    "MobileViT_XXS": 256,
    "ConvNeXtV2_Tiny": 224,
    "TinyViT_11M": 224,
}

# Patch extraction (only used by the full, image-based pipeline).
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
THR = 0.0054  # empty-patch filter threshold (matches training)
INFERENCE_BATCH_SIZE = 64  # does not affect predictions

# Fine-tuned CV checkpoints hosted on Zenodo (anonymous, DOI-citable download).
# File layout on the record: {model}_fold{fold}_best_model.keras
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_VIT_CV_ZENODO_RECORD", "REPLACE_WITH_ZENODO_RECORD_ID")
WEIGHT_MD5: dict[str, str] = {}

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr",
               "Max_Error", "Min_Error", "Median_Error"]

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check
# ===========================================================================
METRIC_TOLERANCES = {
    "MAE":          0.15,
    "RMSE":         0.20,
    "R2":           0.05,
    "MAPE":         0.50,
    "Acc_2yr":      0.50,
    "Acc_5yr":      0.50,
    "Max_Error":    0.50,
    "Median_Error": 0.15,
}

EXPECTED_CV = {
    "SwinV2_Tiny":     {"MAE": (6.55, 0.56), "RMSE": (8.71, 0.46), "R2": (-0.02, 0.02), "MAPE": (33.40, 3.05),
                        "Acc_2yr": (10.90, 4.29), "Acc_5yr": (44.36, 10.78),
                        "Max_Error": (32.74, 7.74), "Median_Error": (5.88, 1.09)},
    "MobileViT_XXS":   {"MAE": (4.69, 0.22), "RMSE": (7.82, 0.58), "R2": (0.18, 0.09), "MAPE": (21.00, 2.05),
                        "Acc_2yr": (40.75, 5.96), "Acc_5yr": (72.23, 4.23),
                        "Max_Error": (34.16, 6.69), "Median_Error": (2.80, 0.72)},
    "ConvNeXtV2_Tiny": {"MAE": (4.76, 0.72), "RMSE": (7.51, 0.44), "R2": (0.24, 0.10), "MAPE": (22.85, 5.84),
                        "Acc_2yr": (37.35, 14.26), "Acc_5yr": (69.08, 12.42),
                        "Max_Error": (30.43, 6.72), "Median_Error": (2.98, 1.16)},
    "TinyViT_11M":     {"MAE": (5.60, 0.71), "RMSE": (7.85, 0.76), "R2": (0.17, 0.09), "MAPE": (27.39, 2.49),
                        "Acc_2yr": (16.25, 7.27), "Acc_5yr": (59.87, 10.28),
                        "Max_Error": (31.06, 6.94), "Median_Error": (4.33, 0.76)},
}

EXPECTED_ENSEMBLE = {
    "OOF Ensemble": {"MAE": 5.10, "RMSE": 7.69, "R2": 0.22, "MAPE": 24.46,
                      "Acc_2yr": 22.90, "Acc_5yr": 69.39,
                      "Max_Error": 39.40, "Median_Error": 3.68},
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
    for model_name in MODEL_NAMES:
        sub = oof_df[oof_df["Model"] == model_name]
        if sub.empty:
            continue
        for fold_id, fold_df in sub.groupby("Fold"):
            m = compute_metrics(fold_df["TrueAge"], fold_df["Prediction"])
            rows.append({"Model": model_name, "Fold": int(fold_id), **m})
    return pd.DataFrame(rows)


def summarize_folds(fold_df: pd.DataFrame) -> dict:
    """Return {model: {metric: (mean, std)}} aggregated across folds."""
    summary = {}
    for model_name in MODEL_NAMES:
        sub = fold_df[fold_df["Model"] == model_name]
        if sub.empty:
            continue
        summary[model_name] = {
            k: (float(sub[k].mean()), float(sub[k].std()))
            for k in METRIC_KEYS if k in sub.columns
        }
    return summary


# ===========================================================================
# Verification (computed vs. results.md)
# ===========================================================================
def build_verification(summary: dict, ens_metrics: dict):
    """Return (all_pass, verification_df) comparing every metric against results.md."""
    rows = []
    all_pass = True
    for model_name in MODEL_NAMES:
        if model_name not in summary:
            continue
        expected_model = EXPECTED_CV.get(model_name, {})
        for metric, tol in METRIC_TOLERANCES.items():
            if metric not in summary[model_name]:
                continue
            mean_val, std_val = summary[model_name][metric]
            expected = expected_model.get(metric)
            if expected is not None:
                exp_mean, exp_std = expected
                ok = abs(mean_val - exp_mean) <= tol
                all_pass &= ok
                rows.append({
                    "Type": "Model", "Name": model_name, "Method": "-", "Metric": metric,
                    "Computed_Mean": round(mean_val, 2), "Computed_Std": round(std_val, 2),
                    "Reported_Mean": exp_mean, "Reported_Std": exp_std,
                    "Status": "PASS" if ok else "FAIL",
                })
    # Ensemble verification
    expected_ens = EXPECTED_ENSEMBLE.get("OOF Ensemble", {})
    for metric, tol in METRIC_TOLERANCES.items():
        if metric not in ens_metrics:
            continue
        computed = ens_metrics[metric]
        expected = expected_ens.get(metric)
        if expected is not None:
            ok = abs(computed - expected) <= tol
            all_pass &= ok
            rows.append({
                "Type": "Ensemble", "Name": "OOF Ensemble", "Method": "-", "Metric": metric,
                "Computed_Mean": round(computed, 2), "Computed_Std": "",
                "Reported_Mean": expected, "Reported_Std": "",
                "Status": "PASS" if ok else "FAIL",
            })
    return all_pass, pd.DataFrame(rows)


# ===========================================================================
# Output helpers
# ===========================================================================
def _round_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    return df.assign(**{c: df[c].round(decimals) for c in numeric_cols})


def _write_per_fold_preds(oof_df: pd.DataFrame, output_dir: Path) -> None:
    """Write one CSV per (model, fold) with TrueAge, rounded Prediction, and AbsError."""
    for model_name in MODEL_NAMES:
        model_sub = oof_df[oof_df["Model"] == model_name]
        if model_sub.empty:
            continue
        for fold_id, fold_df in model_sub.groupby("Fold"):
            out = fold_df[["Model", "Fold", "ImageID", "TrueAge", "Prediction"]].copy()
            out["AbsError"] = (out["Prediction"] - out["TrueAge"]).abs().round(4)
            out["Prediction"] = out["Prediction"].round(4)
            out["TrueAge"] = out["TrueAge"].round(4)
            out.to_csv(output_dir / f"{model_name}_fold{int(fold_id)}_preds.csv", index=False)


def _build_ensemble_oof(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Average all models' OOF predictions per image; return with signed Error and AbsError."""
    grouped = (
        oof_df.groupby("ImageID")
        .agg(Pred_Age=("Prediction", "mean"), True_Age=("TrueAge", "first"))
        .reset_index()
    )
    grouped["Error"] = (grouped["Pred_Age"] - grouped["True_Age"]).round(4)
    grouped["AbsError"] = grouped["Error"].abs()
    grouped["Pred_Age"] = grouped["Pred_Age"].round(4)
    return grouped[["ImageID", "True_Age", "Pred_Age", "Error", "AbsError"]]


def _write_summary_csv(summary: dict, path: Path) -> None:
    """Wide-format summary: one row per model, separate _mean and _std columns per metric."""
    rows = []
    for model_name in MODEL_NAMES:
        if model_name not in summary:
            continue
        row = {"Model": model_name}
        for k in METRIC_KEYS:
            if k in summary[model_name]:
                mean_val, std_val = summary[model_name][k]
                row[f"{k}_mean"] = round(mean_val, 2)
                row[f"{k}_std"] = round(std_val, 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary_readable(summary: dict, path: Path) -> None:
    """Long-format readable summary: one row per (model, metric) with a 'X.XX ± Y.YY' column."""
    rows = []
    for model_name in MODEL_NAMES:
        if model_name not in summary:
            continue
        for k in METRIC_KEYS:
            if k in summary[model_name]:
                mean_val, std_val = summary[model_name][k]
                rows.append({
                    "Model": model_name,
                    "Metric": k,
                    "Mean": round(mean_val, 4),
                    "Std": round(std_val, 4),
                    "Mean_Std": f"{mean_val:.2f} ± {std_val:.2f}",
                })
    pd.DataFrame(rows).to_csv(path, index=False)


# ===========================================================================
# Ground-truth labels
# ===========================================================================
def load_labels_df(data_dir) -> pd.DataFrame:
    """Load NewAgeSplit.csv, downloading the dataset via kagglehub if absent."""
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

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, _progress)
    sys.stdout.write("\r")
    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}; download may be corrupt.")
    tmp.replace(dest)
    return dest


def ensure_weights(weights_dir) -> Path:
    """Ensure all fold checkpoints exist locally, downloading from Zenodo if needed."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    expected = [(m, f, f"{m}_fold{f}_best_model.keras")
                for m in MODEL_NAMES for f in range(1, N_FOLDS + 1)]
    missing = [(m, f, fn) for (m, f, fn) in expected if not (weights_dir / m / fn).is_file()]
    if not missing:
        return weights_dir

    if ZENODO_RECORD_ID == "REPLACE_WITH_ZENODO_RECORD_ID":
        raise RuntimeError(
            "CV weights are missing and no Zenodo record id is configured.\n"
            f"Either place the fold checkpoints under "
            f"{weights_dir}/<Model>/<Model>_fold<k>_best_model.keras\n"
            "or set ZENODO_RECORD_ID in this script (or the HHD_AGE_VIT_CV_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )

    print(f"Downloading {len(missing)} weight file(s) from Zenodo record {ZENODO_RECORD_ID}:")
    for m, _f, fn in missing:
        (weights_dir / m).mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fn}?download=1"
        _download(url, weights_dir / m / fn, WEIGHT_MD5.get(fn))
    return weights_dir


# ===========================================================================
# Full mode: image -> patch -> OOF prediction pipeline (TensorFlow)
# ===========================================================================
def calculate_resized_dimensions(height, width, patch_size=PATCH_SIZE[0],
                                 stride=STRIDE, standard_size=STANDARD_SIZE):
    """Resize keeping aspect ratio so that whole patches tile the image exactly."""
    aspect_ratio = width / height
    if height < width:
        new_height = standard_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = standard_size
        new_height = int(new_width / aspect_ratio)

    def adjust(dim):
        remainder = (dim - patch_size) % stride
        return dim if remainder == 0 else dim - remainder

    return adjust(new_height), adjust(new_width)


def run_full_oof(labels_df, data_dir, weights_dir) -> pd.DataFrame:
    """Rerun out-of-fold inference for every fold checkpoint; return an OOF DataFrame."""
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
    from PIL import Image

    data_dir = Path(data_dir)
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    true_age_dict = dict(zip(labels_df["File"], labels_df["Age"]))

    def read_image_and_resize(img_path):
        path = img_path.numpy().decode("utf-8")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # pragma: no cover - corrupt/missing file
            print(f"Error reading {path}: {exc}")
            return np.zeros((PATCH_SIZE[0], PATCH_SIZE[1], 3), dtype=np.float32)
        width, height = img.size
        new_h, new_w = calculate_resized_dimensions(height, width)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0

    def process_row(row, root_dir, final_size):
        img_path = tf.strings.join([root_dir, row["Set"], row["File"]], separator=os.sep)
        img = tf.py_function(read_image_and_resize, [img_path], tf.float32)
        img.set_shape([None, None, 3])
        patches = tf.image.extract_patches(
            images=tf.expand_dims(img, 0),
            sizes=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
            strides=[1, STRIDE, STRIDE, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, PATCH_SIZE[0], PATCH_SIZE[1], 3])
        patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
        patches = tf.boolean_mask(patches, patch_means > THR)
        # Resize 400x400 patches to ViT input size (224 or 256) using bicubic
        patches = tf.image.resize(patches, [final_size, final_size], method="bicubic")
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, ids

    def make_dataset(df_subset, final_size):
        ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
        ds = ds.map(lambda r: process_row(r, str(data_dir), final_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda p, i: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(p),
            tf.data.Dataset.from_tensor_slices(i),
        )))
        return ds.batch(INFERENCE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Reproduce the exact CV splits used in training.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(labels_df.index, labels_df["AgeGroup"], labels_df["WriterNumber"]))

    records = []
    for model_name in MODEL_NAMES:
        input_size = VIT_MODEL_INPUT_SIZE[model_name]
        for fold, (_, val_idx) in enumerate(splits, start=1):
            ckpt = Path(weights_dir) / model_name / f"{model_name}_fold{fold}_best_model.keras"
            if not ckpt.is_file():
                print(f"  {model_name} fold {fold}: checkpoint missing -- skipping")
                continue
            print(f"  {model_name} fold {fold}: loading {ckpt.name}")
            model = tf.keras.models.load_model(ckpt, compile=False)
            val_df = labels_df.iloc[val_idx].reset_index(drop=True)
            preds_per_image = defaultdict(list)
            for patches, ids in make_dataset(val_df, input_size):
                p = model.predict(patches, verbose=0).ravel()
                for value, iid in zip(p, ids.numpy()):
                    preds_per_image[iid.decode("utf-8")].append(float(value))
            for iid, plist in preds_per_image.items():
                if iid in true_age_dict:
                    records.append({"Model": model_name, "Fold": fold, "ImageID": iid,
                                    "TrueAge": float(true_age_dict[iid]),
                                    "Prediction": float(np.mean(plist))})
            del model
            tf.keras.backend.clear_session()
    return pd.DataFrame(records)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the ViT cross-validation results of Experiment 05.",
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

    # --- Summary: mean ± std across folds ---
    summary = summarize_folds(fold_metrics_df)
    _write_summary_csv(summary, output_dir / "cv_metrics_summary.csv")

    # --- Ensemble metrics (OOF ensemble: simple average of all model OOF preds) ---
    ensemble_df = _build_ensemble_oof(oof_df)
    ens_metrics = compute_metrics(ensemble_df["True_Age"], ensemble_df["Pred_Age"])
    ens_row = {"Model": "Ensemble", **{k: round(v, 2) for k, v in ens_metrics.items()}}
    pd.DataFrame([ens_row]).to_csv(output_dir / "ensemble_metrics.csv", index=False)

    # --- Verification (computed vs. results.md) ---
    all_pass, verification_df = build_verification(summary, ens_metrics)
    verification_df.to_csv(output_dir / "verification.csv", index=False)

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"{status} — outputs in {output_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
