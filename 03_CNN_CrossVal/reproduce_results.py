"""Reproduce the CNN cross-validation results of Experiment 03 (paper Table 2, top).

This is the cross-validation counterpart of 01_CNN_Ensemble/reproduce_results.py.
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
    python 03_CNN_CrossVal/reproduce_results.py                 # fast path
    python 03_CNN_CrossVal/reproduce_results.py --mode full     # from weights
    python 03_CNN_CrossVal/reproduce_results.py --help
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
PREDICTIONS_DIR = SCRIPT_DIR / "Predictions"
OOF_PREDICTIONS_CSV = PREDICTIONS_DIR / "oof_predictions.csv"
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR / "weights"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reproduction_output"
LABELS_CSV_NAME = "NewAgeSplit.csv"

# ===========================================================================
# Experiment definition
# ===========================================================================
MODEL_NAMES = ["ResNet50", "DenseNet121", "InceptionV3", "InceptionResNetV2", "EfficientNetV2M"]
N_FOLDS = 5
SEED = 42

# Patch extraction (only used by the full, image-based pipeline).
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
THR = 0.0054  # empty-patch filter threshold (matches training)
INFERENCE_BATCH_SIZE = 64  # does not affect predictions

# Fine-tuned CV checkpoints hosted on Zenodo (anonymous, DOI-citable download).
# File layout on the record: {model}_fold{fold}_best_model.keras
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_CV_ZENODO_RECORD", "REPLACE_WITH_ZENODO_RECORD_ID")
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
    "ResNet50":          {"MAE": (5.41, 0.78), "RMSE": (8.17, 0.58), "R2": (0.10, 0.06), "MAPE": (25.68, 3.73),
                          "Acc_2yr": (23.72, 5.17), "Acc_5yr": (63.40, 11.26),
                          "Max_Error": (32.99, 7.14), "Median_Error": (3.72, 0.88)},
    "DenseNet121":       {"MAE": (5.46, 1.06), "RMSE": (8.16, 0.56), "R2": (0.11, 0.06), "MAPE": (26.25, 5.47),
                          "Acc_2yr": (21.49, 16.51), "Acc_5yr": (61.61, 16.41),
                          "Max_Error": (34.14, 6.66), "Median_Error": (3.94, 1.33)},
    "InceptionV3":       {"MAE": (6.03, 0.64), "RMSE": (8.41, 0.64), "R2": (0.05, 0.05), "MAPE": (29.97, 3.17),
                          "Acc_2yr": (16.40, 4.30), "Acc_5yr": (52.83, 9.01),
                          "Max_Error": (32.70, 7.60), "Median_Error": (4.80, 0.67)},
    "InceptionResNetV2": {"MAE": (5.69, 0.70), "RMSE": (7.98, 0.68), "R2": (0.17, 0.11), "MAPE": (29.08, 3.16),
                          "Acc_2yr": (16.76, 3.99), "Acc_5yr": (58.32, 9.19),
                          "Max_Error": (32.62, 5.80), "Median_Error": (4.37, 0.59)},
    "EfficientNetV2M":   {"MAE": (7.30, 0.28), "RMSE": (9.03, 0.36), "R2": (-0.07, 0.14), "MAPE": (41.48, 5.73),
                          "Acc_2yr": (11.58, 4.78), "Acc_5yr": (29.47, 2.21),
                          "Max_Error": (31.44, 5.01), "Median_Error": (7.05, 0.68)},
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
def build_verification(summary: dict):
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
                    "Computed_Mean": round(mean_val, 4), "Computed_Std": round(std_val, 4),
                    "Reported_Mean": exp_mean, "Reported_Std": exp_std,
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
                row[f"{k}_mean"] = round(mean_val, 4)
                row[f"{k}_std"] = round(std_val, 4)
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
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if expected_md5 is not None and _md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}; download may be corrupt.")
    tmp.replace(dest)
    return dest


def ensure_weights(weights_dir) -> Path:
    """Ensure all fold checkpoints exist locally, downloading from Zenodo if needed."""
    weights_dir = Path(weights_dir)
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
            "or set ZENODO_RECORD_ID in this script (or the HHD_AGE_CV_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )

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
    import tensorflow as tf
    from PIL import Image

    data_dir = Path(data_dir)
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

    def process_row(row, root_dir):
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
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, ids

    def make_dataset(df_subset):
        ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
        ds = ds.map(lambda r: process_row(r, str(data_dir)), num_parallel_calls=tf.data.AUTOTUNE)
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
        for fold, (_, val_idx) in enumerate(splits, start=1):
            ckpt = Path(weights_dir) / model_name / f"{model_name}_fold{fold}_best_model.keras"
            if not ckpt.is_file():
                print(f"  {model_name} fold {fold}: checkpoint missing -- skipping")
                continue
            print(f"  {model_name} fold {fold}: loading {ckpt.name}")
            model = tf.keras.models.load_model(ckpt, compile=False)
            val_df = labels_df.iloc[val_idx].reset_index(drop=True)
            preds_per_image = defaultdict(list)
            for patches, ids in make_dataset(val_df):
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
        description="Reproduce the CNN cross-validation results of Experiment 03.",
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

    # --- Per-fold raw predictions (self-contained: includes TrueAge + AbsError, rounded) ---
    _write_per_fold_preds(oof_df, output_dir)

    # --- Per-fold metrics (rounded to 4 dp, includes Acc_10yr) ---
    fold_metrics_df = compute_per_fold(oof_df)
    _round_df(fold_metrics_df).to_csv(output_dir / "cv_metrics_per_fold.csv", index=False)

    # --- Summary: mean ± std across folds ---
    summary = summarize_folds(fold_metrics_df)
    _write_summary_csv(summary, output_dir / "cv_metrics_summary.csv")
    _write_summary_readable(summary, output_dir / "cv_metrics_readable.csv")

    # --- Ensemble predictions and metrics ---
    ensemble_df = _build_ensemble_oof(oof_df)
    ensemble_df.to_csv(output_dir / "ensemble_final.csv", index=False)
    ens_metrics = compute_metrics(ensemble_df["True_Age"], ensemble_df["Pred_Age"])
    ens_row = {"Model": "Ensemble", **{k: round(v, 4) for k, v in ens_metrics.items()}}
    pd.DataFrame([ens_row]).to_csv(output_dir / "ensemble_metrics.csv", index=False)

    # --- Verification (computed vs. results.md) ---
    all_pass, verification_df = build_verification(summary)
    verification_df.to_csv(output_dir / "verification.csv", index=False)

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"{status} — outputs in {output_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
