"""Reproduce the ViT Ensemble results of Experiment 04 (development experiment).

Mirrors the structure of ``01_CNN_Ensemble/reproduce_results.py`` but adapted to
the ViT setup: four backbones evaluated on the single official HHD test split,
plus ensemble configurations (grid search and MAE-based weighting). It offers
two modes:

    fast (default)
        Recompute the full ensemble selection and every reported metric from
        the *committed* per-image predictions in ``predictions/``. Requires no
        GPU, no model weights and no TensorFlow -- only the ground-truth ages in
        ``data/NewAgeSplit.csv`` (downloaded automatically if absent).

    full
        Download the trained ViT checkpoints from Zenodo, run the patch-level
        inference pipeline on the validation and test splits, then perform the
        identical ensemble selection. Reproduces the numbers end-to-end from
        the raw images.

In both modes every computed value is compared against the numbers reported in
``results.md`` and a PASS/FAIL summary is printed.

Examples:
    python 04_ViT_Ensemble/reproduce_results.py                 # fast path
    python 04_ViT_Ensemble/reproduce_results.py --mode full     # from images
    python 04_ViT_Ensemble/reproduce_results.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import urllib.request
from collections import defaultdict
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
# Model -> native input resolution (patches are resized to this before inference).
MODEL_INPUT_SIZES = {
    "SwinV2_Tiny": 256,
    "MobileViT_XXS": 256,
    "ConvNeXtV2_Tiny": 224,
    "TinyViT_11M": 224,
}
MODEL_NAMES = list(MODEL_INPUT_SIZES)

# Ensemble groups (models ranked by validation MAE: MobileViT > ConvNeXtV2 > TinyViT > SwinV2).
ENSEMBLE_GROUPS = {
    "Full Ensemble": ["MobileViT_XXS", "ConvNeXtV2_Tiny", "TinyViT_11M", "SwinV2_Tiny"],
    "Best 3":        ["MobileViT_XXS", "ConvNeXtV2_Tiny", "TinyViT_11M"],
    "Best 2":        ["MobileViT_XXS", "ConvNeXtV2_Tiny"],
}
GRID_STEP = 0.1  # weight resolution for the grid search

# Patch extraction (only used by the full, image-based pipeline).
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
THR = 0.0054  # empty-patch filter threshold (matches the training pipeline)
INFERENCE_BATCH_SIZE = 64  # does not affect predictions

# ---------------------------------------------------------------------------
# Pretrained weights hosted on Zenodo (anonymous, DOI-citable download).
# After uploading the four ``*_best_model.keras`` files to a Zenodo record, set
# the integer record id below (visible in the record URL, e.g.
# https://zenodo.org/records/1234567) or export HHD_AGE_VIT_ZENODO_RECORD.
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_VIT_ZENODO_RECORD", "21244620")
WEIGHT_FILES = {name: f"{name}_best_model.keras" for name in MODEL_NAMES}
# Optional integrity check: fill in {filename: md5_hex} once the files are uploaded.
WEIGHT_MD5 = {
    "ConvNeXtV2_Tiny_best_model.keras": "b255c4d09a61b8ad9128e5ef3526e993",
    "MobileViT_XXS_best_model.keras": "c4d6f0b3ccd3fcece3bcd2ca3496fd47",
    "SwinV2_Tiny_best_model.keras": "4cd93b616a3f68fea343724bed3610b3",
    "TinyViT_11M_best_model.keras": "93ea02b7309f9f8cce9893973bdd22e7",
}

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check
# ===========================================================================
METRIC_TOLERANCES = {
    "MAE":                  0.02,
    "RMSE":                 0.02,
    "R2":                   0.02,
    "MAPE (%)":             0.10,
    "Within ±2 Years (%)":  1.00,  # step metric: absorbs one ±-band boundary flip (1/116 = 0.86 pp) under full-mode inference nondeterminism
    "Within ±5 Years (%)":  1.00,
    "Max Error":            0.10,
    "Median Error":         0.05,  # order statistic: middle-rank reordering shifts it a few hundredths in full mode (matches 06_DiT_Ensemble)
}

EXPECTED_INDIVIDUAL_METRICS = {
    "SwinV2_Tiny":     {"MAE": 5.46, "RMSE": 7.46, "R2": -0.39, "MAPE (%)": 30.87, "Within ±2 Years (%)": 11.21, "Within ±5 Years (%)": 55.17, "Max Error": 32.41, "Median Error": 4.83},
    "MobileViT_XXS":   {"MAE": 2.79, "RMSE": 6.13, "R2": 0.06, "MAPE (%)": 13.59, "Within ±2 Years (%)": 65.52, "Within ±5 Years (%)": 93.97, "Max Error": 34.91, "Median Error": 1.35},
    "ConvNeXtV2_Tiny": {"MAE": 3.86, "RMSE": 6.08, "R2": 0.08, "MAPE (%)": 21.87, "Within ±2 Years (%)": 37.07, "Within ±5 Years (%)": 81.03, "Max Error": 27.99, "Median Error": 2.98},
    "TinyViT_11M":     {"MAE": 4.69, "RMSE": 6.23, "R2": 0.03, "MAPE (%)": 28.95, "Within ±2 Years (%)": 18.97, "Within ±5 Years (%)": 70.69, "Max Error": 25.95, "Median Error": 3.68},
}

EXPECTED_ENSEMBLE_METRICS = {
    ("Best 3",        "Grid Search"): {"MAE": 2.77, "RMSE": 5.84, "R2": 0.15, "MAPE (%)": 14.41, "Within ±2 Years (%)": 66.38, "Within ±5 Years (%)": 87.93, "Max Error": 32.42, "Median Error": 1.26},
    ("Best 2",        "Grid Search"): {"MAE": 2.81, "RMSE": 6.06, "R2": 0.08, "MAPE (%)": 13.87, "Within ±2 Years (%)": 65.52, "Within ±5 Years (%)": 93.10, "Max Error": 34.21, "Median Error": 1.44},
    ("Full Ensemble", "Grid Search"): {"MAE": 2.89, "RMSE": 5.57, "R2": 0.22, "MAPE (%)": 15.75, "Within ±2 Years (%)": 64.66, "Within ±5 Years (%)": 82.76, "Max Error": 28.99, "Median Error": 1.40},
    ("Best 3",        "MAE-based"):   {"MAE": 2.89, "RMSE": 5.68, "R2": 0.19, "MAPE (%)": 15.78, "Within ±2 Years (%)": 63.79, "Within ±5 Years (%)": 81.90, "Max Error": 29.89, "Median Error": 1.24},
    ("Full Ensemble", "MAE-based"):   {"MAE": 3.03, "RMSE": 5.75, "R2": 0.17, "MAPE (%)": 16.09, "Within ±2 Years (%)": 53.45, "Within ±5 Years (%)": 85.34, "Max Error": 30.38, "Median Error": 1.68},
    ("Best 2",        "MAE-based"):   {"MAE": 3.04, "RMSE": 5.93, "R2": 0.12, "MAPE (%)": 15.80, "Within ±2 Years (%)": 55.17, "Within ±5 Years (%)": 86.21, "Max Error": 31.84, "Median Error": 1.77},
}


# ===========================================================================
# Metrics
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
        "MAPE (%)": mape,
        "Within ±2 Years (%)": float(np.mean(errors <= 2) * 100),
        "Within ±5 Years (%)": float(np.mean(errors <= 5) * 100),
        "Max Error": float(np.max(errors)),
        "Min Error": float(np.min(errors)),
        "Median Error": float(np.median(errors)),
    }


def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    return df.assign(**{c: df[c].round(decimals) for c in numeric_cols})


# ===========================================================================
# Ensemble weighting (shared by both modes)
# ===========================================================================
def _weighted_row(row, weights, group_models) -> float:
    """Weighted average of the available model predictions in a single row."""
    num = den = 0.0
    for model in group_models:
        value = row.get(model)
        if value is not None and pd.notna(value):
            num += value * weights[model]
            den += weights[model]
    return num / den if den > 0 else np.nan


def _ensemble_arrays(pivot, weights, group_models, true_age_dict):
    """Return aligned (y_true, y_pred) arrays for an ensemble over a pivot table."""
    df = pivot.copy()
    df["Ensemble"] = df.apply(lambda r: _weighted_row(r, weights, group_models), axis=1)
    df = df.dropna(subset=["Ensemble"])
    df = df[df["ImageID"].isin(true_age_dict)]
    y_true = np.array([true_age_dict[i] for i in df["ImageID"]], dtype=float)
    return y_true, df["Ensemble"].to_numpy(dtype=float)


def individual_maes(pivot, true_age_dict, models=MODEL_NAMES) -> dict:
    """Per-model MAE over a pivot table (used to rank models and weight them)."""
    maes = {}
    for model in models:
        if model not in pivot.columns:
            continue
        sub = pivot[["ImageID", model]].dropna()
        sub = sub[sub["ImageID"].isin(true_age_dict)]
        maes[model] = mean_absolute_error([true_age_dict[i] for i in sub["ImageID"]], sub[model])
    return maes


def mae_based_weights(group_models, model_maes) -> dict:
    """Constant weighting: weight_i = (Total - MAE_i) / ((n - 1) * Total)."""
    total = sum(model_maes[m] for m in group_models)
    n = len(group_models)
    return {m: (total - model_maes[m]) / ((n - 1) * total) for m in group_models}


def grid_search_weights(group_models, val_pivot, true_age_dict, step=GRID_STEP):
    """Search weights summing to 1.0 (resolution ``step``) minimising VALIDATION MAE."""
    ranges = [np.arange(step, 1.0, step) for _ in group_models]
    best_weights, best_mae = None, float("inf")
    for combo in itertools.product(*ranges):
        if not np.isclose(sum(combo), 1.0, atol=1e-5):
            continue
        weights = dict(zip(group_models, combo))
        y_true, y_pred = _ensemble_arrays(val_pivot, weights, group_models, true_age_dict)
        mae = mean_absolute_error(y_true, y_pred)
        if mae < best_mae:
            best_mae, best_weights = mae, weights
    return best_weights, best_mae


def select_and_evaluate(val_pivot, test_pivot, true_age_dict):
    """Select ensemble weights on validation and evaluate on test (leakage-free).

    Returns (summary_df, val_maes). summary_df has one row per (group, method),
    sorted by test MAE. val_maes is a {model: val_MAE} dict.
    """
    val_maes = individual_maes(val_pivot, true_age_dict)
    rows = []
    for group, models in ENSEMBLE_GROUPS.items():
        grid_w, grid_val_mae = grid_search_weights(models, val_pivot, true_age_dict)
        mae_w = mae_based_weights(models, val_maes)
        val_true_m, val_pred_m = _ensemble_arrays(val_pivot, mae_w, models, true_age_dict)
        mae_based_val_mae = mean_absolute_error(val_true_m, val_pred_m)
        methods = (
            ("Grid Search", grid_w, grid_val_mae),
            ("MAE-based", mae_w, mae_based_val_mae),
        )
        for method, weights, val_mae in methods:
            y_true, y_pred = _ensemble_arrays(test_pivot, weights, models, true_age_dict)
            rows.append({
                "Ensemble Group": group,
                "Method": method,
                "Val_MAE": val_mae,
                "Weights": json.dumps({m: round(float(weights[m]), 2) for m in models}),
                **compute_metrics(y_true, y_pred),
            })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), val_maes


def individual_metrics_from_pivot(test_pivot, true_age_dict) -> dict:
    """Per-model full metric suite on the test pivot."""
    out = {}
    for model in MODEL_NAMES:
        if model not in test_pivot.columns:
            continue
        sub = test_pivot[["ImageID", model]].dropna()
        sub = sub[sub["ImageID"].isin(true_age_dict)]
        out[model] = compute_metrics([true_age_dict[i] for i in sub["ImageID"]], sub[model])
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


def true_age_dict_from_df(labels_df: pd.DataFrame) -> dict:
    return dict(zip(labels_df["File"], labels_df["Age"].astype(float)))


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
    missing = [(m, fn) for m, fn in WEIGHT_FILES.items()
               if not (weights_dir / fn).is_file()]
    if not missing:
        return weights_dir
    if ZENODO_RECORD_ID == "REPLACE_WITH_ZENODO_RECORD_ID":
        raise RuntimeError(
            "ViT weights are missing and no Zenodo record id is configured.\n"
            f"Either place the checkpoints at {weights_dir}/<Model>_best_model.keras\n"
            "or set ZENODO_RECORD_ID in this script (or the HHD_AGE_VIT_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )
    print(f"Downloading {len(missing)} checkpoint(s) from Zenodo record {ZENODO_RECORD_ID}:")
    weights_dir.mkdir(parents=True, exist_ok=True)
    for _m, fn in missing:
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fn}?download=1"
        _download(url, weights_dir / fn, WEIGHT_MD5.get(fn))
    return weights_dir


# ===========================================================================
# Full mode: image -> patch -> per-image prediction pipeline (TensorFlow)
# ===========================================================================
def calculate_resized_dimensions(height, width, patch_size=PATCH_SIZE[0],
                                 stride=STRIDE, standard_size=STANDARD_SIZE):
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


def run_full_inference(labels_df, data_dir, weights_dir):
    """Run patch-level inference for every ViT checkpoint on val + test splits.

    Returns (val_pivot, test_pivot) DataFrames with ImageID + one column per model.
    """
    os.environ["TF_USE_LEGACY_KERAS"] = "1"  # must precede `import tensorflow`
    import tensorflow as tf
    import tf_keras  # noqa: F401  (fail loudly if the legacy Keras pkg is missing)
    import keras_cv_attention_models  # noqa: F401  (registers custom layers)
    from PIL import Image

    data_dir = Path(data_dir)
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    def read_image_and_resize(img_path):
        path = img_path.numpy().decode("utf-8")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
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
        # Drop near-blank (background) patches, matching the training pipeline.
        patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
        patches = tf.boolean_mask(patches, patch_means > THR)
        patches = tf.image.resize(patches, [final_size, final_size],
                                  method=tf.image.ResizeMethod.BICUBIC)
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, ids

    def make_dataset(split_df, final_size):
        ds = tf.data.Dataset.from_tensor_slices(dict(split_df))
        ds = ds.map(lambda r: process_row(r, str(data_dir), final_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda p, i: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(p),
            tf.data.Dataset.from_tensor_slices(i),
        )))
        return ds.batch(INFERENCE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def run_split(split_name):
        split_df = labels_df[labels_df["Set"] == split_name].reset_index(drop=True)
        per_model = {}
        for name in MODEL_NAMES:
            ckpt = Path(weights_dir) / WEIGHT_FILES[name]
            if not ckpt.is_file():
                print(f"  {name}: checkpoint missing -- skipping")
                continue
            print(f"  {name}: loading {ckpt.name}")
            model = tf.keras.models.load_model(ckpt, compile=False)
            preds_per_image = defaultdict(list)
            for patches, ids in make_dataset(split_df, MODEL_INPUT_SIZES[name]):
                p = model.predict(patches, verbose=0).ravel()
                for value, iid in zip(p, ids.numpy()):
                    preds_per_image[iid.decode("utf-8")].append(float(value))
            per_model[name] = {iid: float(np.mean(v)) for iid, v in preds_per_image.items()}
            del model
            tf.keras.backend.clear_session()

        all_ids = sorted(set().union(*[set(d) for d in per_model.values()])) if per_model else []
        out = pd.DataFrame({"ImageID": all_ids})
        for name in MODEL_NAMES:
            if name in per_model:
                out[name] = [per_model[name].get(iid, np.nan) for iid in all_ids]
        return out

    print("Running inference on validation split...")
    val_pivot = run_split("val")
    print("Running inference on test split...")
    test_pivot = run_split("test")
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
            computed_val = individual_metrics[model].get(metric)
            if computed_val is None:
                continue
            expected_val = expected_model.get(metric)
            if expected_val is not None:
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
            if metric not in row:
                continue
            computed_val = float(row[metric])
            expected_val = expected_ens.get(metric)
            if expected_val is not None:
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
        description="Reproduce the ViT Ensemble results of Experiment 04.",
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
                        help="Where ViT checkpoints are cached/downloaded (full mode).")
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
    true_age_dict = true_age_dict_from_df(labels_df)

    if mode == "full":
        weights_dir = ensure_weights(args.weights_dir)
        val_pivot, test_pivot = run_full_inference(labels_df, args.data_dir, weights_dir)
        # Persist the freshly computed pivots so future runs can use the fast path.
        val_pivot.to_csv(output_dir / "val_image_predictions.csv", index=False)
        test_pivot.to_csv(output_dir / "test_image_predictions.csv", index=False)
    else:
        if not csvs_available:
            raise FileNotFoundError(
                f"Fast mode needs {VAL_PREDICTIONS_CSV} and {TEST_PREDICTIONS_CSV}. "
                "Run with --mode full to regenerate them from the model weights."
            )
        val_pivot = pd.read_csv(VAL_PREDICTIONS_CSV)
        test_pivot = pd.read_csv(TEST_PREDICTIONS_CSV)

    individual_metrics = individual_metrics_from_pivot(test_pivot, true_age_dict)
    summary, val_maes = select_and_evaluate(val_pivot, test_pivot, true_age_dict)

    # --- Individual model metrics CSV (test set) ---
    ind_rows = [{"Model": m, **{k: round(v, 2) for k, v in individual_metrics[m].items()}}
                for m in MODEL_NAMES if m in individual_metrics]
    ind_metrics_path = output_dir / "individual_model_metrics.csv"
    pd.DataFrame(ind_rows).to_csv(ind_metrics_path, index=False)

    # --- Validation MAE per model ---
    val_mae_path = output_dir / "val_mae_per_model.csv"
    pd.DataFrame([{"Model": m, "Val_MAE": round(val_maes[m], 2)}
                  for m in MODEL_NAMES if m in val_maes]).to_csv(val_mae_path, index=False)

    # --- Ensemble metrics CSV ---
    ensemble_path = output_dir / "ensemble_metrics.csv"
    _round_df(summary).to_csv(ensemble_path, index=False)

    # --- Verification CSV (computed vs. results.md) ---
    all_pass, verification_df = build_verification(summary, individual_metrics)
    verification_path = output_dir / "verification.csv"
    verification_df.to_csv(verification_path, index=False)

    # --- Print summary ---
    print("\n" + "=" * 78)
    print("ViT REPRODUCTION SUMMARY  vs  results.md")
    print("=" * 78)

    print("\n  Individual Models (test set):")
    print(f"    {'Model':<18} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'MAPE':>7} {'±2%':>7} {'±5%':>7} {'Max':>7} {'Min':>7} {'Med':>7}")
    for m in MODEL_NAMES:
        if m in individual_metrics:
            mt = individual_metrics[m]
            print(f"    {m:<18} {mt['MAE']:7.2f} {mt['RMSE']:7.2f} {mt['R2']:7.2f} {mt['MAPE (%)']:7.2f} {mt['Within ±2 Years (%)']:7.2f} {mt['Within ±5 Years (%)']:7.2f} {mt['Max Error']:7.2f} {mt['Min Error']:7.2f} {mt['Median Error']:7.2f}")

    print("\n  Ensemble Configurations (weights selected on val, evaluated on test):")
    print(f"    {'Group':<16} {'Method':<12} {'Val_MAE':>8} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'MAPE':>7} {'±2%':>7} {'±5%':>7}")
    for _, row in summary.iterrows():
        print(f"    {row['Ensemble Group']:<16} {row['Method']:<12} {row['Val_MAE']:8.2f} {row['MAE']:7.2f} {row['RMSE']:7.2f} {row['R2']:7.2f} {row['MAPE (%)']:7.2f} {row['Within ±2 Years (%)']:7.2f} {row['Within ±5 Years (%)']:7.2f}")

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n{status} — outputs in {output_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
