"""Reproduce the CNN-ensemble results of Experiment 01 (paper Table 3, top).

This is the reference reproduction script for the repository; the remaining
experiments follow the same structure. It offers two modes:

    fast (default)
        Recompute the full ensemble selection and every reported metric from
        the *committed* per-image predictions in ``predictions/``. Requires no
        GPU, no model weights and no TensorFlow -- only the ground-truth ages
        in ``data/NewAgeSplit.csv`` (downloaded automatically if absent).

    full
        Download the trained model weights from Zenodo, run the patch-level
        inference pipeline on the validation and test splits, then perform the
        identical ensemble selection. Reproduces the numbers end-to-end from
        the raw images.

In both modes every computed value is compared against the numbers reported in
``results.md`` and a PASS/FAIL summary is printed.

Examples:
    python 01_CNN_Ensemble/reproduce_results.py                 # fast path
    python 01_CNN_Ensemble/reproduce_results.py --mode full     # from images
    python 01_CNN_Ensemble/reproduce_results.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
MODEL_NAMES = ["ResNet50", "DenseNet121", "InceptionV3", "InceptionResNetV2", "EfficientNetV2M"]

# Each ensemble combines the k best individual models (ranked by validation MAE).
ENSEMBLE_GROUPS = {
    "Full Ensemble": ["ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet121", "EfficientNetV2M"],
    "Best 4": ["ResNet50", "InceptionResNetV2", "DenseNet121", "InceptionV3"],
    "Best 3": ["ResNet50", "InceptionResNetV2", "DenseNet121"],
    "Best 2": ["ResNet50", "InceptionResNetV2"],
}
GRID_STEP = 0.1  # weight resolution for the grid search

# Patch extraction (only used by the full, image-based pipeline).
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
INFERENCE_BATCH_SIZE = 32  # does not affect predictions

# ---------------------------------------------------------------------------
# Pretrained weights hosted on Zenodo (anonymous, DOI-citable download).
# After uploading the five ``*_best_model.keras`` files to a Zenodo record,
# set the integer record id below (visible in the record URL, e.g.
# https://zenodo.org/records/1234567) or export HHD_AGE_ZENODO_RECORD.
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_ZENODO_RECORD", "REPLACE_WITH_ZENODO_RECORD_ID")
WEIGHT_FILES = {name: f"{name}_best_model.keras" for name in MODEL_NAMES}
# Optional integrity check: fill in {filename: md5_hex} once the files are uploaded.
WEIGHT_MD5: dict[str, str] = {}

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check
# ===========================================================================
EXPECTED_INDIVIDUAL_MAE = {
    "ResNet50": 3.23,
    "InceptionResNetV2": 3.64,
    "DenseNet121": 3.97,
    "InceptionV3": 5.06,
    "EfficientNetV2M": 7.17,
}
EXPECTED_ENSEMBLE_MAE = {
    ("Best 3", "MAE-based"): 2.86,
    ("Best 3", "Grid Search"): 2.86,
    ("Best 4", "Grid Search"): 2.93,
    ("Full Ensemble", "Grid Search"): 3.01,
    ("Best 4", "MAE-based"): 3.14,
    ("Best 2", "Grid Search"): 3.24,
    ("Best 2", "MAE-based"): 3.35,
    ("Full Ensemble", "MAE-based"): 3.77,
}
MAE_TOLERANCE = 0.02  # |computed - reported| must not exceed this to PASS


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
        "Within ±10 Years (%)": float(np.mean(errors <= 10) * 100),
        "Max Error": float(np.max(errors)),
        "Median Error": float(np.median(errors)),
    }


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


def select_and_evaluate(val_pivot, test_pivot, true_age_dict) -> pd.DataFrame:
    """Select ensemble weights on validation and evaluate on test (leakage-free).

    Returns one row per (group, weighting method), sorted by test MAE.
    """
    val_maes = individual_maes(val_pivot, true_age_dict)
    print("\nValidation MAE per model (ranking / MAE-based weights):")
    for model in MODEL_NAMES:
        if model in val_maes:
            print(f"  {model:<20} {val_maes[model]:.4f}")

    rows = []
    for group, models in ENSEMBLE_GROUPS.items():
        grid_w, grid_val_mae = grid_search_weights(models, val_pivot, true_age_dict)
        print(f"  [{group}] grid-search best validation MAE: {grid_val_mae:.4f}")
        methods = (("Grid Search", grid_w), ("MAE-based", mae_based_weights(models, val_maes)))
        for method, weights in methods:
            y_true, y_pred = _ensemble_arrays(test_pivot, weights, models, true_age_dict)
            rows.append({
                "Ensemble Group": group,
                "Method": method,
                "Weights": {m: round(float(weights[m]), 2) for m in models},
                **compute_metrics(y_true, y_pred),
            })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


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
    """Load NewAgeSplit.csv, downloading the dataset via kagglehub if absent."""
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
    """Ensure all model weights are present locally, downloading from Zenodo if needed."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    missing = [(name, fname) for name, fname in WEIGHT_FILES.items()
               if not (weights_dir / fname).is_file()]
    if not missing:
        return weights_dir

    if ZENODO_RECORD_ID == "REPLACE_WITH_ZENODO_RECORD_ID":
        raise RuntimeError(
            "Model weights are missing and no Zenodo record id is configured.\n"
            f"Either place the following files in {weights_dir}:\n"
            + "\n".join(f"  - {fname}" for _, fname in missing)
            + "\nor set ZENODO_RECORD_ID in this script (or the HHD_AGE_ZENODO_RECORD "
            "environment variable) to enable automatic download."
        )

    print(f"Downloading {len(missing)} weight file(s) from Zenodo record {ZENODO_RECORD_ID}:")
    for _, fname in missing:
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fname}?download=1"
        _download(url, weights_dir / fname, WEIGHT_MD5.get(fname))
    return weights_dir


# ===========================================================================
# Full mode: image -> patch -> prediction pipeline (TensorFlow)
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


def run_full_inference(labels_df, data_dir, weights_dir):
    """Run patch-level inference for every model; return (val_pivot, test_pivot, metrics)."""
    import tensorflow as tf
    from PIL import Image

    data_dir = Path(data_dir)
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

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

    def process_image(row, target_dir):
        img_path = tf.strings.join([target_dir, row["File"]], separator="/")
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
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, ids

    def make_dataset(split):
        subset = labels_df[labels_df["Set"] == split].reset_index(drop=True)
        target_dir = str(data_dir / split)
        ds = tf.data.Dataset.from_tensor_slices(dict(subset))
        ds = ds.map(lambda row: process_image(row, target_dir), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda patches, ids: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(ids),
        )))
        return ds.batch(INFERENCE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def infer_image_means(model, dataset):
        grouped = defaultdict(list)
        for patches, ids in dataset:
            preds = model.predict(patches, verbose=0).flatten()
            for pred, image_id in zip(preds, ids.numpy()):
                grouped[image_id.decode("utf-8")].append(float(pred))
        return {image_id: float(np.mean(vals)) for image_id, vals in grouped.items()}

    val_ds, test_ds = make_dataset("val"), make_dataset("test")
    val_preds, test_preds = {}, {}
    for name in MODEL_NAMES:
        model_path = Path(weights_dir) / WEIGHT_FILES[name]
        print(f"\n[{name}] loading {model_path.name} and running inference...")
        model = tf.keras.models.load_model(model_path)
        val_preds[name] = infer_image_means(model, val_ds)
        test_preds[name] = infer_image_means(model, test_ds)

    val_pivot = _pivot_from_predictions(val_preds)
    test_pivot = _pivot_from_predictions(test_preds)
    true_age_dict = dict(zip(labels_df["File"], labels_df["Age"]))
    metrics = individual_metrics_from_pivot(test_pivot, true_age_dict)
    return val_pivot, test_pivot, metrics


def _pivot_from_predictions(per_model_preds) -> pd.DataFrame:
    """Build an ImageID x model pivot table from {model: {image_id: mean_pred}}."""
    image_ids = sorted(set().union(*[set(d) for d in per_model_preds.values()]))
    rows = [{"ImageID": image_id,
             **{model: per_model_preds[model].get(image_id, np.nan) for model in per_model_preds}}
            for image_id in image_ids]
    return pd.DataFrame(rows)


# ===========================================================================
# Reporting
# ===========================================================================
def report(summary, individual_metrics) -> bool:
    """Print the comparison against results.md; return True iff everything passes."""
    all_pass = True

    print("\n" + "=" * 78)
    print("REPRODUCTION SUMMARY (computed vs. results.md)")
    print("=" * 78)

    print("\nIndividual models (test set):")
    print(f"  {'Model':<20}{'MAE':>8}{'reported':>10}   status")
    for model in MODEL_NAMES:
        if model not in individual_metrics:
            continue
        mae = individual_metrics[model]["MAE"]
        expected = EXPECTED_INDIVIDUAL_MAE[model]
        ok = abs(mae - expected) <= MAE_TOLERANCE
        all_pass &= ok
        print(f"  {model:<20}{mae:>8.2f}{expected:>10.2f}   {'PASS' if ok else 'FAIL'}")

    print("\nEnsembles (weights selected on val, evaluated on test):")
    print(f"  {'Group':<16}{'Method':<13}{'MAE':>7}{'reported':>10}   status   weights")
    for _, row in summary.iterrows():
        expected = EXPECTED_ENSEMBLE_MAE.get((row["Ensemble Group"], row["Method"]))
        ok = expected is None or abs(row["MAE"] - expected) <= MAE_TOLERANCE
        all_pass &= ok
        expected_str = "   -" if expected is None else f"{expected:>10.2f}"
        status = " - " if expected is None else ("PASS" if ok else "FAIL")
        print(f"  {row['Ensemble Group']:<16}{row['Method']:<13}{row['MAE']:>7.2f}"
              f"{expected_str}   {status}   {row['Weights']}")

    best = summary.iloc[0]
    print(f"\nBest configuration: {best['Ensemble Group']} ({best['Method']}) -> MAE {best['MAE']:.2f}")
    print("\nOVERALL: " + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    return all_pass


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the CNN-ensemble results of Experiment 01 (paper Table 3, top).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["auto", "fast", "full"], default="auto",
        help="'fast' uses the committed prediction CSVs (no GPU/weights); "
             "'full' downloads weights from Zenodo and runs inference from images; "
             "'auto' picks 'fast' when the CSVs are present.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory containing NewAgeSplit.csv (and train/val/test images for full mode).")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="Where model weights are cached/downloaded (full mode).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where the reproduction summary CSV is written.")
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
    true_age_dict = dict(zip(labels_df["File"], labels_df["Age"]))

    if mode == "full":
        weights_dir = ensure_weights(args.weights_dir)
        val_pivot, test_pivot, individual_metrics = run_full_inference(
            labels_df, args.data_dir, weights_dir)
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

    summary = select_and_evaluate(val_pivot, test_pivot, true_age_dict)

    summary_path = output_dir / "ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)

    all_pass = report(summary, individual_metrics)
    print(f"\nSummary written to {summary_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
