"""Reproduce the CNN cross-validation results of Experiment 03 (paper CV table).

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
in results.md and a PASS/FAIL summary is printed.

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
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
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

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr", "Acc_10yr",
               "Max_Error", "Median_Error", "Min_Error"]

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check.
# NOTE: these are the pre-unification numbers. After re-running training under
# the unified augmentation (brightness + contrast), refresh both results.md and
# this table so the self-check stays meaningful.
# ===========================================================================
EXPECTED_CV = {
    "ResNet50":          {"MAE": (5.41, 0.78), "RMSE": (8.17, 0.58), "R2": (0.10, 0.06), "MAPE": (25.68, 3.73),
                          "Acc_2yr": (23.72, 5.17), "Acc_5yr": (63.40, 11.26), "Acc_10yr": (90.10, 4.42),
                          "Max_Error": (32.99, 7.14), "Median_Error": (3.72, 0.88)},
    "DenseNet121":       {"MAE": (5.46, 1.06), "RMSE": (8.16, 0.56), "R2": (0.11, 0.06), "MAPE": (26.25, 5.47),
                          "Acc_2yr": (21.49, 16.51), "Acc_5yr": (61.61, 16.41), "Acc_10yr": (91.18, 3.84),
                          "Max_Error": (34.14, 6.66), "Median_Error": (3.94, 1.33)},
    "InceptionV3":       {"MAE": (6.03, 0.64), "RMSE": (8.41, 0.64), "R2": (0.05, 0.05), "MAPE": (29.97, 3.17),
                          "Acc_2yr": (16.40, 4.30), "Acc_5yr": (52.83, 9.01), "Acc_10yr": (90.11, 4.84),
                          "Max_Error": (32.70, 7.60), "Median_Error": (4.80, 0.67)},
    "InceptionResNetV2": {"MAE": (5.69, 0.70), "RMSE": (7.98, 0.68), "R2": (0.17, 0.11), "MAPE": (29.08, 3.16),
                          "Acc_2yr": (16.76, 3.99), "Acc_5yr": (58.32, 9.19), "Acc_10yr": (89.32, 4.58),
                          "Max_Error": (32.62, 5.80), "Median_Error": (4.37, 0.59)},
    "EfficientNetV2M":   {"MAE": (7.30, 0.28), "RMSE": (9.03, 0.36), "R2": (-0.07, 0.14), "MAPE": (41.48, 5.73),
                          "Acc_2yr": (11.58, 4.78), "Acc_5yr": (29.47, 2.21), "Acc_10yr": (84.32, 7.19),
                          "Max_Error": (31.44, 5.01), "Median_Error": (7.05, 0.68)},
}
TOLERANCE = 0.15  # |computed_mean - reported_mean| must not exceed this to PASS


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
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape,
        "Acc_2yr": float(np.mean(errors <= 2) * 100),
        "Acc_5yr": float(np.mean(errors <= 5) * 100),
        "Acc_10yr": float(np.mean(errors <= 10) * 100),
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }


# ===========================================================================
# Aggregation + reporting (shared by both modes)
# ===========================================================================
def summarize_oof(oof_df: pd.DataFrame) -> dict:
    """Return {model: {metric: (mean, std)}} computed per fold then aggregated."""
    summary = {}
    for model_name in MODEL_NAMES:
        sub = oof_df[oof_df["Model"] == model_name]
        if sub.empty:
            continue
        per_fold = [compute_metrics(fold_df["TrueAge"], fold_df["Prediction"])
                    for _, fold_df in sub.groupby("Fold")]
        summary[model_name] = {
            k: (float(np.mean([m[k] for m in per_fold])),
                float(np.std([m[k] for m in per_fold])))
            for k in METRIC_KEYS
        }
    return summary


def report(summary: dict) -> bool:
    """Print the comparison against results.md; return True iff everything passes."""
    all_pass = True
    print("\n" + "=" * 78)
    print("CV REPRODUCTION SUMMARY (mean +/- std across folds)  vs  results.md")
    print("=" * 78)
    for model_name in MODEL_NAMES:
        agg = summary.get(model_name)
        if agg is None:
            print(f"\n  {model_name}: NO PREDICTIONS")
            all_pass = False
            continue
        expected = EXPECTED_CV.get(model_name, {})
        print(f"\n  {model_name}:")
        print(f"    {'Metric':<14}{'computed':>18}{'reported':>18}   status")
        for k in METRIC_KEYS:
            mean_val, std_val = agg[k]
            exp = expected.get(k)
            if exp is None:
                print(f"    {k:<14}{mean_val:10.2f} +/- {std_val:5.2f}{'-':>18}")
                continue
            exp_mean, exp_std = exp
            ok = abs(mean_val - exp_mean) <= TOLERANCE
            all_pass &= ok
            print(f"    {k:<14}{mean_val:10.2f} +/- {std_val:5.2f}"
                  f"{exp_mean:11.2f} +/- {exp_std:4.2f}   {'PASS' if ok else 'FAIL'}")
    print("\nOVERALL: " + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    return all_pass


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

    print(f"Downloading {len(missing)} checkpoint(s) from Zenodo record {ZENODO_RECORD_ID}:")
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
                                    "Prediction": float(np.mean(plist)),
                                    "TrueAge": float(true_age_dict[iid])})
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
                        help="Where the recomputed OOF predictions are written (full mode).")
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
        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    else:
        if not OOF_PREDICTIONS_CSV.is_file():
            raise FileNotFoundError(
                f"Fast mode needs {OOF_PREDICTIONS_CSV}. Run with --mode full to "
                "regenerate it from the model weights."
            )
        oof_df = pd.read_csv(OOF_PREDICTIONS_CSV)

    summary = summarize_oof(oof_df)
    all_pass = report(summary)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
