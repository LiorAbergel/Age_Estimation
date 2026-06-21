"""Reproduce the Vision-Transformer results of Experiment 04 (development experiment).

Mirrors the structure of ``01_CNN_Ensemble/reproduce_results.py`` but adapted to
the ViT setup: four backbones evaluated on the single official HHD test split,
plus a simple (equal-weight) average ensemble. It offers two modes:

    fast (default)
        Recompute every reported metric from the *committed* per-image
        predictions in ``predictions/test_image_predictions.csv``. Requires no
        GPU, no model weights and no TensorFlow -- only the ground-truth ages in
        ``data/NewAgeSplit.csv`` (downloaded automatically if absent).

    full
        Download the trained ViT checkpoints from Zenodo, run the patch-level
        inference pipeline on the test split, then recompute the same metrics
        end-to-end from the raw images.

In both modes every computed value is compared against the numbers reported in
``results.md`` and a PASS/FAIL summary is printed.

Examples:
    python 04_ViT/reproduce_results.py                 # fast path
    python 04_ViT/reproduce_results.py --mode full     # from images
    python 04_ViT/reproduce_results.py --help
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
ENSEMBLE_NAME = "Ensemble (all 4)"  # equal-weight average of the four models

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr", "Acc_10yr",
               "Max_Error", "Median_Error"]

# Patch extraction (only used by the full, image-based pipeline).
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
INFERENCE_BATCH_SIZE = 64  # does not affect predictions

# ---------------------------------------------------------------------------
# Pretrained weights hosted on Zenodo (anonymous, DOI-citable download).
# After uploading the four ``*_finetune.keras`` files to a Zenodo record, set
# the integer record id below (visible in the record URL, e.g.
# https://zenodo.org/records/1234567) or export HHD_AGE_VIT_ZENODO_RECORD.
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = os.environ.get("HHD_AGE_VIT_ZENODO_RECORD", "REPLACE_WITH_ZENODO_RECORD_ID")
WEIGHT_FILES = {name: f"{name}_finetune.keras" for name in MODEL_NAMES}
# Optional integrity check: fill in {filename: md5_hex} once the files are uploaded.
WEIGHT_MD5: dict[str, str] = {}

# ===========================================================================
# Reported values (from results.md) -- used for the PASS/FAIL self-check
# ===========================================================================
EXPECTED = {
    "SwinV2_Tiny":     {"MAE": 6.98, "RMSE": 8.04, "R2": -0.62, "MAPE": 43.58,
                        "Acc_2yr": 1.72, "Acc_5yr": 18.10, "Acc_10yr": 96.55,
                        "Max_Error": 29.58, "Median_Error": 6.42},
    "MobileViT_XXS":   {"MAE": 4.42, "RMSE": 6.54, "R2": -0.07, "MAPE": 25.90,
                        "Acc_2yr": 16.38, "Acc_5yr": 74.14, "Acc_10yr": 96.55,
                        "Max_Error": 31.44, "Median_Error": 3.38},
    "ConvNeXtV2_Tiny": {"MAE": 6.30, "RMSE": 7.58, "R2": -0.44, "MAPE": 38.78,
                        "Acc_2yr": 1.72, "Acc_5yr": 41.38, "Acc_10yr": 96.55,
                        "Max_Error": 30.36, "Median_Error": 5.64},
    "TinyViT_11M":     {"MAE": 6.70, "RMSE": 7.79, "R2": -0.52, "MAPE": 41.92,
                        "Acc_2yr": 1.72, "Acc_5yr": 35.34, "Acc_10yr": 96.55,
                        "Max_Error": 28.70, "Median_Error": 5.87},
    ENSEMBLE_NAME:     {"MAE": 6.09, "RMSE": 7.42, "R2": -0.38, "MAPE": 37.51,
                        "Acc_2yr": 1.72, "Acc_5yr": 41.38, "Acc_10yr": 96.55,
                        "Max_Error": 30.02, "Median_Error": 5.33},
}

# MAE is the headline metric; the |computed - reported| MAE gap must stay within
# this band to PASS. (This is a development experiment that you re-run to
# regenerate predictions/, so a modest tolerance is expected.)
TOLERANCE = 0.15


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(y_true, y_pred) -> dict:
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
    }


# ===========================================================================
# Aggregation + reporting
# ===========================================================================
def summarize(preds_df: pd.DataFrame, true_age: dict) -> dict:
    """Return {model: metrics}. preds_df has ImageID + one column per model."""
    ids = [i for i in preds_df["ImageID"] if i in true_age]
    sub = preds_df[preds_df["ImageID"].isin(ids)].reset_index(drop=True)
    y_true = np.array([true_age[i] for i in sub["ImageID"]], dtype=float)

    summary = {}
    model_cols = [m for m in MODEL_NAMES if m in sub.columns]
    for name in model_cols:
        summary[name] = compute_metrics(y_true, sub[name].to_numpy(dtype=float))
    if len(model_cols) == len(MODEL_NAMES):
        ensemble = sub[model_cols].mean(axis=1).to_numpy(dtype=float)
        summary[ENSEMBLE_NAME] = compute_metrics(y_true, ensemble)
    return summary


def report(summary: dict) -> bool:
    all_pass = True
    print("\n" + "=" * 78)
    print("ViT REPRODUCTION SUMMARY  vs  results.md")
    print("=" * 78)
    for name in MODEL_NAMES + [ENSEMBLE_NAME]:
        agg = summary.get(name)
        if agg is None:
            print(f"\n  {name}: NO PREDICTIONS")
            all_pass = False
            continue
        expected = EXPECTED.get(name, {})
        print(f"\n  {name}:")
        print(f"    {'Metric':<14}{'computed':>12}{'reported':>12}   status")
        for k in METRIC_KEYS:
            val = agg[k]
            exp = expected.get(k)
            if exp is None:
                print(f"    {k:<14}{val:12.2f}{'-':>12}")
                continue
            # Only MAE drives PASS/FAIL; other metrics are informational.
            if k == "MAE":
                ok = abs(val - exp) <= TOLERANCE
                all_pass &= ok
                status = "PASS" if ok else "FAIL"
            else:
                status = ""
            print(f"    {k:<14}{val:12.2f}{exp:12.2f}   {status}")
    print("\nOVERALL: " + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    return all_pass


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


def test_age_dict(labels_df: pd.DataFrame) -> dict:
    test_df = labels_df[labels_df["Set"] == "test"]
    return dict(zip(test_df["File"], test_df["Age"].astype(float)))


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
            f"Either place the checkpoints at {weights_dir}/<Model>_finetune.keras\n"
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


def run_full_inference(labels_df, data_dir, weights_dir) -> pd.DataFrame:
    """Run patch-level inference for every ViT checkpoint on the test split."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    import keras_cv_attention_models  # noqa: F401  (registers custom layers)
    from PIL import Image

    data_dir = Path(data_dir)
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    test_df = labels_df[labels_df["Set"] == "test"].reset_index(drop=True)

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
        patches = tf.image.resize(patches, [final_size, final_size])  # bilinear, as in eval
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, ids

    def make_dataset(final_size):
        ds = tf.data.Dataset.from_tensor_slices(dict(test_df))
        ds = ds.map(lambda r: process_row(r, str(data_dir), final_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda p, i: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(p),
            tf.data.Dataset.from_tensor_slices(i),
        )))
        return ds.batch(INFERENCE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    per_model = {}
    for name in MODEL_NAMES:
        ckpt = Path(weights_dir) / WEIGHT_FILES[name]
        if not ckpt.is_file():
            print(f"  {name}: checkpoint missing -- skipping")
            continue
        print(f"  {name}: loading {ckpt.name}")
        model = tf.keras.models.load_model(ckpt, compile=False)
        preds_per_image = defaultdict(list)
        for patches, ids in make_dataset(MODEL_INPUT_SIZES[name]):
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


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the Vision-Transformer results of Experiment 04.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["auto", "fast", "full"], default="auto",
        help="'fast' uses the committed per-image prediction CSV (no GPU/weights); "
             "'full' downloads weights from Zenodo and reruns inference; "
             "'auto' picks 'fast' when the CSV is present.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory containing NewAgeSplit.csv (and images for full mode).")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="Where ViT checkpoints are cached/downloaded (full mode).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where recomputed predictions are written (full mode).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "fast" if TEST_PREDICTIONS_CSV.is_file() else "full"
    print(f"Mode: {mode}")

    labels_df = load_labels_df(args.data_dir)
    true_age = test_age_dict(labels_df)

    if mode == "full":
        weights_dir = ensure_weights(args.weights_dir)
        preds_df = run_full_inference(labels_df, args.data_dir, weights_dir)
        preds_df.to_csv(output_dir / "test_image_predictions.csv", index=False)
    else:
        if not TEST_PREDICTIONS_CSV.is_file():
            raise FileNotFoundError(
                f"Fast mode needs {TEST_PREDICTIONS_CSV}. Re-run train_vit.py to "
                "regenerate it, or run this script with --mode full."
            )
        preds_df = pd.read_csv(TEST_PREDICTIONS_CSV)

    summary = summarize(preds_df, true_age)
    all_pass = report(summary)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
