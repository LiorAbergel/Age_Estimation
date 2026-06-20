"""SOTA CNN ensemble with advanced augmentation (paper Table 3, top).

Trains five ImageNet-pretrained CNN backbones as age regressors on offline
handwriting patches, caches their patch-level predictions, and combines the
best-performing models into weighted ensembles. Ensemble weights are selected
on the validation split and evaluated on the test split (leakage-free), using
two weighting schemes: grid search and inverse-MAE constant weighting.

Pipeline
--------
1. Resize each page to 800 px (aspect ratio preserved) and tile into 400x400
   patches with stride 200; normalise pixels to [0, 1].
2. Train each backbone in two phases: frozen backbone, then fine-tuning.
3. Cache per-patch predictions for the validation and test splits.
4. Average patches into per-image predictions; select ensemble weights on the
   validation split and evaluate on the test split.

Usage (run from the repository root):
    python 01_CNN_Ensemble/train_cnn_ensemble.py
    python 01_CNN_Ensemble/train_cnn_ensemble.py --batch-size 32
    python 01_CNN_Ensemble/train_cnn_ensemble.py --help

The resulting weights reproduce results.md and the committed predictions/ CSVs
consumed by reproduce_results.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, InceptionV3, InceptionResNetV2, DenseNet121
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M

EXPERIMENT_DIRNAME = "01_CNN_Ensemble"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import download_dataset`
from download_dataset import ensure_dataset

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Benign Keras 3 false-positive: with an unknown-cardinality tf.data pipeline
# (flat_map yields a variable number of patches per image), Keras cannot
# pre-compute steps and warns at end-of-dataset even though every epoch runs to
# completion. Silence just this message to keep the training log readable.
warnings.filterwarnings("ignore", message="Your input ran out of data")

# ===========================================================================
# Configuration (defaults; override on the command line)
# ===========================================================================
PATCH_SIZE = (400, 400)
STRIDE = 200
STANDARD_SIZE = 800
THR = 0.0054             # mean-intensity threshold; near-blank patches are dropped
BATCH_SIZE = 128          # used for the published results; lower if VRAM-limited
EPOCHS_FROZEN = 50
EPOCHS_FINE_TUNE = 10
FROZEN_LR = 1e-3
FINE_TUNE_LR = 1e-4
DROPOUT_RATE = 0.5

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "experiment_01"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "experiment_01"

# Backbones, ordered as trained.
ARCHITECTURES = {
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "EfficientNetV2M": EfficientNetV2M,
}
MODEL_NAMES = list(ARCHITECTURES)

# Each ensemble combines the k best individual models (ranked by validation MAE).
ENSEMBLE_GROUPS = {
    "Full Ensemble": ["ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet121", "EfficientNetV2M"],
    "Best 4": ["ResNet50", "InceptionResNetV2", "DenseNet121", "InceptionV3"],
    "Best 3": ["ResNet50", "InceptionResNetV2", "DenseNet121"],
    "Best 2": ["ResNet50", "InceptionResNetV2"],
}
GRID_STEP = 0.1

# Reported test MAEs from results.md (for a sanity comparison after training).
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


# ===========================================================================
# Data pipeline
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


def read_image_and_resize(img_path):
    """Read an image, resize per ``calculate_resized_dimensions``, normalise to [0, 1]."""
    try:
        path = img_path.numpy().decode("utf-8")
        img = Image.open(path).convert("RGB")
        width, height = img.size
        new_h, new_w = calculate_resized_dimensions(height, width)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((PATCH_SIZE[0], PATCH_SIZE[1], 3), dtype=np.float32)


def process_image(row, data_dir, include_id=False):
    """Load one page, extract patches, and broadcast its label (and id) to all patches."""
    img_path = tf.strings.join([data_dir, row["File"]], separator=os.sep)
    img = tf.py_function(func=read_image_and_resize, inp=[img_path], Tout=tf.float32)
    img.set_shape([None, None, 3])

    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
        strides=[1, STRIDE, STRIDE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, [-1, PATCH_SIZE[0], PATCH_SIZE[1], 3])
    labels = tf.fill([tf.shape(patches)[0]], row["Age"])

    # Drop near-blank (background) patches by mean intensity, matching the
    # cross-validation and DiT pipelines so preprocessing is uniform across
    # experiments. Affects both training and (id-carrying) inference datasets.
    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > THR
    patches = tf.boolean_mask(patches, mask)
    labels = tf.boolean_mask(labels, mask)

    if include_id:
        ids = tf.fill([tf.shape(patches)[0]], row["File"])
        return patches, labels, ids
    return patches, labels


# Pre-instantiate the rotation layer once to avoid tf.function retracing.
rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)  # +/- 15 degrees


def advanced_augmentation(image, label):
    """Random rotation (+/-15 deg), zoom (0.9-1.1), brightness, contrast, and Gaussian noise."""
    image = rotation_layer(image, training=True)

    orig_shape = tf.shape(image)[:2]
    zoom_factor = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, label


def create_dataset(data_dir, labels_df, split, batch_size, augment=False, include_id=False):
    """Build a tf.data pipeline of ``(patch, label[, id])`` for one split."""
    subset = labels_df[labels_df["Set"] == split].reset_index(drop=True)
    target_dir = os.path.join(data_dir, split)
    ds = tf.data.Dataset.from_tensor_slices(dict(subset))

    ds = ds.map(lambda row: process_image(row, target_dir, include_id=include_id),
                num_parallel_calls=tf.data.AUTOTUNE)

    if include_id:
        ds = ds.flat_map(lambda patches, labels, ids: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels),
            tf.data.Dataset.from_tensor_slices(ids),
        )))
    else:
        ds = ds.flat_map(lambda patches, labels: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels),
        )))

    if augment:
        ds = ds.map(advanced_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ===========================================================================
# Model
# ===========================================================================
def build_sota_model(base_model_fn, input_shape=(*PATCH_SIZE, 3), dropout_rate=DROPOUT_RATE):
    """ImageNet backbone -> GlobalAveragePooling -> Dropout -> linear regression head."""
    base_model = base_model_fn(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="linear")(x)
    return Model(inputs, outputs), base_model


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(y_true, y_pred) -> dict:
    """Full metric suite reported in the paper for one prediction set."""
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
# Training
# ===========================================================================
class BestModelLogger(tf.keras.callbacks.Callback):
    """Print a single line whenever the monitored metric improves and is saved.

    Mirrors ModelCheckpoint's improvement criterion (strictly lower ``val_mae``)
    so the message aligns with the actual save, without Keras 3's duplicated
    "improved ... saving" / "finished saving" output.
    """

    def __init__(self, save_path, monitor="val_mae"):
        super().__init__()
        self.save_path = str(save_path)
        self.monitor = monitor
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        if self.best is None or current < self.best:
            prev = "inf" if self.best is None else f"{self.best:.5f}"
            print(f"Epoch {epoch + 1}: {self.monitor} improved from {prev} to "
                  f"{current:.5f}; saved model to {self.save_path}")
            self.best = current


class EpochCSVLogger(tf.keras.callbacks.Callback):
    """Append one row per epoch to a CSV, flushing immediately.

    Saved alongside the trained model so the curves travel with the weights
    (e.g. to Google Drive on Colab) and survive a mid-training crash. Columns:
    ``model, phase, epoch`` followed by every Keras log metric. Set
    ``overwrite=True`` for the first phase and ``False`` to append later phases.
    """

    def __init__(self, log_path, model_name, phase, overwrite):
        super().__init__()
        self.log_path = str(log_path)
        self.model_name = model_name
        self.phase = phase
        self.overwrite = overwrite
        self._fieldnames = None
        self._fh = None
        self._writer = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {"model": self.model_name, "phase": self.phase, "epoch": epoch + 1}
        row.update({key: float(value) for key, value in logs.items()})

        if self._writer is None:
            if not self.overwrite and os.path.exists(self.log_path):
                with open(self.log_path, newline="") as fh:
                    self._fieldnames = next(csv.reader(fh), None) or list(row.keys())
                mode, write_header = "a", False
            else:
                self._fieldnames = list(row.keys())
                mode, write_header = "w", True
            self._fh = open(self.log_path, mode, newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames,
                                          extrasaction="ignore", restval="")
            if write_header:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._fh.flush()

    def on_train_end(self, logs=None):
        if self._fh is not None:
            self._fh.close()
            self._fh = self._writer = None


def train_models(train_ds, val_ds, models_dir, epochs_frozen, epochs_fine_tune) -> list:
    """Train (or load) each backbone: a frozen phase followed by fine-tuning.

    Returns the list of available backbone names. The best weights live on disk
    (see ModelCheckpoint), so trained models are NOT kept resident in GPU
    memory: each is freed before the next backbone and reloaded one-at-a-time
    for inference.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    available = []

    for name, architecture in ARCHITECTURES.items():
        print(f"\n{'=' * 30}\n{name}\n{'=' * 30}")
        save_path = models_dir / f"{name}_best_model.keras"
        done_path = models_dir / f"{name}.done"
        # Skip only when training fully completed: the .done marker is written
        # after Phase 2. The .keras checkpoint alone is not enough, since
        # ModelCheckpoint saves it as early as epoch 1 (so a crashed run leaves a
        # partial checkpoint that must NOT be treated as finished).
        if done_path.is_file() and save_path.is_file():
            print(f"Found completed {name}; will load from disk for inference.")
            available.append(name)
            continue
        if save_path.is_file():
            print(f"Found incomplete checkpoint for {name} (no '{done_path.name}' "
                  f"marker); retraining from scratch.")

        model, base_model = build_sota_model(architecture)
        log_path = models_dir / f"{name}_training_log.csv"
        callbacks = [
            ModelCheckpoint(str(save_path), monitor="val_mae", save_best_only=True, mode="min", verbose=0),
            BestModelLogger(save_path, monitor="val_mae"),
            ReduceLROnPlateau(monitor="val_mae", factor=0.1, patience=5, verbose=1),
            EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True, verbose=1),
        ]

        print("Phase 1: frozen backbone")
        model.compile(optimizer=tf.keras.optimizers.Adam(FROZEN_LR), loss="mse", metrics=["mae"])
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen, verbose=2,
                  callbacks=callbacks + [EpochCSVLogger(log_path, name, "frozen", overwrite=True)])

        print("Phase 2: fine-tuning")
        base_model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR), loss="mse", metrics=["mae"])
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_fine_tune, verbose=2,
                  callbacks=callbacks + [EpochCSVLogger(log_path, name, "fine_tune", overwrite=False)])
        print(f"Saved training log to {log_path}")

        # Mark training as fully complete so reruns skip this backbone.
        done_path.write_text(f"{name} training complete\n")
        available.append(name)

        # Release this backbone before training the next one: the best weights
        # are safely on disk, so free GPU/host memory and reset the Keras
        # session to keep only a single model resident at a time.
        del model, base_model
        tf.keras.backend.clear_session()
        gc.collect()
    return available


# ===========================================================================
# Inference / prediction caching
# ===========================================================================
def cache_patch_predictions(model_names, models_dir, dataset, csv_path) -> pd.DataFrame:
    """Write per-patch predictions to ``csv_path`` (resumable); return the DataFrame.

    Models are loaded from disk one at a time and freed afterwards, so only a
    single backbone is resident in GPU memory during inference.
    """
    csv_path = Path(csv_path)
    if csv_path.is_file():
        print(f"Predictions exist at {csv_path}; skipping inference.")
        return pd.read_csv(csv_path)

    models_dir = Path(models_dir)
    print(f"Generating patch-level predictions -> {csv_path}")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "ImageID", "PatchIndex", "Prediction"])
        for name in model_names:
            model = load_model(models_dir / f"{name}_best_model.keras")
            patch_idx = 0
            for patches, _, ids in tqdm(dataset, desc=name):
                preds = model.predict(patches, verbose=0).flatten()
                for i, pred in enumerate(preds):
                    writer.writerow([name, ids.numpy()[i].decode("utf-8"), patch_idx + i, pred])
                patch_idx += len(preds)
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    return pd.read_csv(csv_path)


def image_level_pivot(patch_df) -> pd.DataFrame:
    """Average patch predictions to per-image and return an ImageID x model pivot."""
    means = patch_df.groupby(["Model", "ImageID"])["Prediction"].mean().reset_index()
    return means.pivot(index="ImageID", columns="Model", values="Prediction").reset_index()


# ===========================================================================
# Ensemble (weights selected on val, evaluated on test)
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
    """Select ensemble weights on validation and evaluate on test (leakage-free)."""
    val_maes = individual_maes(val_pivot, true_age_dict)
    print("\nValidation MAE per model:")
    for model in MODEL_NAMES:
        if model in val_maes:
            print(f"  {model:<20} {val_maes[model]:.4f}")

    rows = []
    for group, models in ENSEMBLE_GROUPS.items():
        grid_w, grid_val_mae = grid_search_weights(models, val_pivot, true_age_dict)
        print(f"  [{group}] grid-search best validation MAE: {grid_val_mae:.4f}")
        for method, weights in (("Grid Search", grid_w), ("MAE-based", mae_based_weights(models, val_maes))):
            y_true, y_pred = _ensemble_arrays(test_pivot, weights, models, true_age_dict)
            rows.append({
                "Ensemble Group": group,
                "Method": method,
                "Weights": {m: round(float(weights[m]), 2) for m in models},
                **compute_metrics(y_true, y_pred),
            })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def report_against_results_md(summary) -> None:
    """Print each ensemble's test MAE next to the value reported in results.md."""
    print("\nEnsemble test MAE vs. results.md:")
    print(f"  {'Group':<16}{'Method':<13}{'MAE':>7}{'reported':>10}   status")
    for _, row in summary.iterrows():
        expected = EXPECTED_ENSEMBLE_MAE.get((row["Ensemble Group"], row["Method"]))
        status = " - " if expected is None else ("PASS" if abs(row["MAE"] - expected) <= 0.02 else "FAIL")
        expected_str = "   -" if expected is None else f"{expected:>10.2f}"
        print(f"  {row['Ensemble Group']:<16}{row['Method']:<13}{row['MAE']:>7.2f}{expected_str}   {status}")


# ===========================================================================
# CLI
# ===========================================================================
def _in_colab():
    """True when running on Google Colab, whose local disk is ephemeral."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


# On Colab, persist trained models/results here so they survive runtime crashes.
COLAB_DRIVE_ROOT = Path("/content/drive")
COLAB_PERSIST_BASE = COLAB_DRIVE_ROOT / "MyDrive" / "Age_Estimation" / EXPERIMENT_DIRNAME


def _mount_drive_for_persistence():
    """Mount Google Drive (if needed) and return a persistent base directory.

    Returns ``None`` if Drive cannot be mounted, so the caller can fall back to
    the ephemeral local directory.
    """
    try:
        if not (COLAB_DRIVE_ROOT / "MyDrive").exists():
            from google.colab import drive
            print("Colab detected: mounting Google Drive to persist trained models...")
            drive.mount(str(COLAB_DRIVE_ROOT))
        if (COLAB_DRIVE_ROOT / "MyDrive").exists():
            return COLAB_PERSIST_BASE
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: could not mount Google Drive ({exc}).")
    return None


def resolve_output_dirs(models_dir, results_dir):
    """Pick where to write weights/results, persisting to Drive on Colab.

    Explicit ``--models-dir``/``--results-dir`` values (non-None) always win.
    When they are unset on Colab, redirect to mounted Google Drive so a runtime
    crash does not delete the trained models; otherwise use the repo defaults.
    """
    if _in_colab() and (models_dir is None or results_dir is None):
        base = _mount_drive_for_persistence()
        if base is not None:
            if models_dir is None:
                models_dir = str(base / "models")
            if results_dir is None:
                results_dir = str(base / "results")
            print(f"Persisting outputs to Google Drive: {base}")
        else:
            print("WARNING: Drive unavailable; trained models will be LOST if the "
                  "Colab runtime crashes.")
    if models_dir is None:
        models_dir = str(DEFAULT_MODELS_DIR)
    if results_dir is None:
        results_dir = str(DEFAULT_RESULTS_DIR)
    return models_dir, results_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train CNN backbones and evaluate weighted ensembles (paper Table 3, top).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=None,
                        help=f"Where to save/load weights (default: {DEFAULT_MODELS_DIR}; "
                             "on Colab, auto-redirected to Google Drive to survive crashes).")
    parser.add_argument("--results-dir", default=None,
                        help=f"Where to write prediction CSVs/summaries (default: {DEFAULT_RESULTS_DIR}; "
                             "on Colab, auto-redirected to Google Drive).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="128 was used for the published results; lower if VRAM-limited.")
    parser.add_argument("--epochs-frozen", type=int, default=EPOCHS_FROZEN)
    parser.add_argument("--epochs-fine-tune", type=int, default=EPOCHS_FINE_TUNE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    models_dir, results_dir = resolve_output_dirs(args.models_dir, args.results_dir)
    ensure_dataset(args.data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    labels_df = pd.read_csv(os.path.join(args.data_dir, "NewAgeSplit.csv"))
    true_age_dict = dict(zip(labels_df["File"], labels_df["Age"]))

    # Datasets
    print("Creating datasets...")
    train_ds = create_dataset(args.data_dir, labels_df, "train", args.batch_size, augment=True)
    val_ds_train = create_dataset(args.data_dir, labels_df, "val", args.batch_size, augment=False)

    # Train (or load cached weights)
    model_names = train_models(train_ds, val_ds_train, models_dir,
                               args.epochs_frozen, args.epochs_fine_tune)

    # Inference datasets carry image ids for per-image aggregation
    print("Creating inference datasets...")
    val_ds = create_dataset(args.data_dir, labels_df, "val", args.batch_size, augment=False, include_id=True)
    test_ds = create_dataset(args.data_dir, labels_df, "test", args.batch_size, augment=False, include_id=True)

    # Cache patch-level predictions, then aggregate to per-image pivots
    val_patch_df = cache_patch_predictions(model_names, models_dir, val_ds, results_dir / "val_patch_level_predictions.csv")
    test_patch_df = cache_patch_predictions(model_names, models_dir, test_ds, results_dir / "patch_level_predictions.csv")

    val_pivot = image_level_pivot(val_patch_df)
    test_pivot = image_level_pivot(test_patch_df)
    val_pivot.to_csv(results_dir / "val_image_predictions.csv", index=False)
    test_pivot.to_csv(results_dir / "test_image_predictions.csv", index=False)

    # Ensemble selection (val) and evaluation (test)
    summary = select_and_evaluate(val_pivot, test_pivot, true_age_dict)
    summary_path = results_dir / "ensemble_evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Ensemble Evaluation Summary (test set) ===")
    print(summary.to_string(index=False))
    report_against_results_md(summary)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()