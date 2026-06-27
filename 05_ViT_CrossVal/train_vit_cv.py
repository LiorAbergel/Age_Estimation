"""
Experiment 05: Vision Transformers with 5-Fold Stratified Group CV

Overview:
This script trains and evaluates an ensemble of Vision Transformer (ViT) models
using 5-Fold Cross-Validation. It leverages the 'keras_cv_attention_models'
library for state-of-the-art backbones.

Key Features:
- Models: SwinV2, MobileViT, ConvNeXtV2, TinyViT.
- Input Sizes: Handles specific resolution requirements (224 vs 256).
- Strategy: Stratified Group K-Fold to prevent writer leakage.
- Patching: Includes intensity-based filtering to remove empty patches.

Requirements:
pip install keras-cv-attention-models tf-keras
"""

import os
import gc
import csv
import sys
import warnings

# --- CRITICAL: Must be set BEFORE importing TensorFlow ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import ViT architectures
try:
    import keras_cv_attention_models.swin_transformer_v2 as swin_v2
    import keras_cv_attention_models.mobilevit as mobilevit
    import keras_cv_attention_models.convnext as convnext
    import keras_cv_attention_models.tinyvit as tiny_vit
except ImportError:
    raise ImportError("Please run: pip install keras-cv-attention-models tf-keras")

# Benign Keras 3 false-positive: with an unknown-cardinality tf.data pipeline
# (flat_map yields a variable number of patches per image), Keras cannot
# pre-compute steps and warns at end-of-dataset even though every epoch runs to
# completion. Silence just this message to keep the training log readable.
warnings.filterwarnings("ignore", message="Your input ran out of data")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

EXPERIMENT_DIRNAME = "05_ViT_CrossVal"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 128,
    "THR": 0.0054,  # Intensity threshold for empty patches
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "LR_INIT": 1e-3,
    "LR_FT": 1e-4,
    "DATA_DIR": os.path.join(REPO_ROOT, "data"),
    "CSV_PATH": os.path.join(REPO_ROOT, "data", "NewAgeSplit.csv"),
    "MODELS_DIR": os.path.join(REPO_ROOT, "models", EXPERIMENT_DIRNAME),
    "RESULTS_DIR": os.path.join(REPO_ROOT, "results", EXPERIMENT_DIRNAME)
}



# Define models and their required input resolutions
VIT_MODEL_CONFIGS = {
    "SwinV2_Tiny":     (swin_v2.SwinTransformerV2Tiny_window8, 256),
    "MobileViT_XXS":   (mobilevit.MobileViT_XXS, 256),
    "ConvNeXtV2_Tiny": (convnext.ConvNeXtTiny, 224),
    "TinyViT_11M":     (tiny_vit.TinyViT_11M, 224),
}

# --- Data Processing Functions ---

def calculate_resized_dimensions(height, width, patch_size=400, stride=200, standard_size=800):
    aspect_ratio = width / height
    if height < width:
        new_height = standard_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = standard_size
        new_height = int(new_width / aspect_ratio)

    def adjust_dimension(dim):
        remainder = (dim - patch_size) % stride
        return dim if remainder == 0 else dim - remainder

    return adjust_dimension(new_height), adjust_dimension(new_width)

def read_tiff_image_with_dynamic_resize(img_path):
    try:
        img_path_str = img_path.numpy().decode("utf-8")
        img = Image.open(img_path_str).convert('RGB')
        w, h = img.size
        new_h, new_w = calculate_resized_dimensions(h, w, CONFIG["PATCH_SIZE"][0], CONFIG["STRIDE"])
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception:
        return np.zeros((CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3), dtype=np.float32)

def process_image(row, root_dir, patch_size, step_size):
    """
    Loads image, extracts patches, and filters empty ones based on intensity.
    """
    root = tf.constant(root_dir, dtype=tf.string)
    subset = row['Set']
    fname = row['File']
    img_path = tf.strings.join([root, subset, fname], separator=os.sep)

    img = tf.py_function(func=read_tiff_image_with_dynamic_resize, inp=[img_path], Tout=tf.float32)
    img.set_shape([None, None, 3])

    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, step_size, step_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [-1, patch_size[0], patch_size[1], 3])
    labels = tf.fill([tf.shape(patches)[0]], row['Age'])

    # Filter empty patches
    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > CONFIG["THR"]
    
    patches = tf.boolean_mask(patches, mask)
    labels = tf.boolean_mask(labels, mask)

    return patches, labels

def process_row_with_id(row):
    patches, labels = process_image(row, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
    image_id = tf.fill([tf.shape(patches)[0]], row['File'])
    return patches, labels, image_id

# --- Augmentation & Resizing ---

rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
    image = rotation_layer(image, training=True)
    
    # Zoom
    orig_shape = tf.shape(image)[:2]
    zoom = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    # Random Brightness & Contrast
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)

    # Gaussian Noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label

def resize_for_model(patch, label, final_size):
    """Resizes patch to ViT input requirement (e.g. 224x224)."""
    patch = tf.image.resize(patch, [final_size, final_size], method='bicubic')
    return patch, label

def resize_for_model_with_id(patch, label, img_id, final_size):
    patch = tf.image.resize(patch, [final_size, final_size], method='bicubic')
    return patch, label, img_id

# --- Dataset Generators ---

def patch_data_tf_dataset_from_df(labels_df_subset, data_dir, patch_size, step_size, batch_size, augment=False, final_size=None):
    ds = tf.data.Dataset.from_tensor_slices(dict(labels_df_subset))
    ds = ds.map(
        lambda row: process_image(row, data_dir, patch_size, step_size),
        num_parallel_calls=tf.data.AUTOTUNE
    ).flat_map(
        lambda patches, labels: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels)
        ))
    )
    
    if augment:
        ds = ds.map(advanced_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        
    if final_size:
        ds = ds.map(lambda p, l: resize_for_model(p, l, final_size), num_parallel_calls=tf.data.AUTOTUNE)
        
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def patch_data_tf_dataset_with_ids(labels_df_subset, final_size, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(dict(labels_df_subset))
    ds = ds.map(process_row_with_id, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(lambda p, l, i: tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(p),
        tf.data.Dataset.from_tensor_slices(l),
        tf.data.Dataset.from_tensor_slices(i)
    )))
    # Resize specifically for ViT
    ds = ds.map(lambda p, l, i: resize_for_model_with_id(p, l, i, final_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Model Building ---

def build_backbone_regressor(backbone_fn, input_size, dropout=0.5, pretrained="imagenet"):
    """
    Constructs a regression model using a backbone from keras_cv_attention_models.
    Tries include_top=False first, falls back to num_classes=0.
    """
    try:
        backbone = backbone_fn(include_top=False, input_shape=(input_size, input_size, 3), pretrained=pretrained)
    except TypeError as err:
        if "include_top" not in str(err):
            raise
        backbone = backbone_fn(input_shape=(input_size, input_size, 3), pretrained=pretrained, num_classes=0, classifier_activation=None)

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="linear")(x)
    return Model(backbone.input, out)


# --- Callback Helpers ---

class BestModelLogger(tf.keras.callbacks.Callback):
    """Print a single line whenever val_mae improves and the model is saved."""

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
    """Append one row per epoch to a CSV, flushing immediately."""

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


def _make_callbacks(save_path, log_path, model_name, phase, best_so_far=None):
    """Build the experiment-01/03 callback stack for one training phase."""
    best_logger = BestModelLogger(save_path, monitor="val_mae")
    checkpoint = ModelCheckpoint(str(save_path), monitor="val_mae",
                                 save_best_only=True, mode="min", verbose=0)
    if best_so_far is not None:
        best_logger.best = best_so_far
        checkpoint.best = best_so_far
    callbacks = [
        checkpoint,
        best_logger,
        ReduceLROnPlateau(monitor="val_mae", factor=0.1, patience=5, verbose=1),
        EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True, verbose=1),
        EpochCSVLogger(log_path, model_name, phase, overwrite=(phase == "frozen")),
    ]
    return callbacks, best_logger


# --- CV Training Logic ---

def train_one_fold(train_df, val_df, backbone_fn, input_size, model_name, fold_idx):
    fold_dir = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_idx:02d}')
    os.makedirs(fold_dir, exist_ok=True)

    save_path = os.path.join(fold_dir, f'{model_name}_fold{fold_idx:02d}_best_model.keras')
    done_path = os.path.join(fold_dir, f'{model_name}_fold{fold_idx:02d}.done')
    log_path = os.path.join(fold_dir, f'{model_name}_fold{fold_idx:02d}_training_log.csv')

    # Skip only when training fully completed: the .done marker is written
    # after Phase 2.
    if os.path.exists(done_path) and os.path.exists(save_path):
        print(f"Found completed {model_name} Fold {fold_idx}; skipping.")
        return
    if os.path.exists(save_path):
        print(f"Found incomplete checkpoint for {model_name} Fold {fold_idx} "
              f"(no .done marker); retraining from scratch.")

    print(f"   Building datasets for Fold {fold_idx} (Input: {input_size})...")
    train_ds = patch_data_tf_dataset_from_df(
        train_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], 
        CONFIG["BATCH_SIZE"], augment=True, final_size=input_size
    )
    val_ds = patch_data_tf_dataset_from_df(
        val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], 
        CONFIG["BATCH_SIZE"], augment=False, final_size=input_size
    )

    # --- Phase 1: Frozen Backbone (Adam @ LR_INIT) ---
    print(f"[{model_name} Fold {fold_idx}] Phase 1: Frozen Training")
    model = build_backbone_regressor(backbone_fn, input_size)
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_INIT"]), loss='mse', metrics=['mae'])

    callbacks, best_logger = _make_callbacks(save_path, log_path, model_name, "frozen")
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=CONFIG["EPOCHS_INIT"],
        verbose=2,
        callbacks=callbacks
    )
    phase1_best_mae = best_logger.best

    # Free Phase 1 graph/optimizer before Phase 2 (exp 01/03 pattern to avoid OOM)
    del model, callbacks, best_logger
    tf.keras.backend.clear_session()
    gc.collect()

    # --- Phase 2: Fine-Tuning (Adam @ LR_FT) ---
    # Rebuild from the best Phase 1 checkpoint with all layers trainable.
    print(f"[{model_name} Fold {fold_idx}] Phase 2: Fine-Tuning")
    model = build_backbone_regressor(backbone_fn, input_size)
    model.load_weights(save_path)  # start from best Phase 1 checkpoint

    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_FT"]), loss='mse', metrics=['mae'])

    # Carry Phase 1's best val_mae forward so the checkpoint is only
    # overwritten when Phase 2 actually improves on it.
    callbacks_ft, _ = _make_callbacks(save_path, log_path, model_name, "fine_tune",
                                      best_so_far=phase1_best_mae)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=CONFIG["EPOCHS_FT"],
        verbose=2,
        callbacks=callbacks_ft
    )
    print(f"Saved training log to {log_path}")

    # Mark training as fully complete so reruns skip this fold.
    with open(done_path, "w") as fh:
        fh.write(f"{model_name} Fold {fold_idx} training complete\n")
    
    del model, train_ds, val_ds, callbacks_ft
    tf.keras.backend.clear_session()
    gc.collect()


def run_cv(df_full, backbone_fn, input_size, model_name, n_splits=5):
    """Train one backbone with k-fold CV; return (per-fold metrics, OOF predictions).

    Each fold is trained exactly like experiment 01: a frozen Phase 1 then a
    fine-tuning Phase 2 rebuilt from the best Phase 1 checkpoint. The best
    checkpoint is reloaded from disk for the out-of-fold evaluation.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

    fold_metrics = []
    oof_records = []

    print(f"\n🚀 Starting CV for {model_name}...")
    for fold_id, (tr_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n── {model_name} | Fold {fold_id}/{n_splits} ──")
        tr_df = df_full.iloc[tr_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)
        train_one_fold(tr_df, val_df, backbone_fn, input_size, model_name, fold_id)

        # --- OOF evaluation (matched to Exp 3) ---
        save_path = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}',
                                  f'{model_name}_fold{fold_id:02d}_best_model.keras')
        model = build_backbone_regressor(backbone_fn, input_size)
        model.load_weights(save_path)

        val_ids_ds = patch_data_tf_dataset_with_ids(
            val_df, final_size=input_size, batch_size=CONFIG["BATCH_SIZE"]
        )
        preds_per_image = defaultdict(list)
        for patches, _, img_ids in val_ids_ds:
            p = model.predict(patches, verbose=0).ravel()
            for value, iid in zip(p, img_ids.numpy()):
                iid_str = iid.decode("utf-8") if isinstance(iid, (bytes, bytearray)) else iid
                preds_per_image[iid_str].append(float(value))

        true_map = dict(zip(val_df["File"], val_df["Age"]))
        y_true, y_pred = [], []
        for iid, plist in preds_per_image.items():
            if iid in true_map:
                mean_pred = float(np.mean(plist))
                y_true.append(float(true_map[iid]))
                y_pred.append(mean_pred)
                oof_records.append({"Model": model_name, "Fold": fold_id, "ImageID": iid,
                                    "Prediction": mean_pred, "TrueAge": float(true_map[iid])})

        metrics = compute_evaluation_metrics(np.array(y_true), np.array(y_pred))
        fold_metrics.append(metrics)
        print(f"Fold {fold_id}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")

        del model, val_ids_ds, preds_per_image
        gc.collect()
        tf.keras.backend.clear_session()

    return fold_metrics, oof_records


# --- Final Evaluation ---

def group_predictions_by_image_id(preds_with_ids, labels_df):
    grouped_predictions = defaultdict(list)
    grouped_labels = defaultdict(list)

    for pred, image_id in preds_with_ids:
        img_id = image_id.decode('utf-8') if isinstance(image_id, (bytes, bytearray)) else image_id
        grouped_predictions[img_id].append(float(pred))

    for _, row in labels_df.iterrows():
        fid = row['File']
        if fid in grouped_predictions:
            grouped_labels[fid].append(row['Age'])

    common = set(grouped_predictions.keys()) & set(grouped_labels.keys())
    y_pred = np.array([np.mean(grouped_predictions[k]) for k in common])
    y_true = np.array([np.mean(grouped_labels[k]) for k in common])
    return y_true, y_pred


def compute_evaluation_metrics(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.isnan(mape): mape = 0.0

    pct = lambda thr: 100 * np.mean(errors <= thr)
    within_2 = pct(2)
    within_5 = pct(5)
    within_10 = pct(10)
    max_err = np.max(errors)
    median_err = np.median(errors)

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | MAPE: {mape:.2f}%")
    print(f"±2 yrs: {within_2:.2f}% | ±5 yrs: {within_5:.2f}% | ±10 yrs: {within_10:.2f}%")
    print(f"Max Error: {max_err:.2f} | Median Error: {median_err:.2f}")

    metrics = {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "Acc_2yr": within_2, "Acc_5yr": within_5, "Acc_10yr": within_10,
        "Max_Error": max_err, "Median_Error": median_err,
        "Min_Error": float(np.min(errors)),
    }
    return metrics


def evaluate_models(df_full):
    print(f"\n{'='*40}\nRunning Evaluation\n{'='*40}")
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

    model_summaries = []

    for model_name, (_, input_size) in VIT_MODEL_CONFIGS.items():
        print(f"\n════ Evaluating {model_name} (Input: {input_size}x{input_size}) ════")
        fold_metrics_list = []

        for fold_id, (_, val_idx) in enumerate(splits, start=1):
            ckpt = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}', f'{model_name}_fold{fold_id:02d}_best_model.keras')
            if not os.path.exists(ckpt):
                # Fallback to the old naming scheme if available
                ckpt_ft = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}', f'{model_name}_finetune.keras')
                ckpt_init = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}', f'{model_name}_init.keras')
                ckpt = ckpt_ft if os.path.exists(ckpt_ft) else ckpt_init

            if not os.path.exists(ckpt):
                print(f"  ⚠️ Fold {fold_id} checkpoint not found. Skipping.")
                continue

            model = load_model(ckpt, compile=False)
            val_df = df_full.iloc[val_idx].reset_index(drop=True)
            val_ds = patch_data_tf_dataset_with_ids(
                val_df, final_size=input_size, batch_size=CONFIG["BATCH_SIZE"]
            )

            preds_with_ids = []
            for patches, _, img_ids in val_ds:
                p = model.predict(patches, verbose=0).ravel()
                preds_with_ids.extend(zip(p, img_ids.numpy()))

            y_true, y_pred = group_predictions_by_image_id(preds_with_ids, val_df)
            print(f"\n  [{model_name}] Fold {fold_id}:")
            metrics = compute_evaluation_metrics(y_true, y_pred)
            fold_metrics_list.append(metrics)

            # Save fold-level predictions
            fold_csv = os.path.join(CONFIG["RESULTS_DIR"], f"{model_name}_fold{fold_id}_preds.csv")
            imap = defaultdict(list)
            for pr, iid in preds_with_ids:
                iid_s = iid.decode('utf-8') if isinstance(iid, (bytes, bytearray)) else iid
                imap[iid_s].append(float(pr))
            rows = [{'Model': model_name, 'Fold': fold_id, 'ImageID': k, 'Prediction': np.mean(v)} for k, v in imap.items()]
            pd.DataFrame(rows).to_csv(fold_csv, index=False)

            del model, val_ds
            tf.keras.backend.clear_session()
            gc.collect()

        # Summary across folds
        if fold_metrics_list:
            print(f"\n  ════ {model_name} – CV Summary (n={len(fold_metrics_list)}) ════")
            for k in fold_metrics_list[0].keys():
                vals = np.array([m[k] for m in fold_metrics_list])
                print(f"  {k:<20} {np.mean(vals):.2f} ± {np.std(vals):.2f}")
            model_summaries.append({
                'Model': model_name,
                'MAE': f"{np.mean([m['MAE'] for m in fold_metrics_list]):.2f} ± {np.std([m['MAE'] for m in fold_metrics_list]):.2f}"
            })

    print("\nEvaluation Complete.")
    if model_summaries:
        print(pd.DataFrame(model_summaries))


# --- Colab / output directory handling (matched to experiments 01 and 03) ---

def _in_colab():
    """True when running on Google Colab, whose local disk is ephemeral."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def resolve_output_dirs():
    """On Colab, redirect model/result output to mounted Google Drive.

    Everything under /content is wiped on a Colab runtime crash, which would
    destroy the fold checkpoints (and their .done markers) mid-run. Persisting
    to Drive lets a rerun resume from the last completed fold. Local runs keep
    the repo-relative defaults unchanged.
    """
    if not _in_colab():
        return
    drive_root = "/content/drive"
    persist_base = os.path.join(drive_root, "MyDrive", "Age_Estimation", EXPERIMENT_DIRNAME)
    try:
        if not os.path.exists(os.path.join(drive_root, "MyDrive")):
            from google.colab import drive
            print("Colab detected: mounting Google Drive to persist trained models...")
            drive.mount(drive_root)
        if os.path.exists(os.path.join(drive_root, "MyDrive")):
            CONFIG["MODELS_DIR"] = os.path.join(persist_base, "models")
            CONFIG["RESULTS_DIR"] = os.path.join(persist_base, "results")
            print(f"Persisting outputs to Google Drive: {persist_base}")
            return
    except Exception as exc:
        print(f"WARNING: could not mount Google Drive ({exc}).")
    print("WARNING: Drive unavailable; trained models will be LOST if the "
          "Colab runtime crashes.")


# --- Results Persistence (matched to experiment 03) ---

def save_cv_results(all_fold_metrics, all_oof_records):
    """Persist OOF predictions, per-fold metrics, and the mean±std CV summary.

    ``oof_predictions.csv`` is the input consumed by reproduce_results.py's fast
    path; ``cv_metrics_summary.csv`` reproduces the paper's individual-model CV
    table (mean ± std across folds).
    """
    out_dir = CONFIG["RESULTS_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    oof_path = os.path.join(out_dir, "oof_predictions.csv")
    pd.DataFrame(all_oof_records).to_csv(oof_path, index=False)
    print(f"\nSaved OOF predictions to {oof_path} ({len(all_oof_records)} rows)")

    per_fold_rows = []
    for model_name, fms in all_fold_metrics.items():
        for i, fm in enumerate(fms, start=1):
            row = {"Model": model_name, "Fold": i}
            row.update(fm)
            per_fold_rows.append(row)
    per_fold_path = os.path.join(out_dir, "cv_metrics_per_fold.csv")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)
    print(f"Saved per-fold metrics to {per_fold_path}")

    metric_keys = ["MAE", "RMSE", "R2", "MAPE", "Acc_2yr", "Acc_5yr", "Acc_10yr",
                   "Max_Error", "Median_Error", "Min_Error"]
    summary_rows = []
    print("\n────── CV SUMMARY (mean ± std across folds) ──────")
    for model_name, fms in all_fold_metrics.items():
        if not fms:
            continue
        row = {"Model": model_name}
        print(f"\n{model_name}:")
        for k in metric_keys:
            arr = np.array([fm[k] for fm in fms], dtype=float)
            row[f"{k}_mean"], row[f"{k}_std"] = arr.mean(), arr.std()
            print(f"  {k:<13}: {arr.mean():.2f} ± {arr.std():.2f}")
        summary_rows.append(row)
    summary_path = os.path.join(out_dir, "cv_metrics_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved CV summary to {summary_path}")


# --- Main Execution ---

def main():
    resolve_output_dirs()
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    df_full = pd.read_csv(CONFIG["CSV_PATH"])

    # 1. Cross-validation training + out-of-fold evaluation (paper CV table).
    all_fold_metrics = {}
    all_oof_records = []
    for name, (fn, size) in VIT_MODEL_CONFIGS.items():
        fold_metrics, oof_records = run_cv(df_full, fn, size, name, n_splits=5)
        all_fold_metrics[name] = fold_metrics
        all_oof_records.extend(oof_records)

    save_cv_results(all_fold_metrics, all_oof_records)

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()