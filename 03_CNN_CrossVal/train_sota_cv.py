"""
Experiment 03: SOTA CNN with 5-Fold Stratified Group Cross-Validation (Full Run)

Overview:
This script performs a rigorous evaluation using 5-Fold Cross-Validation.
It ensures that patches from the same writer do not leak between training
and validation sets (GroupKFold), while maintaining class balance (Stratified).

Key Features:
- StratifiedGroupKFold: Splitting based on 'WriterNumber' and 'AgeGroup'.
- Patch Filtering: Removes empty/background patches using an intensity threshold.
- Two-Stage Training: Frozen backbone -> Fine-tuning.
- Ensemble Inference: Averages predictions from all 5 folds on the Test set.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import gc
import csv
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import (
    ResNet50, InceptionV3, InceptionResNetV2, DenseNet121
)
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Benign Keras 3 false-positive: the unknown-cardinality tf.data pipeline
# (flat_map yields a variable number of patches per image) makes Keras warn at
# end-of-dataset even though every epoch runs to completion. Silence just this.
warnings.filterwarnings("ignore", message="Your input ran out of data")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 128,
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "THR": 0.0054,  # Threshold for filtering empty patches
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/experiment_03",
    "RESULTS_DIR": "./results/experiment_03"
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
        # Return a dummy image (black) if loading fails
        return np.zeros((CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3), dtype=np.float32)

def process_image(row, root_dir, patch_size, step_size):
    """
    Loads image, extracts patches, and filters based on intensity threshold.
    """
    # Create full path: root_dir/Subset/Filename
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
    """
    Similar to process_image but also returns the File ID (for inference/tracking).
    """
    patches, labels = process_image(row, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
    image_id = tf.fill([tf.shape(patches)[0]], row['File'])
    return patches, labels, image_id

# --- Augmentation ---

rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
    image = rotation_layer(image, training=True)
    
    # Random Zoom
    orig_shape = tf.shape(image)[:2]
    zoom_factor = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    # Brightness, Contrast & Noise
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0., 1.)

    return image, label

# --- Dataset Generators ---

def patch_data_tf_dataset_from_df(labels_df_subset, data_dir, patch_size, step_size, batch_size, augment=False):
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
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def patch_data_tf_dataset_with_ids_from_df(df_subset, data_dir, patch_size, step_size, batch_size, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
    ds = ds.map(process_row_with_id, num_parallel_calls=tf.data.AUTOTUNE).flat_map(
        lambda patches, labels, image_ids: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels),
            tf.data.Dataset.from_tensor_slices(image_ids)
        ))
    )
    if augment:
        ds = ds.map(lambda p, l, i: (advanced_augmentation(p, l)[0], l, i), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Model Building ---

def build_sota_model(base_model_fn, input_shape=(400, 400, 3), dropout_rate=0.5):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    return Model(inputs, outputs)

# --- Evaluation Helpers ---

def compute_evaluation_metrics(true_images, predicted_images):
    mae = mean_absolute_error(true_images, predicted_images)
    rmse = np.sqrt(mean_squared_error(true_images, predicted_images))
    r2 = r2_score(true_images, predicted_images)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_images - predicted_images) / true_images)) * 100
        if np.isnan(mape): mape = 0.0

    # Threshold-based accuracy and error statistics
    errors = np.abs(true_images - predicted_images)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Acc_2yr": np.mean(errors <= 2) * 100,
        "Acc_5yr": np.mean(errors <= 5) * 100,
        "Acc_10yr": np.mean(errors <= 10) * 100,
        "Max_Error": float(np.max(errors)),
        "Median_Error": float(np.median(errors)),
        "Min_Error": float(np.min(errors)),
    }
    return metrics

# --- Training callbacks (logging, matched to experiment 01) ---

class BestModelLogger(tf.keras.callbacks.Callback):
    """Print a single line whenever val_mae improves and the model is saved.

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

    Saved alongside the fold checkpoint so the curves travel with the weights
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


# --- CV Logic ---

def _make_callbacks(save_path, log_path, model_tag, phase, best_so_far=None):
    """Build the experiment-01 callback stack for one training phase.

    ``best_so_far`` carries the previous phase's best val_mae into the checkpoint
    and logger so a phase overwrites the single fold checkpoint only when it
    actually improves on it (Phase 2 vs. Phase 1). Returns the callback list and
    the BestModelLogger (whose ``.best`` holds the phase's best val_mae).
    """
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
        EpochCSVLogger(log_path, model_tag, phase, overwrite=(phase == "frozen")),
    ]
    return callbacks, best_logger


def run_cv(df_full, base_model_fn, model_name, n_splits=5):
    """Train one backbone with k-fold CV; return (per-fold metrics, OOF predictions).

    Each fold is trained exactly like experiment 01: a frozen Phase 1 then a
    fine-tuning Phase 2 rebuilt from the best Phase 1 checkpoint. The two phases
    share a single ``{model}_fold{fold}_best_model.keras`` checkpoint with
    carry-forward (Phase 2 overwrites it only when it beats Phase 1); the Phase 1
    graph is freed before Phase 2 (unfreezing the backbone roughly triples
    memory); and a ``.done`` marker — written only after Phase 2 — makes a
    crashed fold retrain from scratch instead of being mistaken for finished.
    The best checkpoint is reloaded from disk for the out-of-fold evaluation.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

    ckpt_dir = os.path.join(CONFIG["MODELS_DIR"], model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    fold_metrics = []
    oof_records = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n── {model_name} Fold {fold}/{n_splits} ──")

        train_df = df_full.iloc[train_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)

        save_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold}_best_model.keras")
        done_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold}.done")
        log_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold}_training_log.csv")
        model_tag = f"{model_name}_fold{fold}"

        # Skip training only when the fold fully completed: the .done marker is
        # written after Phase 2. The .keras alone is not enough, since
        # save_best_only writes it as early as epoch 1.
        if os.path.exists(done_path) and os.path.exists(save_path):
            print(f"Found completed fold {fold}; loading from disk for evaluation.")
        else:
            if os.path.exists(save_path):
                print(f"Found incomplete checkpoint for {model_tag} (no .done marker); "
                      f"retraining from scratch.")

            train_ds = patch_data_tf_dataset_from_df(train_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=True)
            val_ds = patch_data_tf_dataset_from_df(val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False)

            # --- Phase 1: frozen backbone ---
            print("Phase 1: frozen backbone")
            model = build_sota_model(base_model_fn, input_shape=(*CONFIG["PATCH_SIZE"], 3))
            model.layers[1].trainable = False
            callbacks, best_logger = _make_callbacks(save_path, log_path, model_tag, "frozen")
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
            model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_INIT"],
                      callbacks=callbacks, verbose=2)
            phase1_best_mae = best_logger.best

            # Free the Phase 1 graph/optimizer before building Phase 2.
            del model, callbacks, best_logger
            gc.collect()
            tf.keras.backend.clear_session()

            # --- Phase 2: fine-tuning, rebuilt from the best Phase 1 checkpoint ---
            print("Phase 2: fine-tuning")
            model = load_model(save_path)
            for layer in model.layers:
                layer.trainable = True
            callbacks_p2, _ = _make_callbacks(save_path, log_path, model_tag, "fine_tune",
                                              best_so_far=phase1_best_mae)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
            model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"],
                      callbacks=callbacks_p2, verbose=2)
            print(f"Saved training log to {log_path}")

            with open(done_path, "w") as fh:
                fh.write(f"{model_tag} training complete\n")
            del model, train_ds, val_ds, callbacks_p2
            gc.collect()
            tf.keras.backend.clear_session()

        # --- Evaluate fold at image level (OOF), from the best checkpoint on disk ---
        model = load_model(save_path)
        val_ids_ds = patch_data_tf_dataset_with_ids_from_df(val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False)
        preds_per_image = defaultdict(list)
        for patches, _, img_ids in val_ids_ds:
            p = model.predict(patches, verbose=0).ravel()
            for value, iid in zip(p, img_ids.numpy()):
                preds_per_image[iid.decode("utf-8")].append(float(value))

        true_map = dict(zip(val_df["File"], val_df["Age"]))
        y_true, y_pred = [], []
        for iid, plist in preds_per_image.items():
            if iid in true_map:
                mean_pred = float(np.mean(plist))
                y_true.append(float(true_map[iid]))
                y_pred.append(mean_pred)
                oof_records.append({"Model": model_name, "Fold": fold, "ImageID": iid,
                                    "Prediction": mean_pred, "TrueAge": float(true_map[iid])})

        metrics = compute_evaluation_metrics(np.array(y_true), np.array(y_pred))
        fold_metrics.append(metrics)
        print(f"Fold {fold}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")

        # Free this fold before the next one.
        del model, val_ids_ds, preds_per_image
        gc.collect()
        tf.keras.backend.clear_session()

    return fold_metrics, oof_records

# --- Final Inference & Ensembling ---

def predict_on_test_set(models_dict, test_df):
    print("\n══════ Running Final Inference on Test Set ══════")
    test_ds = patch_data_tf_dataset_with_ids_from_df(
        test_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False
    )
    
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    all_image_preds = defaultdict(list) # {ImageID: [pred1, pred2, ...]}

    for model_name, folds_to_run in models_dict.items():
        for fold in folds_to_run:
            ckpt_path = os.path.join(CONFIG["MODELS_DIR"], model_name, f"{model_name}_fold{fold}_best_model.keras")
            
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found {ckpt_path}, skipping.")
                continue
                
            print(f"Predicting with {model_name} Fold {fold}...")
            model = load_model(ckpt_path)
            
            # Predict
            fold_preds = defaultdict(list)
            for patches, _, img_ids in test_ds:
                p = model.predict(patches, verbose=0).ravel()
                for img_id_bytes, val in zip(img_ids.numpy(), p):
                    img_id = img_id_bytes.decode('utf-8')
                    fold_preds[img_id].append(float(val))
            
            # Aggregate patch -> image for this fold
            csv_data = []
            for img_id, p_list in fold_preds.items():
                mean_p = np.mean(p_list)
                all_image_preds[img_id].append(mean_p)
                csv_data.append([model_name, fold, img_id, mean_p])
                
            # Save fold CSV
            csv_path = os.path.join(CONFIG["RESULTS_DIR"], f"{model_name}_fold{fold}_preds.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model', 'Fold', 'ImageID', 'Prediction'])
                writer.writerows(csv_data)
            
            del model
            gc.collect()
            tf.keras.backend.clear_session()

    # --- Create Ensemble ---
    print("\nCalculating Ensemble Metrics...")
    ensemble_data = []
    y_true = []
    y_pred = []
    
    # Map True Labels
    true_map = dict(zip(test_df['File'], test_df['Age']))
    
    for img_id, pred_list in all_image_preds.items():
        if img_id in true_map:
            final_pred = np.mean(pred_list)
            y_pred.append(final_pred)
            y_true.append(true_map[img_id])
            ensemble_data.append([img_id, final_pred, true_map[img_id]])
            
    # Save Ensemble CSV
    ens_csv = os.path.join(CONFIG["RESULTS_DIR"], "ensemble_final.csv")
    pd.DataFrame(ensemble_data, columns=['ImageID', 'Pred_Age', 'True_Age']).to_csv(ens_csv, index=False)
    
    # Final Metrics
    if len(y_true) > 0:
        final_metrics = compute_evaluation_metrics(np.array(y_true), np.array(y_pred))
        print(f"🏆 Final Ensemble MAE: {final_metrics['MAE']:.2f} years")
        print(f"🏆 Final Ensemble RMSE: {final_metrics['RMSE']:.2f} years")
    else:
        print("No predictions generated for metrics calculation.")

# --- Main Execution ---

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
    to Drive lets a rerun resume from the last completed phase. Local runs keep
    the repo-relative defaults unchanged.
    """
    if not _in_colab():
        return
    drive_root = "/content/drive"
    persist_base = os.path.join(drive_root, "MyDrive", "Age_Estimation", "03_CNN_CrossVal")
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
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: could not mount Google Drive ({exc}).")
    print("WARNING: Drive unavailable; trained models will be LOST if the "
          "Colab runtime crashes.")


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


def main():
    resolve_output_dirs()  # redirect outputs to Google Drive on Colab (crash-safe)
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    df_full = pd.read_csv(CONFIG["CSV_PATH"])

    # Backbones, ordered as in the original training script.
    models_to_run = {
        'ResNet50': ResNet50,
        'DenseNet121': DenseNet121,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M,
    }

    # 1. Cross-validation training + out-of-fold evaluation (paper CV table).
    all_fold_metrics = {}
    all_oof_records = []
    for name, architecture in models_to_run.items():
        fold_metrics, oof_records = run_cv(df_full, architecture, name, n_splits=5)
        all_fold_metrics[name] = fold_metrics
        all_oof_records.extend(oof_records)

    save_cv_results(all_fold_metrics, all_oof_records)

    # 2. Test-set inference + cross-fold ensemble (kept for completeness; not part
    #    of the paper's individual-model CV table).
    inference_map = {name: [1, 2, 3, 4, 5] for name in models_to_run}
    test_df = df_full[df_full['Set'] == 'test'].reset_index(drop=True)
    predict_on_test_set(inference_map, test_df)

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()