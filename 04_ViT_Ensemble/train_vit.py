"""
Experiment 04: ViT Ensemble for Age Estimation

Overview:
This experiment evaluates modern Vision Transformer architectures and hybrid
ConvNet-Transformer models for handwriting age estimation. It uses the
'keras_cv_attention_models' library to access state-of-the-art backbones.

Models Evaluated:
- Swin Transformer V2 (Tiny)
- MobileViT (XXS)
- ConvNeXt V2 (Tiny)
- TinyViT (11M)

Requirements:
- pip install keras-cv-attention-models tf-keras
"""

import os
import gc
import csv
import sys
import warnings

# keras-cv-attention-models requires legacy tf-keras behavior in newer TF versions.
# Must be set before importing TensorFlow.
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import architectures from external library
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
# completion.  Silence just this message to keep the training log readable.
warnings.filterwarnings("ignore", message="Your input ran out of data")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EXPERIMENT_DIRNAME = "04_ViT_Ensemble"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 128,
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "LR_INIT": 1e-3,
    "LR_FT": 1e-4,
    "THR": 0.0054,
    "DATA_DIR": os.path.join(REPO_ROOT, "data"),
    "CSV_PATH": os.path.join(REPO_ROOT, "data", "NewAgeSplit.csv"),
    "MODELS_DIR": os.path.join(REPO_ROOT, "models", "experiment_04"),
    "RESULTS_DIR": os.path.join(REPO_ROOT, "results", "experiment_04")
}

# --- Data Processing ---

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

    # Filter empty patches (matching exp 01 and 03)
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

# ViTs often require specific input sizes (e.g., 224x224 or 256x256)
# We will resize the 400x400 patches to the model's native resolution during loading.

rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
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

def resize_for_model(patch, label, final_size):
    """Resizes the 400x400 patch to the model's expected input size (e.g. 224x224)."""
    patch = tf.image.resize(patch, [final_size, final_size], method='bicubic')
    return patch, label

def resize_for_model_with_id(patch, label, img_id, final_size):
    # Use the same (bicubic) resize for inference as for training.
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
    
    # Resize specifically for ViT input requirements
    if final_size is not None:
        ds = ds.map(lambda p, l: resize_for_model(p, l, final_size), num_parallel_calls=tf.data.AUTOTUNE)
        
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def patch_data_tf_dataset_with_ids(labels_df, dataset_type, final_size, batch_size=128):
    df = labels_df[labels_df['Set'] == dataset_type].reset_index(drop=True)
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.map(process_row_with_id, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(lambda p, l, i: tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(p),
        tf.data.Dataset.from_tensor_slices(l),
        tf.data.Dataset.from_tensor_slices(i)
    )))
    # Resize patches for inference
    ds = ds.map(lambda p, l, i: resize_for_model_with_id(p, l, i, final_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Model Construction ---

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
    output = Dense(1, activation="linear")(x)

    return Model(backbone.input, output)

# --- Training callbacks (matched to experiments 01 and 03) ---

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
    """Build the experiment-01/03 callback stack for one training phase.

    ``best_so_far`` carries the previous phase's best val_mae into the checkpoint
    and logger so a phase overwrites the checkpoint only when it actually improves
    on it (Phase 2 vs. Phase 1). Returns the callback list and the BestModelLogger.
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
        EpochCSVLogger(log_path, model_name, phase, overwrite=(phase == "frozen")),
    ]
    return callbacks, best_logger


# --- Training Loop ---

def train_one_model(backbone_fn, input_size, labels_df, data_dir, model_name):
    ckpt_root = os.path.join(CONFIG["MODELS_DIR"], model_name)
    os.makedirs(ckpt_root, exist_ok=True)
    save_path = os.path.join(ckpt_root, f"{model_name}_best_model.keras")
    done_path = os.path.join(ckpt_root, f"{model_name}.done")
    log_path = os.path.join(ckpt_root, f"{model_name}_training_log.csv")

    # Skip only when training fully completed: the .done marker is written
    # after Phase 2.  The .keras checkpoint alone is not enough, since
    # ModelCheckpoint saves it as early as epoch 1 (so a crashed run leaves a
    # partial checkpoint that must NOT be treated as finished).
    if os.path.exists(done_path) and os.path.exists(save_path):
        print(f"Found completed {model_name}; will load from disk for inference.")
        return
    if os.path.exists(save_path):
        print(f"Found incomplete checkpoint for {model_name} (no '{model_name}.done' "
              f"marker); retraining from scratch.")

    print(f"\nSetting up Data for {model_name} (Input: {input_size}x{input_size})...")
    train_ds = patch_data_tf_dataset_from_df(
        labels_df[labels_df['Set']=='train'], data_dir,
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"],
        augment=True, final_size=input_size)

    val_ds = patch_data_tf_dataset_from_df(
        labels_df[labels_df['Set']=='val'], data_dir,
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"],
        augment=False, final_size=input_size)

    # --- Phase 1: Frozen Backbone (Adam @ LR_INIT) ---
    print(f"[{model_name}] Phase 1: Frozen Training")
    model = build_backbone_regressor(backbone_fn, input_size)
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_INIT"]), loss='mse', metrics=['mae'])

    callbacks, best_logger = _make_callbacks(save_path, log_path, model_name, "frozen")
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_INIT"],
              callbacks=callbacks, verbose=2)
    phase1_best_mae = best_logger.best

    # Free Phase 1 graph/optimizer before Phase 2 (exp 01/03 pattern to avoid OOM)
    del model, callbacks, best_logger
    tf.keras.backend.clear_session()
    gc.collect()

    # --- Phase 2: Fine-Tuning (Adam @ LR_FT) ---
    # Rebuild from the best Phase 1 checkpoint with all layers trainable.
    print(f"[{model_name}] Phase 2: Fine-Tuning")
    model = build_backbone_regressor(backbone_fn, input_size)
    model.load_weights(save_path)  # start from best Phase 1 checkpoint
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_FT"]), loss='mse', metrics=['mae'])

    # Carry Phase 1's best val_mae forward so the checkpoint is only
    # overwritten when Phase 2 actually improves on it.
    callbacks_ft, _ = _make_callbacks(save_path, log_path, model_name, "fine_tune",
                                     best_so_far=phase1_best_mae)
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"],
              callbacks=callbacks_ft, verbose=2)
    print(f"Saved training log to {log_path}")

    # Mark training as fully complete so reruns skip this backbone.
    with open(done_path, "w") as fh:
        fh.write(f"{model_name} training complete\n")

    # Release this backbone before the next one: the best weights are safely
    # on disk, so free GPU/host memory and reset the Keras session.
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# --- Evaluation Helper ---

def group_predictions_by_image_id(preds_with_ids, labels_df):
    pred_dict = defaultdict(list)
    for pred, img_id in preds_with_ids:
        key = img_id.decode() if isinstance(img_id, bytes) else img_id
        pred_dict[key].append(pred)
    
    true_dict = labels_df.groupby('File')['Age'].mean().to_dict()
    
    common = sorted(set(pred_dict) & set(true_dict))
    y_pred = np.array([np.mean(pred_dict[k]) for k in common])
    y_true = np.array([true_dict[k] for k in common])
    
    return y_true, y_pred, common

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
    destroy the checkpoints (and their .done markers) mid-run.  Persisting
    to Drive lets a rerun resume from the last completed backbone.  Local runs
    keep the repo-relative defaults unchanged.
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


# --- Main ---

def main():
    resolve_output_dirs()
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])

    # 1. Define ViT Models Registry
    # (Name) -> (Backbone Function, Input Resolution)
    MODEL_REGISTRY = {
        "SwinV2_Tiny":     (swin_v2.SwinTransformerV2Tiny_window8, 256),
        "MobileViT_XXS":   (mobilevit.MobileViT_XXS, 256),
        "ConvNeXtV2_Tiny": (convnext.ConvNeXtTiny, 224),
        "TinyViT_11M":     (tiny_vit.TinyViT_11M, 224),
    }

    # 2. Training Loop
    for name, (fn, img_size) in MODEL_REGISTRY.items():
        print(f"\n{'='*40}\nProcessing {name}\n{'='*40}")
        train_one_model(fn, img_size, labels_data, CONFIG["DATA_DIR"], name)

    # 3. Evaluation & Ensemble
    print(f"\n{'='*40}\nRunning Evaluation\n{'='*40}")
    
    # Reload models to ensure clean state (optional, but safe)
    image_level_preds = {}
    pred_by_image = {}   # model_name -> {ImageID: prediction}
    common_ids = None
    y_true_ref = None
    
    for name, (fn, img_size) in MODEL_REGISTRY.items():
        print(f"Evaluating {name}...")
        ckpt_path = os.path.join(CONFIG["MODELS_DIR"], name, f"{name}_best_model.keras")
        model = tf.keras.models.load_model(ckpt_path)
        
        # Generate Test Data (Dynamic Resizing per model)
        test_ds = patch_data_tf_dataset_with_ids(labels_data, 'test', img_size, CONFIG["BATCH_SIZE"])
        
        preds_list = []
        for imgs, _, ids in tqdm(test_ds):
            ps = model.predict(imgs, verbose=0).flatten()
            preds_list.extend(zip(ps, ids.numpy()))
            
        y_true, y_pred, common = group_predictions_by_image_id(preds_list, labels_data)
        compute_evaluation_metrics(y_true, y_pred)
        
        image_level_preds[name] = y_pred
        pred_by_image[name] = dict(zip(common, y_pred))
        if y_true_ref is None:
            y_true_ref = y_true
            common_ids = common
            
        tf.keras.backend.clear_session()

    # 4. Save per-image per-model predictions (consumed by reproduce_results.py)
    if pred_by_image and common_ids is not None:
        preds_df = pd.DataFrame({"ImageID": common_ids})
        for name in MODEL_REGISTRY:
            preds_df[name] = [pred_by_image[name].get(iid, np.nan) for iid in common_ids]
        out_csv = os.path.join(CONFIG["RESULTS_DIR"], "test_image_predictions.csv")
        preds_df.to_csv(out_csv, index=False)
        print(f"Per-image test predictions saved to {out_csv}")

    # 5. Simple Average Ensemble
    print("\n=== ViT Ensemble Metrics ===")
    if image_level_preds:
        # Average all predictions
        ensemble_preds = np.mean(list(image_level_preds.values()), axis=0)
        compute_evaluation_metrics(y_true_ref, ensemble_preds)

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()