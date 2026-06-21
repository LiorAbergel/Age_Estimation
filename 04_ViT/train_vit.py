"""
Experiment 06: Vision Transformers (ViT) for Age Estimation

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

# --- Colab Setup ---
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
import subprocess
subprocess.run(['pip', 'install', '-q', 'keras-cv-attention-models', 'tf-keras', 'tqdm'], check=True)

# --- CRITICAL SETUP ---
# keras-cv-attention-models requires legacy tf-keras behavior in newer TF versions
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import architectures from external library
# Ensure you have run: pip install keras-cv-attention-models
try:
    import keras_cv_attention_models.swin_transformer_v2 as swin_v2
    import keras_cv_attention_models.mobilevit as mobilevit
    import keras_cv_attention_models.convnext as convnext
    import keras_cv_attention_models.tinyvit as tiny_vit
except ImportError:
    raise ImportError("Please run: pip install keras-cv-attention-models")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
DATA_ROOT = '/content/drive/MyDrive/HHD_AgeSplit'

CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 128,
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "LR_INIT": 1e-3,
    "LR_FT": 1e-4,
    "DATA_DIR": DATA_ROOT,
    "CSV_PATH": os.path.join(DATA_ROOT, 'NewAgeSplit.csv'),
    "MODELS_DIR": os.path.join(DATA_ROOT, 'ViT'),
    "RESULTS_DIR": os.path.join(DATA_ROOT, 'ViT', 'results')
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

def patch_data_tf_dataset_with_ids(labels_df, dataset_type, final_size, batch_size=64):
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


def _make_callbacks(save_path, log_path, model_name, phase):
    """Build the experiment-01/03 callback stack for one training phase."""
    best_logger = BestModelLogger(save_path, monitor="val_mae")
    callbacks = [
        ModelCheckpoint(str(save_path), monitor="val_mae", save_best_only=True, mode="min", verbose=0),
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

    print(f"\nSetting up Data for {model_name} (Input: {input_size}x{input_size})...")
    train_ds = patch_data_tf_dataset_from_df(
        labels_df[labels_df['Set']=='train'], data_dir,
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"],
        augment=True, final_size=input_size)

    val_ds = patch_data_tf_dataset_from_df(
        labels_df[labels_df['Set']=='val'], data_dir,
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"],
        augment=False, final_size=input_size)

    # Build Model
    model = build_backbone_regressor(backbone_fn, input_size)
    log_path = os.path.join(ckpt_root, f"{model_name}_training_log.csv")

    # --- Phase 1: Frozen Backbone (Adam @ LR_INIT) ---
    print(f"[{model_name}] Phase 1: Frozen Training")
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_INIT"]), loss='mse', metrics=['mae'])

    ckpt_init = os.path.join(ckpt_root, f"{model_name}_init.keras")
    if os.path.exists(ckpt_init):
        print(f"Loading existing init model: {ckpt_init}")
        model.load_weights(ckpt_init)
    else:
        callbacks, _ = _make_callbacks(ckpt_init, log_path, model_name, "frozen")
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_INIT"],
                  callbacks=callbacks, verbose=2)

    # --- Phase 2: Fine-Tuning (Adam @ LR_FT) ---
    print(f"[{model_name}] Phase 2: Fine-Tuning")
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["LR_FT"]), loss='mse', metrics=['mae'])

    ckpt_ft = os.path.join(ckpt_root, f"{model_name}_finetune.keras")
    if os.path.exists(ckpt_ft):
        print(f"Loading existing finetuned model: {ckpt_ft}")
        model.load_weights(ckpt_ft)
    else:
        callbacks_ft, _ = _make_callbacks(ckpt_ft, log_path, model_name, "fine_tune")
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"],
                  callbacks=callbacks_ft, verbose=2)

    return model

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
    mape = np.mean(errors / y_true) * 100
    
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
        "Within_2yr": within_2, "Within_5yr": within_5, "Within_10yr": within_10,
        "Max_Error": max_err, "Median_Error": median_err
    }
    return metrics

# --- Main ---

def main():
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

    trained_models = {}

    # 2. Training Loop
    for name, (fn, img_size) in MODEL_REGISTRY.items():
        print(f"\n{'='*40}\nProcessing {name}\n{'='*40}")
        model = train_one_model(fn, img_size, labels_data, CONFIG["DATA_DIR"], name)
        trained_models[name] = (model, img_size)
        
        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()

    # 3. Evaluation & Ensemble
    print(f"\n{'='*40}\nRunning Evaluation\n{'='*40}")
    
    # Reload models to ensure clean state (optional, but safe)
    image_level_preds = {}
    pred_by_image = {}   # model_name -> {ImageID: prediction}
    common_ids = None
    y_true_ref = None
    
    for name, (fn, img_size) in MODEL_REGISTRY.items():
        print(f"Evaluating {name}...")
        ckpt_path = os.path.join(CONFIG["MODELS_DIR"], name, f"{name}_finetune.keras")
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
    main()