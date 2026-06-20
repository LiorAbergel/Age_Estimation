"""
Experiment 07: Vision Transformers with 5-Fold Stratified Group CV

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
import shutil

# --- Colab Setup ---
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
import subprocess
subprocess.run(['pip', 'install', '-q', 'keras-cv-attention-models', 'tf-keras', 'tqdm'], check=True)

# --- CRITICAL: Must be set BEFORE importing TensorFlow ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import ViT architectures
try:
    import keras_cv_attention_models.swin_transformer_v2 as swin_v2
    import keras_cv_attention_models.mobilevit as mobilevit
    import keras_cv_attention_models.convnext as convnext
    import keras_cv_attention_models.tinyvit as tiny_vit
except ImportError:
    raise ImportError("Please install required lib: pip install keras-cv-attention-models")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
DATA_ROOT = '/content/drive/MyDrive/HHD_AgeSplit'

CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 64,
    "THR": 0.0054,  # Intensity threshold for empty patches
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 30,
    "LR_INIT": 1e-3,
    "LR_FT": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_EPOCHS": 5,
    "CLIP_NORM": 1.0,
    "DATA_DIR": DATA_ROOT,
    "CSV_PATH": os.path.join(DATA_ROOT, 'NewAgeSplit.csv'),
    "MODELS_DIR": os.path.join(DATA_ROOT, 'ViT_CV2'),
    "RESULTS_DIR": os.path.join(DATA_ROOT, 'ViT_CV2', 'results')
}

# --- Mixed Precision ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Resume Configuration ---
# Set RESUME_MODEL to the model name to resume from (None = start fresh)
# Set RESUME_FOLD to the fold number (1-indexed) to resume from
# Set RESUME_PHASE1_EPOCH to the epoch to resume Phase 1 from (if crashed mid-phase)
RESUME_MODEL = "ConvNeXtV2_Tiny"
RESUME_FOLD = 1
RESUME_PHASE1_EPOCH = 35  # Will load best ckpt and continue from this epoch

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
        return np.zeros((800, 800, 3), dtype=np.float32)

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

    # Random Contrast
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)

    # Gaussian Noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label

def resize_for_model(patch, label, final_size):
    """Resizes patch to ViT input requirement (e.g. 224x224)."""
    patch = tf.image.resize(patch, [final_size, final_size])
    return patch, label

def resize_for_model_with_id(patch, label, img_id, final_size):
    patch = tf.image.resize(patch, [final_size, final_size])
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

def patch_data_tf_dataset_with_ids(labels_df_subset, final_size, batch_size=64):
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
    # Use float32 for the output layer to maintain regression precision under mixed_float16
    out = Dense(1, activation="linear", dtype="float32")(x)
    return Model(backbone.input, out)

# --- CV Training Logic ---

def train_one_fold(train_df, val_df, backbone_fn, input_size, model_name, fold_idx):
    fold_dir = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_idx:02d}')
    os.makedirs(fold_dir, exist_ok=True)

    ckpt_init = os.path.join(fold_dir, f'{model_name}_init.keras')
    ckpt_ft = os.path.join(fold_dir, f'{model_name}_finetune.keras')

    # Resume Check
    if os.path.exists(ckpt_ft):
        print(f"✅ Fold {fold_idx} already finished. Skipping...")
        return

    print(f"   Building datasets for Fold {fold_idx} (Input: {input_size})...")
    train_ds = patch_data_tf_dataset_from_df(
        train_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], 
        CONFIG["BATCH_SIZE"], augment=True, final_size=input_size
    )
    val_ds = patch_data_tf_dataset_from_df(
        val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], 
        CONFIG["BATCH_SIZE"], augment=False, final_size=input_size
    )

    model = build_backbone_regressor(backbone_fn, input_size)

    # --- Stage 1: Frozen ---
    # Determine if we need to resume Phase 1 mid-training
    resume_p1_epoch = 0
    if (RESUME_MODEL == model_name and RESUME_FOLD == fold_idx
            and RESUME_PHASE1_EPOCH > 0 and os.path.exists(ckpt_init)):
        resume_p1_epoch = RESUME_PHASE1_EPOCH

    if not os.path.exists(ckpt_init) or resume_p1_epoch > 0:
        print("   ❄️ Stage 1: Frozen Backbone")
        if resume_p1_epoch > 0:
            print(f"   ↪️ Resuming Phase 1 from epoch {resume_p1_epoch} (loading best checkpoint)...")
            model = load_model(ckpt_init)
        for layer in model.layers[:-2]:
            layer.trainable = False

        # Cosine decay with warmup for Phase 1
        steps_per_epoch_est = 200  # approximate; adapts via callbacks
        total_steps_p1 = CONFIG["EPOCHS_INIT"] * steps_per_epoch_est
        warmup_steps_p1 = CONFIG["WARMUP_EPOCHS"] * steps_per_epoch_est

        lr_schedule_p1 = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-6,
            decay_steps=total_steps_p1,
            alpha=1e-6,
            warmup_target=CONFIG["LR_INIT"],
            warmup_steps=warmup_steps_p1
        )
        optimizer_p1 = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule_p1,
            weight_decay=CONFIG["WEIGHT_DECAY"],
            clipnorm=CONFIG["CLIP_NORM"]
        )
        model.compile(optimizer=optimizer_p1, loss='mse', metrics=['mae'])
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=CONFIG["EPOCHS_INIT"],
            initial_epoch=resume_p1_epoch,
            verbose=2,
            callbacks=[
                ModelCheckpoint(ckpt_init, monitor='val_mae', save_best_only=True, verbose=1),
                EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)
            ]
        )
    else:
        print("   🔹 Stage 1 checkpoint found. Skipping Stage 1.")

    # --- Stage 2: Fine-Tuning ---
    print("   🔥 Stage 2: Fine-Tuning")
    # Reload best frozen weights
    model.load_weights(ckpt_init)

    for layer in model.layers:
        layer.trainable = True

    # Cosine decay with warmup for Phase 2
    steps_per_epoch_est = 200
    total_steps_p2 = CONFIG["EPOCHS_FT"] * steps_per_epoch_est
    warmup_steps_p2 = 3 * steps_per_epoch_est  # shorter warmup for fine-tune

    lr_schedule_p2 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-7,
        decay_steps=total_steps_p2,
        alpha=1e-7,
        warmup_target=CONFIG["LR_FT"],
        warmup_steps=warmup_steps_p2
    )
    optimizer_p2 = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule_p2,
        weight_decay=CONFIG["WEIGHT_DECAY"],
        clipnorm=CONFIG["CLIP_NORM"]
    )
    model.compile(optimizer=optimizer_p2, loss='mse', metrics=['mae'])

    model.fit(
        train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"], verbose=2,
        callbacks=[
            ModelCheckpoint(ckpt_ft, monitor='val_mae', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_mae', patience=7, restore_best_weights=True)
        ]
    )
    
    del model, train_ds, val_ds
    tf.keras.backend.clear_session()
    gc.collect()

def run_cv(df_full, backbone_fn, input_size, model_name, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

    print(f"\n🚀 Starting CV for {model_name}...")
    for fold_id, (tr_idx, val_idx) in enumerate(splits, start=1):
        # Skip folds before the resume point for the resume model
        if RESUME_MODEL == model_name and RESUME_FOLD is not None and fold_id < RESUME_FOLD:
            print(f"\n── {model_name} | Fold {fold_id}/{n_splits} ── ⏭️ Skipped (before resume point)")
            continue
        print(f"\n── {model_name} | Fold {fold_id}/{n_splits} ──")
        tr_df = df_full.iloc[tr_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)
        train_one_fold(tr_df, val_df, backbone_fn, input_size, model_name, fold_id)

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
            ckpt_path = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}', f'{model_name}_finetune.keras')
            ckpt_init = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold_id:02d}', f'{model_name}_init.keras')
            ckpt = ckpt_path if os.path.exists(ckpt_path) else ckpt_init

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

# --- Main Execution ---

def main():
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    df_full = pd.read_csv(CONFIG["CSV_PATH"])
    
    # 1. Train CV Folds
    model_names = list(VIT_MODEL_CONFIGS.keys())
    start_idx = model_names.index(RESUME_MODEL) if RESUME_MODEL else 0
    for name in model_names[start_idx:]:
        fn, size = VIT_MODEL_CONFIGS[name]
        run_cv(df_full, fn, size, name, n_splits=5)

    # 2. Evaluate
    evaluate_models(df_full)

if __name__ == "__main__":
    main()