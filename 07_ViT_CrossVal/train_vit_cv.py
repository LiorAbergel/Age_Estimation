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
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CRITICAL: Legacy Keras for compatibility ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 64,
    "THR": 0.0054,  # Intensity threshold for empty patches
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/vit_cv",
    "RESULTS_DIR": "./results/vit_cv_predictions"
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

    # Random Contrast
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

def build_backbone_regressor(backbone_fn, input_size, dropout=0.5):
    """Constructs the ViT regressor. Handles library-specific kwargs."""
    try:
        # Standard arg for most backbones
        base = backbone_fn(input_shape=(input_size, input_size, 3), pretrained="imagenet", include_top=False)
    except TypeError:
        # Fallback for some models in library that use num_classes=0 for feature extraction
        base = backbone_fn(input_shape=(input_size, input_size, 3), pretrained="imagenet", num_classes=0)

    # Some backbones return a list or tuple, ensure we get the output tensor
    if isinstance(base.output, (list, tuple)):
        x = base.output[0]
    else:
        x = base.output

    # GAP if output is 4D (H, W, C), otherwise skip if already 2D
    if len(x.shape) == 4:
        x = GlobalAveragePooling2D()(x)
        
    x = Dropout(dropout)(x)
    out = Dense(1, activation="linear")(x)
    return Model(base.input, out)

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
    if not os.path.exists(ckpt_init):
        print("   ❄️ Stage 1: Frozen Backbone")
        # Heuristic: Freeze all except last 2 layers
        for layer in model.layers[:-2]:
            layer.trainable = False
            
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(
            train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_INIT"], verbose=2,
            callbacks=[
                ModelCheckpoint(ckpt_init, monitor='val_mae', save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=4),
                EarlyStopping(monitor='val_mae', patience=8, restore_best_weights=True)
            ]
        )
    else:
        print("   🔹 Stage 1 checkpoint found. Skipping Stage 1.")

    # --- Stage 2: Fine-Tuning ---
    print("   🔥 Stage 2: Fine-Tuning")
    # Reload best frozen weights
    model.load_weights(ckpt_init)
    
    # Unfreeze
    for layer in model.layers:
        layer.trainable = True
        
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    
    model.fit(
        train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"], verbose=2,
        callbacks=[
            ModelCheckpoint(ckpt_ft, monitor='val_mae', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=4),
            EarlyStopping(monitor='val_mae', patience=8, restore_best_weights=True)
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
        print(f"\n── {model_name} | Fold {fold_id}/{n_splits} ──")
        tr_df = df_full.iloc[tr_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)
        train_one_fold(tr_df, val_df, backbone_fn, input_size, model_name, fold_id)

# --- Final Evaluation ---

def group_predictions_by_image_id(preds_with_ids, labels_df):
    grouped_predictions = defaultdict(list)
    
    for pred, image_id in preds_with_ids:
        image_id_str = image_id.decode('utf-8')
        grouped_predictions[image_id_str].append(float(pred))
        
    # Get true values from dataframe
    true_map = dict(zip(labels_df['File'], labels_df['Age']))
    
    y_true, y_pred, ids = [], [], []
    for img_id, p_list in grouped_predictions.items():
        if img_id in true_map:
            y_pred.append(np.mean(p_list))
            y_true.append(true_map[img_id])
            ids.append(img_id)
            
    return np.array(y_true), np.array(y_pred), ids

def evaluate_models(df_full):
    print("\n══════ Running Final Inference (All Models, All Folds) ══════")
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    # We evaluate on the TEST set (held out from CV)
    test_df = df_full[df_full['Set'] == 'test'].reset_index(drop=True)
    
    model_results = []

    for model_name, (_, input_size) in VIT_MODEL_CONFIGS.items():
        print(f"\nEvaluating {model_name}...")
        
        # Aggregate predictions across 5 folds
        all_fold_preds = defaultdict(list) # {img_id: [mean_pred_fold1, mean_pred_fold2...]}
        
        test_ds = patch_data_tf_dataset_with_ids(
            test_df, final_size=input_size, batch_size=CONFIG["BATCH_SIZE"]
        )
        
        for fold in range(1, 6):
            ckpt_path = os.path.join(CONFIG["MODELS_DIR"], model_name, f'fold_{fold:02d}', f'{model_name}_finetune.keras')
            
            if not os.path.exists(ckpt_path):
                print(f"  ⚠️ Warning: Fold {fold} checkpoint not found.")
                continue
                
            model = load_model(ckpt_path, compile=False)
            
            # Predict
            preds_with_ids = []
            for patches, _, img_ids in test_ds:
                p = model.predict(patches, verbose=0).ravel()
                preds_with_ids.extend(zip(p, img_ids.numpy()))
            
            # Group by image for this fold
            _, fold_y_pred, fold_ids = group_predictions_by_image_id(preds_with_ids, test_df)
            
            for img_id, val in zip(fold_ids, fold_y_pred):
                all_fold_preds[img_id].append(val)
                
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        # Average across folds (Ensemble for this model)
        final_y_true, final_y_pred = [], []
        true_map = dict(zip(test_df['File'], test_df['Age']))
        
        for img_id, folds_preds in all_fold_preds.items():
            if img_id in true_map:
                final_y_true.append(true_map[img_id])
                final_y_pred.append(np.mean(folds_preds))
        
        # Metrics
        mae = mean_absolute_error(final_y_true, final_y_pred)
        print(f"  🏆 {model_name} Ensemble MAE: {mae:.2f}")
        model_results.append({'Model': model_name, 'MAE': mae})
        
        # Save Predictions
        res_df = pd.DataFrame({'ImageID': list(all_fold_preds.keys()), 
                               'Pred': [np.mean(v) for v in all_fold_preds.values()]})
        res_df.to_csv(os.path.join(CONFIG["RESULTS_DIR"], f"{model_name}_test_preds.csv"), index=False)

    print("\nEvaluation Complete.")
    print(pd.DataFrame(model_results))

# --- Main Execution ---

def main():
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    df_full = pd.read_csv(CONFIG["CSV_PATH"])
    
    # 1. Train CV Folds
    for name, (fn, size) in VIT_MODEL_CONFIGS.items():
        run_cv(df_full, fn, size, name, n_splits=5)

    # 2. Evaluate
    evaluate_models(df_full)

if __name__ == "__main__":
    main()