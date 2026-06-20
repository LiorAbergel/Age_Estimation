"""
Experiment 05: SOTA CNN with 5-Fold Stratified Group Cross-Validation (Full Run)

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
import gc
import csv
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 64,
    "EPOCHS_INIT": 50,
    "EPOCHS_FT": 10,
    "THR": 0.0054,  # Threshold for filtering empty patches
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/cv_strat_group",
    "RESULTS_DIR": "./results/cv_predictions"
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

    # Contrast & Noise
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

def group_predictions_by_image_id(predictions_with_ids, labels_df):
    grouped_predictions = defaultdict(list)
    grouped_labels = defaultdict(list)
    
    for pred, image_id in predictions_with_ids:
        image_id_str = image_id.decode('utf-8')
        grouped_predictions[image_id_str].append(pred)
        
    for _, row in labels_df.iterrows():
        file_id = row['File']
        if file_id in grouped_predictions:
            grouped_labels[file_id].append(row['Age'])
            
    common_ids = set(grouped_predictions.keys()) & set(grouped_labels.keys())
    predicted_images = [np.mean(grouped_predictions[img_id]) for img_id in common_ids]
    true_images = [np.mean(grouped_labels[img_id]) for img_id in common_ids]
    
    return np.array(predicted_images), np.array(true_images)

def compute_evaluation_metrics(true_images, predicted_images):
    mae = mean_absolute_error(true_images, predicted_images)
    rmse = np.sqrt(mean_squared_error(true_images, predicted_images))
    r2 = r2_score(true_images, predicted_images)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_images - predicted_images) / true_images)) * 100
        if np.isnan(mape): mape = 0.0
    
    # Calculate accuracy within thresholds
    errors = np.abs(true_images - predicted_images)
    within_2 = np.mean(errors <= 2) * 100
    within_5 = np.mean(errors <= 5) * 100
    
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "Acc_2yr": within_2, "Acc_5yr": within_5}
    return metrics

# --- CV Logic ---

def run_cv(df_full, base_model_fn, model_name, n_splits=5):
    # 1. Prepare Folds
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))
    
    ckpt_dir = os.path.join(CONFIG["MODELS_DIR"], model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n── {model_name} Fold {fold}/{n_splits} ──")

        # 2. Slice Dataframes
        train_df = df_full.iloc[train_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)

        # 3. Datasets
        train_ds = patch_data_tf_dataset_from_df(train_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=True)
        val_ds = patch_data_tf_dataset_from_df(val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False)

        # 4. Phase 1: Frozen Training
        model = build_sota_model(base_model_fn, input_shape=(*CONFIG["PATCH_SIZE"], 3))
        model.layers[1].trainable = False
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        ckpt_init = os.path.join(ckpt_dir, f"{model_name}_fold{fold}_init.keras")
        # Check if already trained
        if os.path.exists(ckpt_init):
            print(f"Loading existing init model: {ckpt_init}")
            model.load_weights(ckpt_init)
        else:
            callbacks = [
                ModelCheckpoint(ckpt_init, monitor="val_mae", save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor="val_mae", factor=0.2, patience=4, verbose=1),
                EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True, verbose=1)
            ]
            model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_INIT"], callbacks=callbacks, verbose=2)

        # 5. Phase 2: Fine-Tuning
        model.load_weights(ckpt_init) # Ensure we start from best frozen weights
        model.layers[1].trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

        ckpt_ft = os.path.join(ckpt_dir, f"{model_name}_fold{fold}_finetune.keras")
        if os.path.exists(ckpt_ft):
             print(f"Loading existing finetuned model: {ckpt_ft}")
             model.load_weights(ckpt_ft)
        else:
            callbacks_ft = [
                ModelCheckpoint(ckpt_ft, monitor="val_mae", save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor="val_mae", factor=0.2, patience=4, verbose=1),
                EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True, verbose=1)
            ]
            model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FT"], callbacks=callbacks_ft, verbose=2)

        # 6. Evaluate Fold (Image Level)
        val_ids_ds = patch_data_tf_dataset_with_ids_from_df(val_df, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False)
        
        preds = []
        for patches, _, img_ids in val_ids_ds:
            p = model.predict(patches, verbose=0).ravel()
            preds.extend(zip(p, img_ids.numpy()))

        y_pred, y_true = group_predictions_by_image_id(preds, val_df)
        metrics = compute_evaluation_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        print(f"Fold {fold} Metrics: MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.2f}")

        # Cleanup
        del model, train_ds, val_ds, val_ids_ds
        gc.collect()
        tf.keras.backend.clear_session()

    return fold_metrics

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
            ckpt_path = os.path.join(CONFIG["MODELS_DIR"], model_name, f"{model_name}_fold{fold}_finetune.keras")
            
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

def main():
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    df_full = pd.read_csv(CONFIG["CSV_PATH"])
    
    # 1. Run Cross-Validation Training
    models_to_run = {
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'DenseNet121': DenseNet121,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M
    }
    
    for name, architecture in models_to_run.items():
        run_cv(df_full, architecture, name, n_splits=5)

    # 2. Run Inference on Test Set
    # We use ALL 5 folds for ALL models
    inference_map = {
        'ResNet50': [1, 2, 3, 4, 5],
        'InceptionV3': [1, 2, 3, 4, 5],
        'DenseNet121': [1, 2, 3, 4, 5],
        'InceptionResNetV2': [1, 2, 3, 4, 5],
        'EfficientNetV2M': [1, 2, 3, 4, 5]
    }
    
    test_df = df_full[df_full['Set'] == 'test'].reset_index(drop=True)
    predict_on_test_set(inference_map, test_df)

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()