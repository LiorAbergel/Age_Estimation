"""
Experiment 03: Enhanced SOTA Ensemble with Advanced Augmentation

Overview:
This experiment builds upon the previous ensemble approach by introducing 
aggressive data augmentation and a comprehensive ensemble weight optimization 
strategy (Grid Search vs. MAE-based).

Key Features:
- Advanced Augmentation: Random Rotation, Zoom, Brightness, and Gaussian Noise.
- Architectures: ResNet50, InceptionV3, InceptionResNetV2, DenseNet121, EfficientNetV2M.
- Evaluation: Patch-level prediction caching and post-hoc weight optimization.
"""

import os
import csv
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import (
    ResNet50, InceptionV3, InceptionResNetV2, DenseNet121
)
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, cohen_kappa_score

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 32, # 128 often causes OOM on 400x400 patches. Adjust if you have high VRAM.
    "EPOCHS_FROZEN": 50,
    "EPOCHS_FINE_TUNE": 10,
    "DATA_DIR": "./data",  
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/experiment_03_augmented",
    "RESULTS_DIR": "./results/experiment_03_augmented",
    "PREDICTIONS_CSV": "patch_level_predictions.csv",
    "SUMMARY_CSV": "ensemble_evaluation_summary.csv"
}

# --- Data Processing Functions ---

def calculate_resized_dimensions(height, width, patch_size=400, stride=200, standard_size=800):
    """
    Calculates dimensions to maintain aspect ratio and compatibility with patch extraction.
    """
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

def read_image_and_resize(img_path):
    """
    Reads image, resizes dynamically, and normalizes.
    """
    try:
        img_path_str = img_path.numpy().decode("utf-8")
        img = Image.open(img_path_str).convert('RGB')
        w, h = img.size
        new_h, new_w = calculate_resized_dimensions(h, w, CONFIG["PATCH_SIZE"][0], CONFIG["STRIDE"])
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        return np.zeros((CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3), dtype=np.float32)

def process_image(row, data_dir, include_id=False):
    """
    Loads image, extracts patches, and associates labels.
    """
    img_path = tf.strings.join([data_dir, row['File']], separator=os.sep)
    img = tf.py_function(func=read_image_and_resize, inp=[img_path], Tout=tf.float32)
    img.set_shape([None, None, 3]) 

    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 1],
        strides=[1, CONFIG["STRIDE"], CONFIG["STRIDE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [-1, CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3])
    labels = tf.fill([tf.shape(patches)[0]], row['Age'])
    
    if include_id:
        ids = tf.fill([tf.shape(patches)[0]], row['File'])
        return patches, labels, ids
    return patches, labels

# --- Augmentation ---

# Pre-instantiate layer to avoid retracing
rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
    """
    Applies Rotation, Zoom/Crop, Brightness, and Gaussian Noise.
    """
    # 1. Random Rotation
    image = rotation_layer(image, training=True)

    # 2. Random Zoom (Scale 0.9 to 1.1)
    orig_shape = tf.shape(image)[:2]
    zoom_factor = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    # 3. Random Brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # 4. Gaussian Noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def create_dataset(data_dir, labels_df, dataset_type, augment=False, include_id=False):
    """
    Creates tf.data.Dataset with optional advanced augmentation.
    """
    subset_df = labels_df[labels_df['Set'] == dataset_type].reset_index(drop=True)
    target_dir = os.path.join(data_dir, dataset_type)
    
    ds = tf.data.Dataset.from_tensor_slices(dict(subset_df))

    if include_id:
        ds = ds.map(lambda row: process_image(row, target_dir, include_id=True), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda patches, labels, ids: tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(patches),
             tf.data.Dataset.from_tensor_slices(labels),
             tf.data.Dataset.from_tensor_slices(ids))
        ))
    else:
        ds = ds.map(lambda row: process_image(row, target_dir, include_id=False), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.flat_map(lambda patches, labels: tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(patches),
             tf.data.Dataset.from_tensor_slices(labels))
        ))

    if augment:
        ds = ds.map(advanced_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Model Building ---

def build_sota_model(base_model_fn, input_shape=(400, 400, 3), dropout_rate=0.5):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False 
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x) 
    model = Model(inputs, outputs)
    return model, base_model

# --- Evaluation Metrics ---

def compute_evaluation_metrics(true_images, predicted_images):
    mae = mean_absolute_error(true_images, predicted_images)
    rmse = np.sqrt(mean_squared_error(true_images, predicted_images))
    r2 = r2_score(true_images, predicted_images)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_images - predicted_images) / true_images)) * 100
        if np.isnan(mape): mape = 0.0

    errors = np.abs(true_images - predicted_images)
    within_2 = np.mean(errors <= 2) * 100
    within_5 = np.mean(errors <= 5) * 100
    within_10 = np.mean(errors <= 10) * 100
    
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape,
        "Within ±2 Years (%)": within_2, "Within ±5 Years (%)": within_5,
        "Within ±10 Years (%)": within_10,
        "Max Error": np.max(errors), "Median Error": np.median(errors)
    }

# --- Main Execution Flow ---

def train_models(train_ds, val_ds):
    models_to_train = {
        'ResNet50': ResNet50,
        'DenseNet121': DenseNet121,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M
    }
    
    trained_models = {}
    
    for name, architecture in models_to_train.items():
        print(f"\n{'='*30}\nTraining {name}\n{'='*30}")
        save_path = os.path.join(CONFIG["MODELS_DIR"], f"{name}_best_model.keras")
        
        if os.path.exists(save_path):
            print(f"Model found at {save_path}, loading...")
            trained_models[name] = load_model(save_path)
            continue

        model, base_model = build_sota_model(architecture)
        
        callbacks = [
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        ]

        # Phase 1: Frozen
        print("Phase 1: Frozen Training")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FROZEN"], callbacks=callbacks)

        # Phase 2: Fine-Tuning
        print("Phase 2: Fine-Tuning")
        base_model.trainable = True
        # It's often good practice to re-compile with a lower LR
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FINE_TUNE"], callbacks=callbacks)
        
        trained_models[name] = model
        
    return trained_models

def generate_predictions_csv(trained_models, test_ds):
    csv_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["PREDICTIONS_CSV"])
    
    if os.path.exists(csv_path):
        print(f"Predictions CSV already exists at {csv_path}. Skipping inference.")
        return pd.read_csv(csv_path)

    print("\nGenerating Patch-Level Predictions (Inference)...")
    
    # We will write to CSV iteratively to save memory
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'ImageID', 'PatchIndex', 'Prediction'])
        
        for model_name, model in trained_models.items():
            print(f"Inferencing {model_name}...")
            patch_idx = 0
            for batch in tqdm(test_ds, desc=model_name):
                patches, _, image_ids = batch
                preds = model.predict(patches, verbose=0).flatten()
                
                # Write batch
                for i in range(len(preds)):
                    writer.writerow([
                        model_name,
                        image_ids.numpy()[i].decode('utf-8'),
                        patch_idx + i,
                        preds[i]
                    ])
                patch_idx += len(preds)
                
    print(f"Predictions saved to {csv_path}")
    return pd.read_csv(csv_path)

def weighted_ensemble_from_row(row, weights, group_models):
    ensemble_pred = 0.0
    total_weight = 0.0
    for model in group_models:
        if model in row and pd.notna(row[model]):
            ensemble_pred += row[model] * weights[model]
            total_weight += weights[model]
    return ensemble_pred / total_weight if total_weight > 0 else np.nan

def optimize_ensembles(pivot_df, true_age_dict):
    print("\n=== Ensemble Optimization (Grid Search & MAE-Based) ===")
    
    # Calculate Individual Model MAEs first
    model_maes = {}
    cols = [c for c in pivot_df.columns if c != 'ImageID']
    for model in cols:
        temp = pivot_df[['ImageID', model]].dropna()
        y_true = [true_age_dict[mid] for mid in temp['ImageID'] if mid in true_age_dict]
        y_pred = [val for mid, val in zip(temp['ImageID'], temp[model]) if mid in true_age_dict]
        model_maes[model] = mean_absolute_error(y_true, y_pred)
        print(f"{model} MAE: {model_maes[model]:.2f}")

    # Define Groups
    ensemble_groups = {
        'Full Ensemble': ['ResNet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'EfficientNetV2M'],
        'Best 4': ['ResNet50', 'InceptionResNetV2', 'DenseNet121', 'InceptionV3'], 
        'Best 3': ['ResNet50', 'InceptionResNetV2', 'DenseNet121'],       
        'Best 2': ['ResNet50', 'InceptionResNetV2']             
    }
    
    results_summary = []

    for group_name, group_models in ensemble_groups.items():
        print(f"\nProcessing Group: {group_name}")
        
        # 1. MAE Based Weights
        total_mae = sum(model_maes[m] for m in group_models)
        n = len(group_models)
        # Formula: weight_i = (Total - MAE_i) / ((n-1)*Total)
        mae_weights = {m: (total_mae - model_maes[m]) / ((n - 1) * total_mae) for m in group_models}
        
        # 2. Grid Search Weights
        # Generate valid combinations that sum to 1.0 (step 0.1)
        grid_step = 0.1
        weight_ranges = [np.arange(0.1, 1.0, grid_step) for _ in group_models]
        valid_combos = []
        for combo in itertools.product(*weight_ranges):
            if np.isclose(sum(combo), 1.0, atol=1e-5):
                valid_combos.append(dict(zip(group_models, combo)))
        
        best_grid_weights = None
        best_grid_mae = float('inf')
        
        # Run Grid Search on CPU (using DataFrame)
        for weights in valid_combos:
            # Quick vector calculation
            df_temp = pivot_df.copy()
            df_temp['Ensemble'] = df_temp.apply(lambda r: weighted_ensemble_from_row(r, weights, group_models), axis=1)
            df_temp = df_temp.dropna(subset=['Ensemble'])
            
            y_t = [true_age_dict[mid] for mid in df_temp['ImageID'] if mid in true_age_dict]
            y_p = df_temp['Ensemble'].tolist()
            
            curr_mae = mean_absolute_error(y_t, y_p)
            if curr_mae < best_grid_mae:
                best_grid_mae = curr_mae
                best_grid_weights = weights

        # Evaluate both methods fully
        for method, w_dict in [('Grid Search', best_grid_weights), ('MAE-based', mae_weights)]:
            # Final Eval
            df_temp = pivot_df.copy()
            df_temp['Ensemble'] = df_temp.apply(lambda r: weighted_ensemble_from_row(r, w_dict, group_models), axis=1)
            df_temp = df_temp.dropna(subset=['Ensemble'])
            
            y_t = np.array([true_age_dict[mid] for mid in df_temp['ImageID'] if mid in true_age_dict])
            y_p = np.array(df_temp['Ensemble'].tolist())
            
            metrics = compute_evaluation_metrics(y_t, y_p)
            
            row = {
                "Ensemble Group": group_name,
                "Method": method,
                "Weights": str(w_dict)
            }
            row.update(metrics)
            results_summary.append(row)

    return pd.DataFrame(results_summary)

def main():
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # 1. Load Data
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])

    # 2. Datasets
    print("Creating Datasets...")
    # NOTE: Augmentation is True for training
    train_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'train', augment=True)
    val_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'val', augment=False)
    test_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'test', augment=False, include_id=True)
    
    # 3. Train
    trained_models = train_models(train_ds, val_ds)

    # 4. Generate Predictions (Save to CSV)
    # Reload models if we skipped training to ensure we have the objects for inference
    if not trained_models:
        # Load from disk for inference
        from tensorflow.keras.models import load_model
        model_names = ['ResNet50', 'DenseNet121', 'InceptionV3', 'InceptionResNetV2', 'EfficientNetV2M']
        for name in model_names:
            path = os.path.join(CONFIG["MODELS_DIR"], f"{name}_best_model.keras")
            if os.path.exists(path):
                trained_models[name] = load_model(path)

    generate_predictions_csv(trained_models, test_ds)
    
    # 5. Optimization & Final Evaluation
    print("\nReading predictions for Ensemble Optimization...")
    pred_csv_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["PREDICTIONS_CSV"])
    df = pd.read_csv(pred_csv_path)
    
    # Pivot: Rows=Images, Cols=Models
    avg_df = df.groupby(['Model', 'ImageID'])['Prediction'].mean().reset_index()
    pivot_df = avg_df.pivot(index='ImageID', columns='Model', values='Prediction').reset_index()
    
    # True Age Dict
    true_age_dict = dict(zip(labels_data['File'], labels_data['Age']))
    
    # Run Optimization
    summary_df = optimize_ensembles(pivot_df, true_age_dict)
    
    # Save Summary
    summary_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["SUMMARY_CSV"])
    summary_df.to_csv(summary_path, index=False)
    
    # Print
    pd.set_option('display.max_columns', None)
    print("\n=== Final Evaluation Summary ===")
    print(summary_df)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()