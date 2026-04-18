"""
Experiment 04: SOTA CNNs with Label Scaling and LR Scheduling

Overview:
This experiment optimizes the training dynamics by:
1. Scaling target labels (Age) to [0, 1] range to aid convergence.
2. Using an Exponential Decay Learning Rate Scheduler.
3. Applying the previously defined Advanced Augmentation pipeline.

Architecture:
- Models: ResNet50, InceptionV3, DenseNet121, EfficientNetV2M
- Loss: MSE (on scaled labels)
- Metrics: MAE (converted back to years for reporting)
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import (
    ResNet50, InceptionV3, InceptionResNetV2, DenseNet121
)
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    "BATCH_SIZE": 32,
    "EPOCHS_FROZEN": 50,
    "EPOCHS_FINE_TUNE": 20,
    "DATA_DIR": "./data",  
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/experiment_04_scaled",
    "RESULTS_DIR": "./results/experiment_04_scaled",
    "PREDICTIONS_CSV": "scaled_predictions.csv"
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

def read_image_and_resize(img_path):
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

def process_image(row, data_dir, label_col='ScaledAge', include_id=False):
    """
    Reads image, extracts patches, and returns (Patch, ScaledLabel).
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
    
    # Use the SCALED age here
    labels = tf.fill([tf.shape(patches)[0]], row[label_col])
    
    if include_id:
        ids = tf.fill([tf.shape(patches)[0]], row['File'])
        return patches, labels, ids
    return patches, labels

# --- Augmentation ---

rotation_layer = tf.keras.layers.RandomRotation(factor=(-0.04167, 0.04167))
translation_layer = tf.keras.layers.RandomTranslation(
        height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=1.0)

def advanced_augmentation(image, label):
    # Rotation
    image = rotation_layer(image, training=True)
    
    # Zoom
    orig_shape = tf.shape(image)[:2]
    zoom_factor = tf.random.uniform([], 1.0, 1.2) # Zoom in only
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    # Translation
    image = translation_layer(image, training=True)
    
    # Brightness & Contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.2)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def create_dataset(data_dir, labels_df, dataset_type, augment=False, include_id=False):
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

# --- Evaluation Metrics (Inverse Transform Logic) ---

def compute_evaluation_metrics(true_scaled, pred_scaled, scaler):
    # Inverse transform
    y_true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    errors = np.abs(y_true - y_pred)
    within_5 = np.mean(errors <= 5) * 100
    
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | Acc(5yr): {within_5:.2f}%")
    return mae, rmse, y_true, y_pred

# --- Main ---

def main():
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    # 1. Load Data & Scale Labels
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])
    
    # Initialize and Fit Scaler
    print("Fitting Scaler on Age column...")
    scaler = MinMaxScaler()
    labels_data['ScaledAge'] = scaler.fit_transform(labels_data[['Age']])
    
    # 2. Datasets
    print("Creating Datasets...")
    train_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'train', augment=True)
    val_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'val', augment=False)
    test_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'test', augment=False, include_id=True)
    
    # 3. Training Loop
    models_map = {
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'DenseNet121': DenseNet121,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M
    }
    
    trained_models = {}
    
    for name, architecture in models_map.items():
        print(f"\n{'='*30}\nTraining {name} (Scaled)\n{'='*30}")
        save_path = os.path.join(CONFIG["MODELS_DIR"], f"{name}_best_model.keras")
        
        # Check resume
        if os.path.exists(save_path):
            print(f"Loading existing model from {save_path}")
            trained_models[name] = load_model(save_path)
            continue
            
        model, base_model = build_sota_model(architecture)
        
        # Callbacks
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        
        # --- Phase 1: Frozen with Scheduler ---
        print("Phase 1: Frozen Backbone (Exponential Decay LR)")
        
        # 
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FROZEN"], 
                  callbacks=[checkpoint, early_stopping])
        
        # --- Phase 2: Fine-Tuning (Fixed Low LR) ---
        print("Phase 2: Fine-Tuning (Fixed LR 1e-4)")
        base_model.trainable = True
        
        optimizer_ft = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer_ft, loss='mse', metrics=['mae'])
        
        model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FINE_TUNE"], 
                  callbacks=[checkpoint, early_stopping])
        
        trained_models[name] = model

    # 4. Evaluation & Inverse Scaling
    print("\nGenerating Predictions (Inverse Scaled)...")
    csv_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["PREDICTIONS_CSV"])
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'ImageID', 'PatchIndex', 'Prediction_Scaled'])
        
        for model_name, model in trained_models.items():
            print(f"Inferencing {model_name}...")
            patch_idx = 0
            for batch in tqdm(test_ds):
                patches, _, image_ids = batch
                # Predict (Outputs are 0-1)
                preds_scaled = model.predict(patches, verbose=0).flatten()
                
                for i in range(len(preds_scaled)):
                    writer.writerow([
                        model_name,
                        image_ids.numpy()[i].decode('utf-8'),
                        patch_idx + i,
                        preds_scaled[i]
                    ])
                patch_idx += len(preds_scaled)

    print(f"Predictions saved to {csv_path}")
    
    # 5. Quick Aggregate Report
    # We load the CSV, aggregate by image, then Inverse Transform
    print("\n--- Final Metrics (Aggregated by Image) ---")
    df = pd.read_csv(csv_path)
    true_age_map = dict(zip(labels_data['File'], labels_data['Age']))
    
    for model_name in trained_models.keys():
        model_df = df[df['Model'] == model_name]
        # Average scaled predictions per image
        img_preds = model_df.groupby('ImageID')['Prediction_Scaled'].mean()
        
        y_true_final = []
        y_pred_scaled_final = []
        
        for img_id, pred_scaled in img_preds.items():
            if img_id in true_age_map:
                y_true_final.append(true_age_map[img_id])
                y_pred_scaled_final.append(pred_scaled)
        
        # INVERSE TRANSFORM
        y_pred_final = scaler.inverse_transform(
            np.array(y_pred_scaled_final).reshape(-1, 1)
        ).flatten()
        
        print(f"\nModel: {model_name}")
        mae = mean_absolute_error(y_true_final, y_pred_final)
        print(f"MAE: {mae:.2f} years")

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()