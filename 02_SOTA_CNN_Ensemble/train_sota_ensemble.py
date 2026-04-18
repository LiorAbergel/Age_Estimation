"""
Experiment 02: SOTA CNN Ensemble for Age Estimation

Overview:
This script evaluates industry-standard architectures (ResNet50, InceptionV3, 
InceptionResNetV2, DenseNet121, EfficientNetV2M) using Transfer Learning 
and a patch-based approach.

Key Features:
- Patch-Based: 400x400 patches (stride 200) to capture local stroke details.
- Transfer Learning: Two-stage training (Frozen backbone -> Fine-tuning).
- Ensemble Strategy: Simple Average of predictions from all 5 models.
"""

import os
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
    # Reduced batch size to prevent OOM on standard GPUs (increase to 64/128 if you have >24GB VRAM)
    "BATCH_SIZE": 32, 
    "EPOCHS_FROZEN": 50,
    "EPOCHS_FINE_TUNE": 10,
    "LR_FROZEN": 1e-3,
    "LR_FINE_TUNE": 1e-4,
    "DATA_DIR": "./data",  
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/experiment_02",
    "RESULTS_DIR": "./results",
    "AUGMENT": True
}

def calculate_resized_dimensions(height, width, patch_size=400, stride=200, standard_size=800):
    """
    Calculates dimensions to maintain aspect ratio and compatibility with patch extraction.
    """
    aspect_ratio = width / height

    # Scale so smaller side is standard_size
    if height < width:
        new_height = standard_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = standard_size
        new_height = int(new_width / aspect_ratio)

    # Adjust to ensure coverage by patches
    def adjust_dimension(dim):
        remainder = (dim - patch_size) % stride
        return dim if remainder == 0 else dim - remainder

    return adjust_dimension(new_height), adjust_dimension(new_width)

def read_image_and_resize(img_path):
    """
    Reads image using PIL, resizes dynamically, and normalizes to [0,1].
    Wrapped in tf.py_function for use in TF pipeline.
    """
    try:
        img_path_str = img_path.numpy().decode("utf-8")
        img = Image.open(img_path_str)
        img = img.convert('RGB')
        w, h = img.size
        
        new_h, new_w = calculate_resized_dimensions(
            h, w, 
            CONFIG["PATCH_SIZE"][0], 
            CONFIG["STRIDE"]
        )
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        # Return a dummy black image to prevent pipeline crash, filter out later if needed
        return np.zeros((CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3), dtype=np.float32)

def process_image(row, data_dir, include_id=False):
    """
    Loads image, extracts patches, and associates labels.
    """
    img_path = tf.strings.join([data_dir, row['File']], separator=os.sep)

    # Use py_function to call PIL logic
    img = tf.py_function(func=read_image_and_resize, inp=[img_path], Tout=tf.float32)
    img.set_shape([None, None, 3]) 

    # Extract Patches
    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 1],
        strides=[1, CONFIG["STRIDE"], CONFIG["STRIDE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Reshape to (Num_Patches, 400, 400, 3)
    patches = tf.reshape(patches, [-1, CONFIG["PATCH_SIZE"][0], CONFIG["PATCH_SIZE"][1], 3])
    
    # Replicate labels for all patches
    labels = tf.fill([tf.shape(patches)[0]], row['Age'])
    
    if include_id:
        ids = tf.fill([tf.shape(patches)[0]], row['File'])
        return patches, labels, ids
    
    return patches, labels

def create_dataset(data_dir, labels_df, dataset_type, augment=False, include_id=False):
    """
    Creates a tf.data.Dataset that yields individual patches.
    """
    subset_df = labels_df[labels_df['Set'] == dataset_type].reset_index(drop=True)
    target_dir = os.path.join(data_dir, dataset_type)
    
    ds = tf.data.Dataset.from_tensor_slices(dict(subset_df))

    # 1. Load Image & Extract Patches
    if include_id:
        ds = ds.map(lambda row: process_image(row, target_dir, include_id=True), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        # Flatten: (Batch_of_Patches) -> (Patch)
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

    # 2. Augmentation (Training Only)
    if augment:
        def augment_fn(patch, label):
            patch = tf.image.random_flip_left_right(patch)
            return patch, label
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Batching
    ds = ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    return ds

def build_regression_model(base_model_class, input_shape=(400, 400, 3)):
    """
    Wraps a pre-trained backbone with a regression head.
    """
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False) # Important: keep BatchNormalization in inference mode
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def ensemble_predict(models, test_ds):
    """
    Generates predictions using simple averaging of all models.
    """
    results = []
    
    print("\nGenerating Ensemble Predictions...")
    # Note: We iterate over the dataset once.
    # For a massive dataset, it's efficient to predict one batch at a time for all models.
    
    for batch in tqdm(test_ds, desc="Inference"):
        patches, labels, ids = batch
        
        # Get predictions from all models
        batch_preds = []
        for name, model in models.items():
            p = model.predict(patches, verbose=0)
            batch_preds.append(p)
        
        # Average across models (Ensemble)
        # Shape: (Num_Models, Batch, 1) -> Mean -> (Batch, 1)
        avg_preds = np.mean(batch_preds, axis=0).flatten()
        
        # Store results
        current_ids = [i.decode('utf-8') for i in ids.numpy()]
        current_labels = labels.numpy()
        
        for pred, file_id, true_val in zip(avg_preds, current_ids, current_labels):
            results.append((pred, file_id, true_val))
            
    return results

def aggregate_predictions(raw_results):
    """
    Aggregates patch-level predictions back to image-level using the mean.
    """
    img_preds = defaultdict(list)
    img_truth = {}
    
    for pred, file_id, true_val in raw_results:
        img_preds[file_id].append(pred)
        img_truth[file_id] = true_val
        
    final_preds = []
    final_truth = []
    file_ids = []
    
    for file_id in img_preds:
        # Mean of all patches for this image
        final_preds.append(np.mean(img_preds[file_id]))
        final_truth.append(img_truth[file_id])
        file_ids.append(file_id)
        
    return np.array(final_truth), np.array(final_preds), file_ids

def main():
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # --- Load Data ---
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}")
        return

    labels_data = pd.read_csv(CONFIG["CSV_PATH"])
    
    print("Creating Datasets...")
    train_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'train', augment=CONFIG["AUGMENT"])
    val_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'val', augment=False)
    # Test set includes IDs
    test_ds = create_dataset(CONFIG["DATA_DIR"], labels_data, 'test', augment=False, include_id=True)
    print("Datasets Ready.")

    # --- Training Loop ---
    MODEL_ARCHITECTURES = {
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'DenseNet121': DenseNet121,
        'EfficientNetV2M': EfficientNetV2M
    }

    trained_models = {}

    for name, architecture in MODEL_ARCHITECTURES.items():
        print(f"\n{'='*40}")
        print(f"Processing Model: {name}")
        print(f"{'='*40}")
        
        save_path = os.path.join(CONFIG["MODELS_DIR"], f"{name}_best.keras")
        
        # SKIP LOGIC: Check if model already exists to save time on re-runs
        if os.path.exists(save_path):
            print(f"Found existing model at {save_path}. Loading...")
            try:
                model = load_model(save_path)
                trained_models[name] = model
                continue
            except Exception as e:
                print(f"Error loading {name}, retraining. Error: {e}")

        # Build Model
        model, base_model = build_regression_model(architecture)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        ]
        
        # --- Phase 1: Frozen Backbone ---
        print(f"Phase 1: Frozen Training ({CONFIG['EPOCHS_FROZEN']} epochs)")
        base_model.trainable = False
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["LR_FROZEN"]),
                      loss='mse', metrics=['mae'])
        
        history_frozen = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FROZEN"], callbacks=callbacks)
        
        # --- Phase 2: Fine-Tuning ---
        print(f"Phase 2: Fine-Tuning ({CONFIG['EPOCHS_FINE_TUNE']} epochs)")
        base_model.trainable = True
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["LR_FINE_TUNE"]),
                      loss='mse', metrics=['mae'])
        
        history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_FINE_TUNE"], callbacks=callbacks)
        
        trained_models[name] = model
        print(f"Finished {name}.")

    # --- Evaluation ---
    if not trained_models:
        print("No models available for inference.")
        return

    # Run Ensemble Inference
    raw_results = ensemble_predict(trained_models, test_ds)

    # Aggregate
    y_true, y_pred, file_ids = aggregate_predictions(raw_results)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    errors = np.abs(y_true - y_pred)
    within_5 = np.mean(errors <= 5) * 100

    print(f"\n--- Experiment 02 Results (SOTA Ensemble) ---")
    print(f"MAE:  {mae:.2f} years")
    print(f"RMSE: {rmse:.2f} years")
    print(f"R²:   {r2:.4f}")
    print(f"Accuracy (±5 years): {within_5:.2f}%")

    # --- Save Results ---
    # 1. Plot
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Error Distribution (Ensemble)')
    plt.xlabel('Absolute Error (Years)')
    plot_path = os.path.join(CONFIG["RESULTS_DIR"], 'ensemble_error_dist.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # 2. CSV
    results_df = pd.DataFrame({
        'File': file_ids,
        'True Age': y_true,
        'Predicted Age': np.round(y_pred, 2)
    })
    csv_save_path = os.path.join(CONFIG["RESULTS_DIR"], 'ensemble_predictions.csv')
    results_df.to_csv(csv_save_path, index=False)
    print(f"Predictions saved to {csv_save_path}")

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()