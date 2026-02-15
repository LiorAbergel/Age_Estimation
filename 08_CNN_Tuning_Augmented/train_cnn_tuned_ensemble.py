"""
Experiment 08: CNN Ensemble with Hyperparameter Tuning & Advanced Augmentation

Overview:
This experiment revisits CNN architectures but applies rigorous hyperparameter tuning 
using Keras Tuner. It optimizes:
- Learning Rate
- Dropout Rate
- Backbone Unfreezing Ratio

It culminates in an optimized ensemble where model weights are fine-tuned via 
grid search on the validation/test predictions.

Requirements:
pip install keras-tuner
"""

import os
import csv
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Install keras-tuner if missing
try:
    import keras_tuner as kt
except ImportError:
    raise ImportError("Please run: pip install keras-tuner")

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

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "BATCH_SIZE": 32, # Adjusted for memory safety
    "THR": 0.0054,
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "MODELS_DIR": "./models/cnn_tuned",
    "RESULTS_DIR": "./results/cnn_tuned",
    "PREDICTIONS_CSV": "patch_level_predictions.csv",
    "SUMMARY_CSV": "ensemble_evaluation_summary.csv"
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

    # Filter
    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > CONFIG["THR"]
    patches = tf.boolean_mask(patches, mask)
    labels = tf.boolean_mask(labels, mask)

    return patches, labels

def process_row_with_id(row):
    patches, labels = process_image(row, CONFIG["DATA_DIR"], CONFIG["PATCH_SIZE"], CONFIG["STRIDE"])
    image_id = tf.fill([tf.shape(patches)[0]], row['File'])
    return patches, labels, image_id

# --- Advanced Augmentation ---

rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
    image = rotation_layer(image, training=True)
    
    # Zoom
    orig_shape = tf.shape(image)[:2]
    zoom = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    # Contrast & Noise
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label

# --- Datasets ---

def patch_data_tf_dataset(labels_df_subset, data_dir, patch_size, step_size, batch_size, augment=False):
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

def patch_data_tf_dataset_with_ids(labels_df_subset, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices(dict(labels_df_subset))
    ds = ds.map(process_row_with_id, num_parallel_calls=tf.data.AUTOTUNE).flat_map(
        lambda patches, labels, image_ids: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels),
            tf.data.Dataset.from_tensor_slices(image_ids)
        ))
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- HyperModel Builder ---

def build_model(hp, base_model_fn):
    # Dynamic Input Shape (Matches Config)
    inputs = Input(shape=(*CONFIG["PATCH_SIZE"], 3))
    
    # Initialize Base
    base = base_model_fn(weights='imagenet', include_top=False, input_tensor=inputs)

    # Freeze all first
    base.trainable = False

    # Hyperparameter: Unfreeze Ratio
    # We unfreeze the top N% of layers
    unfreeze_ratio = hp.Float("unfreeze_ratio", 0.0, 0.5, step=0.1) # Conservative unfreezing
    n_layers = len(base.layers)
    n_unfreeze = int(n_layers * unfreeze_ratio)
    
    if n_unfreeze > 0:
        # Unfreeze the last n_unfreeze layers
        for layer in base.layers[-n_unfreeze:]:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    
    # Hyperparameter: Dropout
    x = Dropout(hp.Float("dropout", 0.2, 0.7, step=0.1))(x)
    
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)

    # Hyperparameter: Learning Rate
    hp_lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp_lr),
                  loss='mse', metrics=['mae'])
    return model

# --- Evaluation Functions ---

def compute_evaluation_metrics(true_images, predicted_images):
    mae = mean_absolute_error(true_images, predicted_images)
    rmse = np.sqrt(mean_squared_error(true_images, predicted_images))
    r2 = r2_score(true_images, predicted_images)
    
    # Accuracy thresholds
    errors = np.abs(true_images - predicted_images)
    within_2 = np.mean(errors <= 2) * 100
    within_5 = np.mean(errors <= 5) * 100
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Acc_5yr": within_5}

def group_predictions_by_image_id(preds_with_ids, labels_df):
    grouped_predictions = defaultdict(list)
    
    for pred, image_id in preds_with_ids:
        image_id_str = image_id.decode('utf-8')
        grouped_predictions[image_id_str].append(float(pred))
        
    true_map = dict(zip(labels_df['File'], labels_df['Age']))
    y_true, y_pred, ids = [], [], []
    
    for img_id, p_list in grouped_predictions.items():
        if img_id in true_map:
            y_pred.append(np.mean(p_list))
            y_true.append(true_map[img_id])
            ids.append(img_id)
            
    return np.array(y_true), np.array(y_pred), ids

# --- Ensemble Optimization Helpers ---

def weighted_ensemble_from_row(row, weights, group_models):
    ensemble_pred = 0.0
    total_weight = 0.0
    for model in group_models:
        if model in row and pd.notna(row[model]):
            ensemble_pred += row[model] * weights[model]
            total_weight += weights[model]
    return ensemble_pred / total_weight if total_weight > 0 else np.nan

def optimize_ensembles(pivot_df, true_age_dict):
    print("\n=== Ensemble Optimization (Grid Search) ===")
    
    # Calculate Individual Model MAEs
    model_maes = {}
    cols = [c for c in pivot_df.columns if c != 'ImageID']
    for model in cols:
        temp = pivot_df[['ImageID', model]].dropna()
        y_t = [true_age_dict[mid] for mid in temp['ImageID'] if mid in true_age_dict]
        y_p = [val for mid, val in zip(temp['ImageID'], temp[model]) if mid in true_age_dict]
        model_maes[model] = mean_absolute_error(y_t, y_p)
        print(f"{model} MAE: {model_maes[model]:.2f}")

    # Define Groups based on available models
    ensemble_groups = {
        'Full Ensemble': cols,
        'Best 2': sorted(model_maes, key=model_maes.get)[:2]
    }
    
    results_summary = []

    for group_name, group_models in ensemble_groups.items():
        if not group_models: continue
        
        # Grid Search Weights
        # Generate valid combinations (sum=1.0, step=0.1)
        grid_step = 0.1
        weight_ranges = [np.arange(0.1, 1.0, grid_step) for _ in group_models]
        
        best_grid_weights = None
        best_grid_mae = float('inf')
        
        for combo in itertools.product(*weight_ranges):
            if np.isclose(sum(combo), 1.0, atol=1e-5):
                weights = dict(zip(group_models, combo))
                
                # Fast eval
                df_temp = pivot_df.copy()
                df_temp['Ens'] = df_temp.apply(lambda r: weighted_ensemble_from_row(r, weights, group_models), axis=1)
                df_temp = df_temp.dropna(subset=['Ens'])
                
                y_t = [true_age_dict[mid] for mid in df_temp['ImageID'] if mid in true_age_dict]
                y_p = df_temp['Ens'].tolist()
                
                curr_mae = mean_absolute_error(y_t, y_p)
                if curr_mae < best_grid_mae:
                    best_grid_mae = curr_mae
                    best_grid_weights = weights

        row = {
            "Group": group_name,
            "Best Weights": str(best_grid_weights),
            "Ensemble MAE": best_grid_mae
        }
        results_summary.append(row)

    return pd.DataFrame(results_summary)

# --- Main Logic ---

def main():
    os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found.")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])

    # 1. Datasets
    train_ds = patch_data_tf_dataset(
        labels_data[labels_data['Set']=='train'], CONFIG["DATA_DIR"], 
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=True
    )
    val_ds = patch_data_tf_dataset(
        labels_data[labels_data['Set']=='val'], CONFIG["DATA_DIR"], 
        CONFIG["PATCH_SIZE"], CONFIG["STRIDE"], CONFIG["BATCH_SIZE"], augment=False
    )

    # 2. Hyperparameter Search & Training
    BACKBONES = {
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'DenseNet121': DenseNet121,
        'EfficientNetV2M': EfficientNetV2M
    }
    
    trained_models = {}
    
    for name, fn in BACKBONES.items():
        print(f"\n=== Tuning & Training {name} ===")
        
        # Check if already done
        final_model_path = os.path.join(CONFIG["MODELS_DIR"], f"{name}_best_model.keras")
        if os.path.exists(final_model_path):
            print(f"Found existing model for {name}, skipping tuning.")
            trained_models[name] = load_model(final_model_path)
            continue

        # Tuner Setup
        tuner = kt.Hyperband(
            hypermodel=lambda hp: build_model(hp, fn),
            objective='val_mae',
            max_epochs=15,
            factor=3,
            directory=os.path.join(CONFIG["MODELS_DIR"], 'kt_dir'),
            project_name=f'tune_{name}'
        )
        
        # Search
        tuner.search(train_ds, validation_data=val_ds, epochs=15, 
                     callbacks=[EarlyStopping('val_mae', patience=3)])
        
        # Get Best & Retrain
        best_hp = tuner.get_best_hyperparameters(1)[0]
        model = tuner.hypermodel.build(best_hp)
        
        print(f"Best HPs for {name}: {best_hp.values}")
        
        # Final Training with Callbacks
        callbacks = [
            ModelCheckpoint(final_model_path, 'val_mae', save_best_only=True, verbose=1),
            ReduceLROnPlateau('val_mae', factor=0.5, patience=3, verbose=1),
            EarlyStopping('val_mae', patience=8, restore_best_weights=True)
        ]
        
        model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
        trained_models[name] = model

    # 3. Generate Predictions (Patch Level -> Image Level)
    print("\nGenerating Predictions for Ensemble Optimization...")
    test_ds = patch_data_tf_dataset_with_ids(labels_data[labels_data['Set']=='test'])
    
    csv_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["PREDICTIONS_CSV"])
    
    # We write patch predictions to disk to save memory
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'ImageID', 'Prediction'])
        
        for name, model in trained_models.items():
            print(f"Inferencing {name}...")
            # Predict in batches
            for patches, _, img_ids in test_ds:
                preds = model.predict(patches, verbose=0).flatten()
                for pid, pval in zip(img_ids.numpy(), preds):
                    writer.writerow([name, pid.decode('utf-8'), pval])

    # 4. Ensemble Optimization
    print("\nOptimizing Ensembles...")
    df_preds = pd.read_csv(csv_path)
    
    # Aggregate Patch -> Image (Mean)
    img_preds = df_preds.groupby(['Model', 'ImageID'])['Prediction'].mean().reset_index()
    pivot_df = img_preds.pivot(index='ImageID', columns='Model', values='Prediction').reset_index()
    
    true_age_dict = dict(zip(labels_data['File'], labels_data['Age']))
    
    summary_df = optimize_ensembles(pivot_df, true_age_dict)
    
    print("\n=== Final Ensemble Results ===")
    print(summary_df)
    summary_df.to_csv(os.path.join(CONFIG["RESULTS_DIR"], CONFIG["SUMMARY_CSV"]), index=False)

if __name__ == "__main__":
    main()