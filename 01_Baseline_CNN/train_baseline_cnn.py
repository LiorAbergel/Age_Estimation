"""
Experiment 01: Baseline CNN for Age Estimation

Overview:
This script establishes a baseline performance metric for age estimation from 
handwriting using a custom, lightweight Convolutional Neural Network (CNN).

Architecture:
- Structure: 3 Convolutional blocks (32, 64, 128 filters) followed by Max Pooling.
- Head: Flatten -> Dense (128) -> Dropout -> Output (1).
- Input: 128x128 RGB images.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "IMG_SIZE": (128, 128),
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-3,
    "DROPOUT_RATE": 0.5,
    # Paths updated for relative structure
    "DATA_DIR": "./data",  
    "CSV_PATH": "./data/NewAgeSplit.csv",
    "RESULTS_DIR": "./results",
    "MODEL_FILENAME": "experiment_01_model.keras",
    "PREDICTIONS_FILENAME": "experiment_01_predictions.csv"
}

def load_images_and_labels(data_dir, dataset_type, labels_df):
    """
    Loads images, corresponding age labels, and filenames.
    """
    subset_df = labels_df[labels_df['Set'] == dataset_type]
    images = []
    labels = []
    filenames = []

    print(f"Loading {dataset_type} data...")
    
    target_dir = os.path.join(data_dir, dataset_type)
    if not os.path.exists(target_dir):
        # Graceful fallback or error if data is missing
        print(f"Warning: Directory not found: {target_dir}")
        return np.array([]), np.array([]), np.array([])

    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f"Processing {dataset_type}"):
        img_path = os.path.join(target_dir, row['File'])
        
        if os.path.exists(img_path):
            try:
                # Load and preprocess
                img = tf.keras.utils.load_img(img_path, target_size=CONFIG["IMG_SIZE"])
                img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize [0,1]
                images.append(img)
                labels.append(row['Age'])
                filenames.append(row['File'])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(filenames)

def build_model(input_shape):
    """
    Defines a Sequential CNN with increasing filter depth to capture 
    hierarchical features of the handwriting.
    """
    model = Sequential([
        # Block 1: Spatial features
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), 
        MaxPooling2D((2, 2)), 
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Regression Head
        Flatten(),
        Dense(128, activation='relu'), 
        Dropout(CONFIG["DROPOUT_RATE"]), 
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # Create results directory
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    model_save_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["MODEL_FILENAME"])

    print(f"TensorFlow Version: {tf.__version__}")

    # --- Load Metadata ---
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}. Please ensure data is set up correctly.")
        return

    labels_data = pd.read_csv(CONFIG["CSV_PATH"])
    
    # Load Datasets
    X_train, y_train_raw, _ = load_images_and_labels(CONFIG["DATA_DIR"], 'train', labels_data)
    X_val, y_val_raw, _ = load_images_and_labels(CONFIG["DATA_DIR"], 'val', labels_data)
    X_test, y_test_raw, test_files = load_images_and_labels(CONFIG["DATA_DIR"], 'test', labels_data)
    
    if len(X_train) == 0:
        print("No training data found. Exiting.")
        return

    print(f"\nDataset Shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    # --- Normalization ---
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    print(f"Labels scaled to range: {y_train.min()} - {y_train.max()}")

    # --- Build Model ---
    model = build_model((CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3))
    model.summary()

    # --- Training Setup ---
    # Augmentation for training only
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=CONFIG["BATCH_SIZE"])
    val_generator = val_datagen.flow(X_val, y_val, batch_size=CONFIG["BATCH_SIZE"])
    test_generator = test_datagen.flow(X_test, y_test, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    checkpoint = ModelCheckpoint(
        model_save_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min',
        verbose=1
    )

    # --- Run Training ---
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=CONFIG["EPOCHS"],
        callbacks=[checkpoint]
    )

    # --- Evaluation ---
    print("\nEvaluating best model...")
    best_model = load_model(model_save_path)

    predictions_scaled = best_model.predict(test_generator)
    predictions_years = scaler.inverse_transform(predictions_scaled).flatten()

    # Metrics
    mae = np.mean(np.abs(y_test_raw - predictions_years))
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_years))
    r2 = r2_score(y_test_raw, predictions_years)
    
    # Accuracy Thresholds
    errors = np.abs(y_test_raw - predictions_years)
    within_5 = np.mean(errors <= 5) * 100

    print(f"\n--- Final Results (Baseline CNN) ---")
    print(f"MAE:  {mae:.2f} years")
    print(f"RMSE: {rmse:.2f} years")
    print(f"R²:   {r2:.4f}")
    print(f"Accuracy (±5 years):  {within_5:.2f}%")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Error Distribution
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Absolute Error (Years)')
    plt.ylabel('Frequency')

    # Cumulative Error
    plt.subplot(1, 2, 2)
    errors_sorted = np.sort(errors)
    cumulative_percentage = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted) * 100
    plt.plot(errors_sorted, cumulative_percentage, marker='.', linestyle='none')
    plt.title('Cumulative Error Curve')
    plt.xlabel('Error Threshold (Years)')
    plt.ylabel('Percentage of Samples (%)')
    plt.grid(True)
    plt.axvline(x=5, color='r', linestyle='--', label='5 Year Threshold')
    plt.legend()

    plot_path = os.path.join(CONFIG["RESULTS_DIR"], 'error_analysis.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Error analysis plot saved to {plot_path}")
    plt.close()

    # --- Save Predictions ---
    results_df = pd.DataFrame({
        'Image': test_files,
        'True Age': y_test_raw,
        'Predicted Age': np.round(predictions_years, 2)
    })

    pred_save_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["PREDICTIONS_FILENAME"])
    results_df.to_csv(pred_save_path, index=False)
    print(f"Predictions saved to: {pred_save_path}")

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()