"""
Standalone Colab script: Fine-tune ResNet50 fold 3 from the init checkpoint.

Reproduces exactly the fine-tuning phase of run_cv() from the training notebook.
Assumes the init checkpoint already exists on Google Drive.
"""

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

from google.colab import drive
drive.mount('/content/drive')

# ---------------------------------------------------------------------------
# Configuration — must match the original training notebook exactly
# ---------------------------------------------------------------------------
DATA_ROOT = '/content/drive/MyDrive/HHD_AgeSplit'
CKPT_ROOT = '/content/drive/MyDrive/HHD_AgeSplit/CV_STRAT_GROUP'
CSV_PATH  = os.path.join(DATA_ROOT, 'NewAgeSplit.csv')

MODEL_NAME = 'ResNet50'
FOLD = 3  # The missing fold

PATCH_SIZE = (400, 400)
STRIDE = 200
BATCH_SIZE = 64
EPOCHS_FT = 10
THR = 0.0054

# ---------------------------------------------------------------------------
# Data processing — identical to training notebook
# ---------------------------------------------------------------------------

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
        img = Image.open(img_path.numpy().decode("utf-8")).convert('RGB')
        new_h, new_w = calculate_resized_dimensions(img.size[1], img.size[0])
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except:
        return np.zeros((800, 800, 3), dtype=np.float32)


def process_image(row, root_dir, patch_size, step_size):
    root = tf.constant(root_dir, dtype=tf.string)
    subset = row['Set']
    fname = row['File']
    img_path = tf.strings.join([root, subset, fname], separator=os.path.sep)

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

    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > THR
    patches = tf.boolean_mask(patches, mask)
    labels = tf.boolean_mask(labels, mask)

    return patches, labels


# --- Augmentation (identical to training notebook) ---
rotation_layer = tf.keras.layers.RandomRotation(factor=0.04167)

def advanced_augmentation(image, label):
    image = rotation_layer(image, training=True)

    orig_shape = tf.shape(image)[:2]
    zoom_factor = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, orig_shape[0], orig_shape[1])

    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0., 1.)

    return image, label


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
        ds = ds.map(lambda img, lbl: advanced_augmentation(img, lbl), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Model ---
def build_sota_model(base_model_fn, input_shape=(400, 400, 3), dropout_rate=0.5):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    return Model(inputs, outputs)


# --- Evaluation helpers ---
def process_row_with_id(row):
    patches, labels = process_image(row, DATA_ROOT, PATCH_SIZE, STRIDE)
    image_id = tf.fill([tf.shape(patches)[0]], row['File'])
    return patches, labels, image_id


def patch_data_tf_dataset_with_ids_from_df(df_subset, data_dir, patch_size, step_size, batch_size, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
    ds = ds.map(process_row_with_id, num_parallel_calls=tf.data.AUTOTUNE).flat_map(
        lambda patches, labels, image_ids: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(patches),
            tf.data.Dataset.from_tensor_slices(labels),
            tf.data.Dataset.from_tensor_slices(image_ids),
        ))
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def group_predictions_by_image_id(predictions_with_ids, labels_df):
    grouped_predictions = defaultdict(list)
    grouped_labels = defaultdict(list)

    for pred, image_id in predictions_with_ids:
        image_id_str = image_id.decode('utf-8') if isinstance(image_id, bytes) else image_id
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
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_images - predicted_images) / true_images)) * 100
        if np.isnan(mape): mape = 0.0
    errors = np.abs(true_images - predicted_images)
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "Acc_2yr": np.mean(errors <= 2) * 100,
        "Acc_5yr": np.mean(errors <= 5) * 100,
    }


# ---------------------------------------------------------------------------
# Main: Fine-tune ResNet50 Fold 3
# ---------------------------------------------------------------------------
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

df_full = pd.read_csv(CSV_PATH)

# Reproduce the exact same splits as the training notebook
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

train_idx, val_idx = splits[FOLD - 1]  # 0-indexed
train_df = df_full.iloc[train_idx].reset_index(drop=True)
val_df = df_full.iloc[val_idx].reset_index(drop=True)
print(f"Fold {FOLD}: {len(train_df)} train, {len(val_df)} val images")

# --- Check init checkpoint exists ---
ckpt_dir = os.path.join(CKPT_ROOT, MODEL_NAME)
ckpt_init = os.path.join(ckpt_dir, f"{MODEL_NAME}_fold{FOLD}_init.keras")
ckpt_ft = os.path.join(ckpt_dir, f"{MODEL_NAME}_fold{FOLD}_ft.keras")

if not os.path.isfile(ckpt_init):
    raise FileNotFoundError(f"Init checkpoint not found: {ckpt_init}")
print(f"Loading init checkpoint: {ckpt_init}")

# --- Build model and load init weights (same as Colab run_cv lines 744-752) ---
model = build_sota_model(ResNet50, input_shape=(*PATCH_SIZE, 3), dropout_rate=0.5)
model.load_weights(ckpt_init)
model.layers[1].trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
model.summary()

# --- Create datasets ---
print("\nCreating datasets...")
train_ds = patch_data_tf_dataset_from_df(train_df, DATA_ROOT, PATCH_SIZE, STRIDE, BATCH_SIZE, augment=True)
val_ds = patch_data_tf_dataset_from_df(val_df, DATA_ROOT, PATCH_SIZE, STRIDE, BATCH_SIZE, augment=False)

# --- Fine-tune (same callbacks as Colab run_cv lines 767-776) ---
print(f"\nFine-tuning {MODEL_NAME} fold {FOLD} for {EPOCHS_FT} epochs...")
callbacks = [
    ModelCheckpoint(ckpt_ft, monitor="val_mae", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_mae", factor=0.2, patience=4, verbose=1),
    EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    callbacks=callbacks,
    verbose=2
)

# --- Evaluate on fold 3 val set (image-level, same as Colab run_cv lines 782-799) ---
print("\nEvaluating on fold 3 validation set...")
del train_ds, val_ds
gc.collect()

val_ids_ds = patch_data_tf_dataset_with_ids_from_df(val_df, DATA_ROOT, PATCH_SIZE, STRIDE, BATCH_SIZE, augment=False)
preds = []
for patches, _, img_ids in val_ids_ds:
    p = model.predict(patches, verbose=0).ravel()
    preds.extend(zip(p, img_ids.numpy()))

y_pred_img, y_true_img = group_predictions_by_image_id(preds, val_df)
metrics = compute_evaluation_metrics(y_true_img, y_pred_img)

print(f"\n{'='*50}")
print(f"ResNet50 Fold {FOLD} Fine-Tuned Results:")
for k, v in metrics.items():
    print(f"  {k}: {v:.2f}")
print(f"{'='*50}")
print(f"\nCheckpoint saved to: {ckpt_ft}")
print(f"File exists: {os.path.isfile(ckpt_ft)}")
