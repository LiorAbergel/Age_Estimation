"""
Verification script: Load trained CNN CV models from Google Drive,
reproduce OOF (out-of-fold) metrics, and compare against results.md.

The Colab notebook saves fine-tuned checkpoints as:
    {CKPT_ROOT}/{model_name}/{model_name}_fold{fold}_ft.keras


Usage (Colab):
    Run all cells — GDrive is auto-mounted.

Usage (local):
    1. Download model folders from GDrive to 03_CNN_CrossVal/gdrive_models/
    2. Run: python 03_CNN_CrossVal/verify_cv_from_gdrive.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/HHD_AgeSplit'
    CKPT_ROOT = '/content/drive/MyDrive/HHD_AgeSplit/CV_STRAT_GROUP'
    RESULTS_DIR = '/content/cv_verification_results'
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(REPO_ROOT, 'data')
    CKPT_ROOT = os.path.join(SCRIPT_DIR, 'gdrive_models')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'verification_results')

CSV_PATH = os.path.join(DATA_DIR, 'NewAgeSplit.csv')

PATCH_SIZE = (400, 400)
STRIDE = 200
BATCH_SIZE = 64
STANDARD_SIZE = 800
THR = 0.0054  # empty-patch filter threshold

MODEL_NAMES = ['ResNet50', 'DenseNet121', 'InceptionV3', 'InceptionResNetV2', 'EfficientNetV2M']
N_FOLDS = 5

# Expected results from results.md (mean ± std across 5 folds)
# Keys match compute_evaluation_metrics output
EXPECTED_CV = {
    'ResNet50': {
        'MAE': (5.41, 0.78), 'RMSE': (8.17, 0.58), 'R2': (0.10, 0.06),
        'MAPE (%)': (25.68, 3.73),
        'Within +/-2 Years (%)': (23.72, 5.17), 'Within +/-5 Years (%)': (63.40, 11.26),
        'Within +/-10 Years (%)': (90.10, 4.42),
        'Max Error': (32.99, 7.14), 'Median Error': (3.72, 0.88),
    },
    'DenseNet121': {
        'MAE': (5.46, 1.06), 'RMSE': (8.16, 0.56), 'R2': (0.11, 0.06),
        'MAPE (%)': (26.25, 5.47),
        'Within +/-2 Years (%)': (21.49, 16.51), 'Within +/-5 Years (%)': (61.61, 16.41),
        'Within +/-10 Years (%)': (91.18, 3.84),
        'Max Error': (34.14, 6.66), 'Median Error': (3.94, 1.33),
    },
    'InceptionResNetV2': {
        'MAE': (5.69, 0.70), 'RMSE': (7.98, 0.68), 'R2': (0.17, 0.11),
        'MAPE (%)': (29.08, 3.16),
        'Within +/-2 Years (%)': (16.76, 3.99), 'Within +/-5 Years (%)': (58.32, 9.19),
        'Within +/-10 Years (%)': (89.32, 4.58),
        'Max Error': (32.62, 5.80), 'Median Error': (4.37, 0.59),
    },
    'InceptionV3': {
        'MAE': (6.03, 0.64), 'RMSE': (8.41, 0.64), 'R2': (0.05, 0.05),
        'MAPE (%)': (29.97, 3.17),
        'Within +/-2 Years (%)': (16.40, 4.30), 'Within +/-5 Years (%)': (52.83, 9.01),
        'Within +/-10 Years (%)': (90.11, 4.84),
        'Max Error': (32.70, 7.60), 'Median Error': (4.80, 0.67),
    },
    'EfficientNetV2M': {
        'MAE': (7.30, 0.28), 'RMSE': (9.03, 0.36), 'R2': (-0.07, 0.14),
        'MAPE (%)': (41.48, 5.73),
        'Within +/-2 Years (%)': (11.58, 4.78), 'Within +/-5 Years (%)': (29.47, 2.21),
        'Within +/-10 Years (%)': (84.32, 7.19),
        'Max Error': (31.44, 5.01), 'Median Error': (7.05, 0.68),
    },
}

# ---------------------------------------------------------------------------
# Data processing (matches both Colab and repo logic)
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


def read_image_and_resize(img_path):
    try:
        img_path_str = img_path.numpy().decode("utf-8")
        img = Image.open(img_path_str).convert('RGB')
        w, h = img.size
        new_h, new_w = calculate_resized_dimensions(h, w, PATCH_SIZE[0], STRIDE, STANDARD_SIZE)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return np.zeros((PATCH_SIZE[0], PATCH_SIZE[1], 3), dtype=np.float32)


def process_row_with_id(row, root_dir):
    root = tf.constant(root_dir, dtype=tf.string)
    subset = row['Set']
    fname = row['File']
    img_path = tf.strings.join([root, subset, fname], separator=os.sep)

    img = tf.py_function(func=read_image_and_resize, inp=[img_path], Tout=tf.float32)
    img.set_shape([None, None, 3])

    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
        strides=[1, STRIDE, STRIDE, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [-1, PATCH_SIZE[0], PATCH_SIZE[1], 3])

    # Filter empty patches
    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > THR
    patches = tf.boolean_mask(patches, mask)

    n_patches = tf.shape(patches)[0]
    labels = tf.fill([n_patches], row['Age'])
    ids = tf.fill([n_patches], row['File'])
    return patches, labels, ids


def create_dataset(df_subset, root_dir):
    ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
    ds = ds.map(lambda row: process_row_with_id(row, root_dir), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(lambda p, a, i: tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(p),
        tf.data.Dataset.from_tensor_slices(a),
        tf.data.Dataset.from_tensor_slices(i)
    )))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_evaluation_metrics(true_ages, pred_ages):
    mae = mean_absolute_error(true_ages, pred_ages)
    rmse = np.sqrt(mean_squared_error(true_ages, pred_ages))
    r2 = r2_score(true_ages, pred_ages)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_ages - pred_ages) / true_ages)) * 100
        if np.isnan(mape):
            mape = 0.0
    errors = np.abs(true_ages - pred_ages)
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape,
        "Within +/-2 Years (%)": np.mean(errors <= 2) * 100,
        "Within +/-5 Years (%)": np.mean(errors <= 5) * 100,
        "Within +/-10 Years (%)": np.mean(errors <= 10) * 100,
        "Max Error": np.max(errors),
        "Median Error": np.median(errors),
    }


def run_inference(model, dataset):
    """Returns dict {image_id: [patch_pred_1, patch_pred_2, ...]}."""
    preds_per_image = defaultdict(list)
    true_ages = {}
    for patches, ages, img_ids in dataset:
        p = model.predict(patches, verbose=0).ravel()
        for pred, age, pid in zip(p, ages.numpy(), img_ids.numpy()):
            pid_str = pid.decode('utf-8')
            preds_per_image[pid_str].append(pred)
            true_ages[pid_str] = float(age)
    return preds_per_image, true_ages


def find_checkpoint(model_name, fold, ckpt_roots):
    """Search for the fine-tuned checkpoint across multiple roots and naming conventions."""
    suffixes = ['_ft.keras', '_finetune.keras']
    for root in ckpt_roots:
        if root is None:
            continue
        for suffix in suffixes:
            path = os.path.join(root, model_name, f"{model_name}_fold{fold}{suffix}")
            if os.path.isfile(path):
                return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU count: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Data dir:  {DATA_DIR}")
    print(f"CKPT root: {CKPT_ROOT}")

    # --- Load CSV ---
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return
    df_full = pd.read_csv(CSV_PATH)
    true_age_dict = dict(zip(df_full['File'], df_full['Age']))

    # --- Reproduce CV splits ---
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_full.index, df_full["AgeGroup"], df_full["WriterNumber"]))

    ckpt_roots = [CKPT_ROOT]

    # =====================================================================
    # 1) OOF Evaluation — reproduce per-fold val metrics (matches results.md)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: OOF (Out-of-Fold) Evaluation — per-fold val metrics")
    print("=" * 70)

    all_model_fold_metrics = {}  # {model_name: [fold_metrics_dict, ...]}
    all_predictions = []  # accumulate OOF predictions for single CSV

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name} ---")
        fold_metrics_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            ckpt_path = find_checkpoint(model_name, fold_idx, ckpt_roots)
            if ckpt_path is None:
                print(f"  Fold {fold_idx}: checkpoint NOT FOUND — skipping")
                continue

            print(f"  Fold {fold_idx}: loading {os.path.basename(ckpt_path)} from {os.path.dirname(ckpt_path)}")
            model = tf.keras.models.load_model(ckpt_path, compile=False)

            # Create val dataset for this fold
            val_df = df_full.iloc[val_idx].reset_index(drop=True)
            val_ds = create_dataset(val_df, DATA_DIR)

            # Run inference
            preds_per_image, fold_true_ages = run_inference(model, val_ds)

            # Aggregate to image level
            y_pred, y_true = [], []
            for img_id, plist in preds_per_image.items():
                if img_id in true_age_dict:
                    mean_pred = np.mean(plist)
                    y_pred.append(mean_pred)
                    y_true.append(true_age_dict[img_id])
                    all_predictions.append({
                        'Model': model_name,
                        'Fold': fold_idx,
                        'ImageID': img_id,
                        'Prediction': mean_pred,
                        'TrueAge': true_age_dict[img_id],
                    })

            metrics = compute_evaluation_metrics(np.array(y_true), np.array(y_pred))
            fold_metrics_list.append(metrics)
            print(f"    MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")

            del model, val_ds
            gc.collect()
            tf.keras.backend.clear_session()

        all_model_fold_metrics[model_name] = fold_metrics_list

    # Print OOF summary and compare with results.md
    print("\n" + "=" * 70)
    print("OOF SUMMARY (mean +/- std across folds)  vs  results.md")
    print("=" * 70)

    METRIC_KEYS = [
        'MAE', 'RMSE', 'R2', 'MAPE (%)',
        'Within +/-2 Years (%)', 'Within +/-5 Years (%)', 'Within +/-10 Years (%)',
        'Max Error', 'Median Error',
    ]
    METRIC_SHORT = [
        'MAE', 'RMSE', 'R2', 'MAPE',
        '+/-2yr', '+/-5yr', '+/-10yr',
        'MaxErr', 'MedErr',
    ]

    for model_name in MODEL_NAMES:
        fms = all_model_fold_metrics.get(model_name, [])
        if not fms:
            print(f"  {model_name}: NO FOLDS EVALUATED")
            continue

        expected = EXPECTED_CV.get(model_name, {})
        print(f"\n  {model_name}:")
        print(f"    {'Metric':<12} {'Actual':>18}  {'Expected':>18}  {'Status'}")
        print(f"    {'-'*12} {'-'*18}  {'-'*18}  {'-'*8}")

        for key, short in zip(METRIC_KEYS, METRIC_SHORT):
            arr = np.array([m[key] for m in fms])
            mean_val, std_val = arr.mean(), arr.std()

            exp = expected.get(key)
            if exp is not None:
                exp_mean, exp_std = exp
                status = 'MATCH' if abs(mean_val - exp_mean) < 0.15 else 'MISMATCH'
                print(f"    {short:<12} {mean_val:7.2f} +/- {std_val:5.2f}  {exp_mean:7.2f} +/- {exp_std:5.2f}  {status}")
            else:
                print(f"    {short:<12} {mean_val:7.2f} +/- {std_val:5.2f}  {'N/A':>18}")

    # --- Save OOF predictions (single CSV) ---
    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(RESULTS_DIR, 'oof_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nOOF predictions saved to {predictions_path} ({len(predictions_df)} rows)")

    # --- Save per-fold metrics summary ---
    summary_rows = []
    for model_name in MODEL_NAMES:
        fms = all_model_fold_metrics.get(model_name, [])
        if fms:
            for i, fm in enumerate(fms):
                row = {'Model': model_name, 'Fold': i + 1}
                row.update(fm)
                summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, 'cv_verification_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"OOF per-fold metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
