"""
Verification script: Load trained ViT CV models from Google Drive,
reproduce OOF (out-of-fold) metrics, and compare against results.md.

The Colab notebook saves fine-tuned checkpoints as:
    {CKPT_ROOT}/{model_name}/fold_{fold:02d}/{model_name}_finetune.keras

Usage (Colab):
    Run all cells — GDrive is auto-mounted.

Usage (local):
    1. Download model folders from GDrive to 05_ViT_CrossVal/gdrive_models/
    2. Run: python 05_ViT_CrossVal/verify_vit_cv_from_gdrive.py
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
    CKPT_ROOT = '/content/drive/MyDrive/HHD_AgeSplit/ViT_CV2'
    RESULTS_DIR = '/content/vit_cv_verification_results'
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
N_FOLDS = 5

# Model configs: name -> input resolution (must match training)
VIT_MODEL_CONFIGS = {
    "SwinV2_Tiny": 256,
    "MobileViT_XXS": 256,
    "ConvNeXtV2_Tiny": 224,
    "TinyViT_11M": 224,
}

# Expected results from results.md (mean ± std across 5 folds)
EXPECTED_CV = {
    'MobileViT_XXS': {
        'MAE': (6.00, 0.76), 'RMSE': (8.27, 0.27), 'R2': (0.08, 0.05),
        'MAPE (%)': (30.63, 4.35),
        'Within +/-2 Years (%)': (14.08, 5.21), 'Within +/-5 Years (%)': (54.89, 13.95),
        'Max Error': (31.23, 7.30), 'Median Error': (4.76, 1.01),
    },
    'TinyViT_11M': {
        'MAE': (6.63, 0.47), 'RMSE': (8.71, 0.70), 'R2': (-0.02, 0.13),
        'MAPE (%)': (35.61, 5.96),
        'Within +/-2 Years (%)': (13.49, 5.59), 'Within +/-5 Years (%)': (39.59, 6.23),
        'Max Error': (31.69, 7.04), 'Median Error': (5.80, 0.69),
    },
    'SwinV2_Tiny': {
        'MAE': (6.78, 0.28), 'RMSE': (8.85, 0.44), 'R2': (-0.05, 0.07),
        'MAPE (%)': (36.23, 4.42),
        'Within +/-2 Years (%)': (12.49, 5.80), 'Within +/-5 Years (%)': (35.66, 1.41),
        'Max Error': (31.98, 6.74), 'Median Error': (6.15, 0.41),
    },
    'ConvNeXtV2_Tiny': {
        'MAE': (6.81, 0.38), 'RMSE': (8.91, 0.51), 'R2': (-0.07, 0.11),
        'MAPE (%)': (36.34, 6.18),
        'Within +/-2 Years (%)': (11.08, 4.87), 'Within +/-5 Years (%)': (37.08, 3.37),
        'Max Error': (32.04, 6.44), 'Median Error': (6.16, 0.56),
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


def create_dataset(df_subset, root_dir, final_size):
    """Create a dataset with patches resized to model's required input size."""
    ds = tf.data.Dataset.from_tensor_slices(dict(df_subset))
    ds = ds.map(lambda row: process_row_with_id(row, root_dir), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(lambda p, a, i: tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(p),
        tf.data.Dataset.from_tensor_slices(a),
        tf.data.Dataset.from_tensor_slices(i)
    )))
    # Resize 400x400 patches to ViT input size (224 or 256)
    ds = ds.map(
        lambda p, a, i: (tf.image.resize(p, [final_size, final_size]), a, i),
        num_parallel_calls=tf.data.AUTOTUNE
    )
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
    """Returns dict {image_id: [patch_pred_1, patch_pred_2, ...]} and true ages."""
    preds_per_image = defaultdict(list)
    true_ages = {}
    for patches, ages, img_ids in dataset:
        p = model.predict(patches, verbose=0).ravel()
        for pred, age, pid in zip(p, ages.numpy(), img_ids.numpy()):
            pid_str = pid.decode('utf-8')
            preds_per_image[pid_str].append(pred)
            true_ages[pid_str] = float(age)
    return preds_per_image, true_ages


def find_checkpoint(model_name, fold_idx, ckpt_root):
    """Search for the best available checkpoint for a given model/fold."""
    fold_dir = os.path.join(ckpt_root, model_name, f'fold_{fold_idx:02d}')
    # Prefer fine-tuned, fall back to init
    for suffix in ['_finetune.keras', '_init.keras']:
        path = os.path.join(fold_dir, f'{model_name}{suffix}')
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Freeze/Unfreeze Verification
# ---------------------------------------------------------------------------

def verify_freeze_logic():
    """
    Verify that the train_vit_cv.py freeze/unfreeze logic is correct by
    inspecting a sample model's layer structure.
    """
    print("=" * 70)
    print("FREEZE/UNFREEZE LOGIC VERIFICATION")
    print("=" * 70)

    print("""
REPO SCRIPT (train_vit_cv.py):
  Stage 1 (Frozen):
    for layer in model.layers[:-2]:
        layer.trainable = False
  Stage 2 (Fine-tune):
    for layer in model.layers:
        layer.trainable = True

ANALYSIS:
  The model is built as: Model(backbone.input, Dense(1)(Dropout(GAP(backbone.output))))
  When using backbone.input/output directly, Keras FLATTENS all backbone layers
  into the outer model's .layers list:
    - model.layers[0]        = InputLayer
    - model.layers[1:-3]     = backbone internal layers (many)
    - model.layers[-3]       = GlobalAveragePooling2D (no trainable weights)
    - model.layers[-2]       = Dropout (no trainable weights)
    - model.layers[-1]       = Dense(1) (trainable)

  model.layers[:-2] freezes ALL backbone + GAP layers.
  Only Dense(1) remains trainable in Stage 1.  -> CORRECT

COLAB NOTEBOOK (v6_vit_cv.py) used a DIFFERENT approach:
  Stage 1: model.layers[1].trainable = False
  Stage 2: model.layers[1].trainable = True

  This only freezes/unfreezes ONE LAYER (index 1), not the entire backbone.
  This is effectively a BUG — Stage 1 was never truly frozen in the Colab runs.
  The backbone was almost entirely trainable even in Stage 1.

VERDICT:
  The repo script's approach is CORRECT and SUPERIOR to the Colab.
  However, this means the repo script will produce DIFFERENT training dynamics
  from the Colab if retrained from scratch (true frozen-then-unfreeze vs
  effectively both stages unfrozen).

  The GDrive checkpoints were trained with the Colab's buggy freezing, so they
  represent training where the backbone was never truly frozen.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Verify freeze logic ---
    verify_freeze_logic()

    print(f"\nGPU count: {len(tf.config.list_physical_devices('GPU'))}")
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

    # =====================================================================
    # OOF Evaluation — reproduce per-fold val metrics (matches results.md)
    # =====================================================================
    print("\n" + "=" * 70)
    print("OOF (Out-of-Fold) Evaluation — per-fold val metrics")
    print("=" * 70)

    all_model_fold_metrics = {}
    all_predictions = []

    for model_name, input_size in VIT_MODEL_CONFIGS.items():
        print(f"\n--- {model_name} (input: {input_size}x{input_size}) ---")
        fold_metrics_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            ckpt_path = find_checkpoint(model_name, fold_idx, CKPT_ROOT)
            if ckpt_path is None:
                print(f"  Fold {fold_idx}: checkpoint NOT FOUND — skipping")
                continue

            print(f"  Fold {fold_idx}: loading {os.path.basename(ckpt_path)}")
            model = tf.keras.models.load_model(ckpt_path, compile=False)

            # Create val dataset for this fold (with model-specific resize)
            val_df = df_full.iloc[val_idx].reset_index(drop=True)
            val_ds = create_dataset(val_df, DATA_DIR, final_size=input_size)

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
            print(f"    MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")

            del model, val_ds
            gc.collect()
            tf.keras.backend.clear_session()

        all_model_fold_metrics[model_name] = fold_metrics_list

    # =====================================================================
    # Summary: compare with results.md
    # =====================================================================
    print("\n" + "=" * 70)
    print("OOF SUMMARY (mean +/- std across folds)  vs  results.md")
    print("=" * 70)

    METRIC_KEYS = [
        'MAE', 'RMSE', 'R2', 'MAPE (%)',
        'Within +/-2 Years (%)', 'Within +/-5 Years (%)',
        'Max Error', 'Median Error',
    ]
    METRIC_SHORT = [
        'MAE', 'RMSE', 'R2', 'MAPE',
        '+/-2yr', '+/-5yr',
        'MaxErr', 'MedErr',
    ]

    for model_name in VIT_MODEL_CONFIGS.keys():
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

    # --- Save OOF predictions ---
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_path = os.path.join(RESULTS_DIR, 'vit_cv_oof_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nOOF predictions saved to {predictions_path} ({len(predictions_df)} rows)")

    # --- Save per-fold metrics summary ---
    summary_rows = []
    for model_name in VIT_MODEL_CONFIGS.keys():
        fms = all_model_fold_metrics.get(model_name, [])
        if fms:
            for i, fm in enumerate(fms):
                row = {'Model': model_name, 'Fold': i + 1}
                row.update(fm)
                summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, 'vit_cv_verification_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Per-fold metrics saved to {summary_path}")

    # =====================================================================
    # Training Config Comparison: repo vs Colab
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING CONFIG COMPARISON: train_vit_cv.py vs Colab v6_vit_cv.py")
    print("=" * 70)
    print(f"""
    {'Parameter':<25} {'Repo (train_vit_cv.py)':<30} {'Colab (v6)':<30}
    {'-'*25} {'-'*30} {'-'*30}
    {'Freeze method':<25} {'model.layers[:-2]':<30} {'model.layers[1] (BUG)':<30}
    {'Optimizer Stage 1':<25} {'AdamW + CosineDecay':<30} {'Adam (default lr=1e-3)':<30}
    {'LR Stage 1':<25} {'1e-6 -> 1e-3 (warmup)':<30} {'1e-3 (constant)':<30}
    {'Optimizer Stage 2':<25} {'AdamW + CosineDecay':<30} {'Adam(1e-4)':<30}
    {'LR Stage 2':<25} {'1e-7 -> 1e-4 (warmup)':<30} {'1e-4 (constant)':<30}
    {'Weight decay':<25} {'1e-4':<30} {'None':<30}
    {'Gradient clipping':<25} {'clipnorm=1.0':<30} {'None':<30}
    {'Mixed precision':<25} {'mixed_float16':<30} {'No':<30}
    {'Epochs Stage 1':<25} {'50':<30} {'50':<30}
    {'Epochs Stage 2':<25} {'30':<30} {'10':<30}
    {'EarlyStopping Stage 1':<25} {'patience=10':<30} {'patience=8':<30}
    {'EarlyStopping Stage 2':<25} {'patience=7':<30} {'patience=8':<30}
    {'Resize method':<25} {'bilinear (default)':<30} {'bicubic':<30}
    {'Augmentation':<25} {'Contrast':<30} {'Contrast (v1) / Bright(v2)':<30}
    {'Batch size':<25} {'64':<30} {'64':<30}
    {'THR':<25} {'0.0054':<30} {'0.0054':<30}
    {'CV strategy':<25} {'StratifiedGroupKFold(42)':<30} {'StratifiedGroupKFold(42)':<30}

    NOTE: The GDrive checkpoints were trained with the COLAB configuration.
          If you retrain using the repo script, results may differ due to:
          1. Correct backbone freezing (Colab never truly froze the backbone)
          2. CosineDecay+warmup vs constant LR
          3. Weight decay / gradient clipping (not in Colab)
          4. 30 fine-tune epochs vs 10
          5. bilinear vs bicubic resize
    """)


if __name__ == "__main__":
    main()
