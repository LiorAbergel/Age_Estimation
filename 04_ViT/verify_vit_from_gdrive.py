"""
Verification script: Load retrained ViT models from Google Drive (ViT2),
run inference on the test set, compute individual + ensemble metrics,
and compare against the original results.md.

Usage (Colab):
    1. Mount Google Drive
    2. Run this script

Usage (local):
    1. Download ViT2 folder from GDrive
    2. Set MODELS_DIR to that folder path
    3. Run: python 04_ViT/verify_vit_from_gdrive.py
"""

import os
import gc
import numpy as np
import pandas as pd

# --- CRITICAL SETUP ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Need these imports so load_model can deserialize custom layers
try:
    import keras_cv_attention_models.swin_transformer_v2 as swin_v2
    import keras_cv_attention_models.mobilevit as mobilevit
    import keras_cv_attention_models.convnext as convnext
    import keras_cv_attention_models.tinyvit as tiny_vit
except ImportError:
    raise ImportError("Please run: pip install keras-cv-attention-models tf-keras")

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
    DATA_ROOT = '/content/drive/MyDrive/HHD_AgeSplit'
    MODELS_DIR = os.path.join(DATA_ROOT, 'ViT2')
    RESULTS_DIR = os.path.join(DATA_ROOT, 'ViT2', 'verification_results')
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_ROOT = os.path.join(REPO_ROOT, 'data')
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'gdrive_models', 'ViT2')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'verification_results')

CSV_PATH = os.path.join(DATA_ROOT, 'NewAgeSplit.csv')

PATCH_SIZE = (400, 400)
STRIDE = 200
BATCH_SIZE = 32  # small for inference — doesn't affect predictions
STANDARD_SIZE = 800

# Model name -> (checkpoint subpath, input resolution)
MODEL_REGISTRY = {
    "SwinV2_Tiny":     256,
    "MobileViT_XXS":   256,
    "ConvNeXtV2_Tiny": 224,
    "TinyViT_11M":     224,
}

# Expected results from results.md (original training) for comparison
EXPECTED = {
    "SwinV2_Tiny":     {"MAE": 6.98, "RMSE": 8.04, "R2": -0.62, "MAPE": 43.58, "Within_2yr": 1.72,  "Within_5yr": 18.10, "Within_10yr": 96.55, "Max_Error": 29.58, "Median_Error": 6.42},
    "MobileViT_XXS":   {"MAE": 4.42, "RMSE": 6.54, "R2": -0.07, "MAPE": 25.90, "Within_2yr": 16.38, "Within_5yr": 74.14, "Within_10yr": 96.55, "Max_Error": 31.44, "Median_Error": 3.38},
    "ConvNeXtV2_Tiny": {"MAE": 6.30, "RMSE": 7.58, "R2": -0.44, "MAPE": 38.78, "Within_2yr": 1.72,  "Within_5yr": 41.38, "Within_10yr": 96.55, "Max_Error": 30.36, "Median_Error": 5.64},
    "TinyViT_11M":     {"MAE": 6.70, "RMSE": 7.79, "R2": -0.52, "MAPE": 41.92, "Within_2yr": 1.72,  "Within_5yr": 35.34, "Within_10yr": 96.55, "Max_Error": 28.70, "Median_Error": 5.87},
}
EXPECTED_ENSEMBLE = {"MAE": 6.09, "RMSE": 7.42, "R2": -0.38, "MAPE": 37.51, "Within_2yr": 1.72, "Within_5yr": 41.38, "Within_10yr": 96.55, "Max_Error": 30.02, "Median_Error": 5.33}

METRIC_KEYS = ["MAE", "RMSE", "R2", "MAPE", "Within_2yr", "Within_5yr", "Within_10yr", "Max_Error", "Median_Error"]

# ---------------------------------------------------------------------------
# Data processing (mirrors train_vit.py)
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


def process_image(row, data_dir):
    root = tf.constant(data_dir, dtype=tf.string)
    subset = row['Set']
    fname = row['File']
    img_path = tf.strings.join([root, subset, fname], separator='/')

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
    labels = tf.fill([tf.shape(patches)[0]], row['Age'])
    ids = tf.fill([tf.shape(patches)[0]], row['File'])
    return patches, labels, ids


def create_test_dataset(data_dir, labels_df, final_size):
    """Creates a test dataset with patches resized to the model's input size."""
    subset_df = labels_df[labels_df['Set'] == 'test'].reset_index(drop=True)

    ds = tf.data.Dataset.from_tensor_slices(dict(subset_df))
    ds = ds.map(lambda row: process_image(row, data_dir), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(lambda p, l, i: tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(p),
        tf.data.Dataset.from_tensor_slices(l),
        tf.data.Dataset.from_tensor_slices(i)
    )))
    # Resize to model-specific input size (bilinear, matching train_vit.py inference)
    ds = ds.map(
        lambda p, l, i: (tf.image.resize(p, [final_size, final_size]), l, i),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_evaluation_metrics(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(errors / y_true) * 100
        if np.isnan(mape):
            mape = 0.0

    pct = lambda thr: 100 * np.mean(errors <= thr)

    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        "Within_2yr": pct(2), "Within_5yr": pct(5), "Within_10yr": pct(10),
        "Max_Error": np.max(errors), "Median_Error": np.median(errors),
    }


def run_inference(model, dataset):
    """Returns list of (prediction, image_id_str) tuples."""
    results = []
    for batch in tqdm(dataset, desc="  Inference"):
        patches, _, image_ids = batch
        preds = model.predict(patches, verbose=0).flatten()
        for p, img_id in zip(preds, image_ids.numpy()):
            results.append((float(p), img_id.decode('utf-8') if isinstance(img_id, bytes) else str(img_id)))
    return results


def aggregate_image_level(predictions_with_ids, true_age_dict):
    """Average patch predictions per image, return (y_true, y_pred) arrays."""
    grouped = defaultdict(list)
    for pred, img_id in predictions_with_ids:
        grouped[img_id].append(pred)

    common = sorted(set(grouped) & set(true_age_dict))
    y_pred = np.array([np.mean(grouped[k]) for k in common])
    y_true = np.array([true_age_dict[k] for k in common])
    return y_true, y_pred


def print_metrics(metrics, label=""):
    if label:
        print(f"\n  {label}")
    print(f"    MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | R²: {metrics['R2']:.3f} | MAPE: {metrics['MAPE']:.2f}%")
    print(f"    ±2 yrs: {metrics['Within_2yr']:.2f}% | ±5 yrs: {metrics['Within_5yr']:.2f}% | ±10 yrs: {metrics['Within_10yr']:.2f}%")
    print(f"    Max Error: {metrics['Max_Error']:.2f} | Median Error: {metrics['Median_Error']:.2f}")


def print_comparison_table(model_name, new_metrics, old_metrics):
    """Print side-by-side comparison of new vs old (results.md) metrics."""
    print(f"\n  {'Metric':<18s} {'New':>10s} {'Old (results.md)':>18s} {'Delta':>10s}")
    print(f"  {'-'*58}")
    for key in METRIC_KEYS:
        new_val = new_metrics.get(key, float('nan'))
        old_val = old_metrics.get(key, float('nan'))
        delta = new_val - old_val
        # For MAE, RMSE, MAPE, Max_Error, Median_Error: lower is better (negative delta = improvement)
        # For R2, Within_*: higher is better (positive delta = improvement)
        better_higher = key in ("R2", "Within_2yr", "Within_5yr", "Within_10yr")
        improved = (delta > 0) if better_higher else (delta < 0)
        arrow = "▲" if improved else ("▼" if delta != 0 else "=")
        print(f"  {key:<18s} {new_val:>10.2f} {old_val:>18.2f} {delta:>+9.2f} {arrow}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"GPU count: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Data dir:   {DATA_ROOT}")

    # --- Check prerequisites ---
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return
    labels_data = pd.read_csv(CSV_PATH)
    true_age_dict = labels_data.groupby('File')['Age'].mean().to_dict()

    # --- Discover and load models ---
    print("\n=== Loading Models ===")
    loaded_models = {}
    for name, input_size in MODEL_REGISTRY.items():
        ckpt_path = os.path.join(MODELS_DIR, name, f"{name}_finetune.keras")
        if not os.path.isfile(ckpt_path):
            print(f"  WARNING: {name} not found at {ckpt_path} — skipping")
            continue
        print(f"  Loading {name}...")
        loaded_models[name] = tf.keras.models.load_model(ckpt_path)

    if not loaded_models:
        print("ERROR: No models found. Check MODELS_DIR path.")
        return

    print(f"\nLoaded {len(loaded_models)} / {len(MODEL_REGISTRY)} models: {list(loaded_models.keys())}")

    # --- Individual model evaluation ---
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL EVALUATION (Test Set)")
    print("=" * 70)

    all_results = {}
    image_level_preds = {}
    y_true_ref = None

    for name, model in loaded_models.items():
        input_size = MODEL_REGISTRY[name]
        print(f"\n--- {name} (input: {input_size}x{input_size}) ---")

        test_ds = create_test_dataset(DATA_ROOT, labels_data, input_size)
        preds = run_inference(model, test_ds)
        y_true, y_pred = aggregate_image_level(preds, true_age_dict)

        metrics = compute_evaluation_metrics(y_true, y_pred)
        all_results[name] = metrics
        image_level_preds[name] = y_pred

        if y_true_ref is None:
            y_true_ref = y_true

        print_metrics(metrics, label="New Results:")

        # Compare with results.md
        old = EXPECTED.get(name, {})
        if old:
            print_comparison_table(name, metrics, old)

        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()

    # --- Ensemble evaluation ---
    if len(image_level_preds) > 1 and y_true_ref is not None:
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION (Simple Average — Test Set)")
        print("=" * 70)

        ensemble_preds = np.mean(list(image_level_preds.values()), axis=0)
        ensemble_metrics = compute_evaluation_metrics(y_true_ref, ensemble_preds)
        all_results["Ensemble (all)"] = ensemble_metrics

        print_metrics(ensemble_metrics, label="New Ensemble Results:")
        print_comparison_table("Ensemble", ensemble_metrics, EXPECTED_ENSEMBLE)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("FULL SUMMARY TABLE")
    print("=" * 70)

    header = f"  {'Model':<20s}"
    for key in METRIC_KEYS:
        header += f" {key:>12s}"
    print(header)
    print(f"  {'-' * (20 + 13 * len(METRIC_KEYS))}")

    for name, metrics in all_results.items():
        row = f"  {name:<20s}"
        for key in METRIC_KEYS:
            row += f" {metrics.get(key, float('nan')):>12.2f}"
        print(row)

    # --- Old results for reference ---
    print(f"\n  {'--- Old results.md ---'}")
    for name in list(EXPECTED.keys()) + ["Ensemble (all 4)"]:
        old = EXPECTED.get(name, EXPECTED_ENSEMBLE if "Ensemble" in name else {})
        row = f"  {name:<20s}"
        for key in METRIC_KEYS:
            row += f" {old.get(key, float('nan')):>12.2f}"
        print(row)

    # --- Save results ---
    summary_rows = []
    for name, metrics in all_results.items():
        row = {"Model": name}
        row.update(metrics)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(RESULTS_DIR, 'vit2_verification_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nResults saved to {summary_path}")

    # Save per-image predictions
    if y_true_ref is not None and image_level_preds:
        pred_df_data = {'TrueAge': y_true_ref}
        for name, preds in image_level_preds.items():
            pred_df_data[name] = preds
        if len(image_level_preds) > 1:
            pred_df_data['Ensemble'] = np.mean(list(image_level_preds.values()), axis=0)
        pred_df = pd.DataFrame(pred_df_data)
        pred_path = os.path.join(RESULTS_DIR, 'vit2_test_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"Per-image predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
