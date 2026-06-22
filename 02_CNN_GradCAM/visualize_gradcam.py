"""
Experiment 04: Explainability with Grad-CAM

Overview:
This script loads the pre-trained models from Experiment 03 and utilizes 
Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the 
model's focus. It specifically targets:
1. Various Age Groups (Young, Middle, Old)
2. Performance Extremes (Best predictions vs. Worst failures)
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

# Import model loading functions
from tensorflow.keras.models import Model, load_model

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
    "DATA_DIR": "./data",  
    "CSV_PATH": "./data/NewAgeSplit.csv",
    # Pointing to models from Experiment 01 (CNN ensemble)
    "MODELS_DIR": "./models/experiment_01",
    "OUTPUT_DIR": "./results/gradcam_analysis"
}


def _in_colab():
    """True when running on Google Colab, whose local disk is ephemeral."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


# On Colab, 01_CNN_Ensemble/train_cnn_ensemble.py persists weights to Google
# Drive, so load them (and write outputs) from there instead of ephemeral disk.
if _in_colab():
    from pathlib import Path
    _drive_root = Path("/content/drive")
    try:
        if not (_drive_root / "MyDrive").exists():
            from google.colab import drive
            print("Colab detected: mounting Google Drive to load trained models...")
            drive.mount(str(_drive_root))
    except Exception as exc:
        print(f"WARNING: could not mount Google Drive ({exc}).")
    _persist_base = _drive_root / "MyDrive" / "Age_Estimation" / "01_CNN_Ensemble"
    CONFIG["MODELS_DIR"] = str(_persist_base / "models")
    CONFIG["OUTPUT_DIR"] = str(_persist_base / "gradcam_analysis")
    print(f"Loading models from Google Drive: {CONFIG['MODELS_DIR']}")

# --- Layer Targets for SOTA Architectures ---
# These are the standard last convolutional layers for Keras applications
LAYER_NAMES = {
    "ResNet50": "conv5_block3_out",
    "InceptionV3": "mixed10", 
    "InceptionResNetV2": "conv_7b_ac",
    "DenseNet121": "relu",  # The last relu before global pooling
    "EfficientNetV2M": "top_activation"
}

# --- Data Processing (Reused from Training) ---
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

def process_image(row, data_dir):
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
    
    # We return File ID too for tracking
    ids = tf.fill([tf.shape(patches)[0]], row['File'])
    
    return patches, labels, ids

def create_test_dataset(data_dir, labels_df):
    subset_df = labels_df[labels_df['Set'] == 'test'].reset_index(drop=True)
    target_dir = os.path.join(data_dir, 'test')
    
    ds = tf.data.Dataset.from_tensor_slices(dict(subset_df))
    ds = ds.map(lambda row: process_image(row, target_dir), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Flatten: (Batch_of_Patches) -> (Patch)
    ds = ds.flat_map(lambda patches, labels, ids: tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(patches),
         tf.data.Dataset.from_tensor_slices(labels),
         tf.data.Dataset.from_tensor_slices(ids))
    ))
    
    return ds.batch(1) # Batch 1 for easier individual processing

# --- Grad-CAM Implementation ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    
    # Handle nested models (Transfer Learning often wraps the base model)
    # Check if the model has a 'backbone' or if it's the model itself
    target_layer = None
    
    # Recursively search for layer (useful if nested in Functional model)
    try:
        target_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model([model.inputs], [target_layer.output, model.output])
    except ValueError:
        # If layer is inside a nested 'functional' layer (common in transfer learning)
        # We assume the first layer is the base model
        base_model = model.layers[0] # Assuming first layer is the CNN backbone
        try:
            target_layer = base_model.get_layer(last_conv_layer_name)
            grad_model = Model([base_model.inputs], [target_layer.output, base_model.output])
        except Exception as e:
            print(f"Could not find layer {last_conv_layer_name}. Skipping Grad-CAM.")
            return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # For regression, our class of interest is the scalar output itself
        score = preds[0]

    # Compute gradients of the score with respect to the feature map
    grads = tape.gradient(score, last_conv_layer_output)

    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in feature map by 'how important this channel is'
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, original_img, title, filename):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap
    jet = plt.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + original_img * 255.0 * 0.6
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Patch")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main Logic ---

def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    # 1. Load Data
    print("Loading test data...")
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print("Error: CSV not found.")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])
    test_ds = create_test_dataset(CONFIG["DATA_DIR"], labels_data)

    # 2. Select Sample Patches (Sampling Phase)
    # We will iterate through X samples to find good candidates for visualization
    print("Scanning test set for interesting samples...")
    
    samples = [] # List of dicts: {img, true_age, id}
    
    # Limit scanning to first 300 patches to save time, or adjust as needed
    SCAN_LIMIT = 300 
    
    for i, batch in tqdm(enumerate(test_ds), total=SCAN_LIMIT):
        if i >= SCAN_LIMIT: break
        
        patch, label, file_id = batch
        samples.append({
            "img": patch.numpy()[0], # (400,400,3)
            "true_age": label.numpy()[0],
            "id": file_id.numpy()[0].decode('utf-8')
        })

    # 3. Load Models & Visualize
    model_files = [f for f in os.listdir(CONFIG["MODELS_DIR"]) if f.endswith('.keras')]
    
    for model_file in model_files:
        model_name = model_file.split('_')[0] # e.g., "ResNet50"
        
        # Check if we support Grad-CAM for this architecture
        if model_name not in LAYER_NAMES:
            print(f"Skipping {model_name} (Layer target not defined in script)")
            continue
            
        print(f"\nProcessing {model_name}...")
        model_path = os.path.join(CONFIG["MODELS_DIR"], model_file)
        model = load_model(model_path)
        
        # Run inference on our collected samples
        results = []
        for s in samples:
            # Add batch dimension
            input_img = np.expand_dims(s["img"], axis=0)
            pred = model.predict(input_img, verbose=0)[0][0]
            err = abs(pred - s["true_age"])
            
            results.append({
                "sample": s,
                "pred": pred,
                "error": err
            })
            
        # --- Categorize Results ---
        # 1. Young (<20), Middle (20-50), Old (>50)
        # 2. Best (Low Error), Worst (High Error)
        
        categories = {
            "Young_Best":  [r for r in results if r["sample"]["true_age"] < 25],
            "Young_Worst": [r for r in results if r["sample"]["true_age"] < 25],
            "Old_Best":    [r for r in results if r["sample"]["true_age"] > 50],
            "Old_Worst":   [r for r in results if r["sample"]["true_age"] > 50],
            "General_Fail": results # Just high error in general
        }
        
        # Sort lists
        categories["Young_Best"].sort(key=lambda x: x["error"])
        categories["Young_Worst"].sort(key=lambda x: x["error"], reverse=True)
        categories["Old_Best"].sort(key=lambda x: x["error"])
        categories["Old_Worst"].sort(key=lambda x: x["error"], reverse=True)
        categories["General_Fail"].sort(key=lambda x: x["error"], reverse=True)
        
        # Select 1 top candidate from each category
        candidates = {
            "Young_Accurate": categories["Young_Best"][0] if categories["Young_Best"] else None,
            "Young_Inaccurate": categories["Young_Worst"][0] if categories["Young_Worst"] else None,
            "Old_Accurate": categories["Old_Best"][0] if categories["Old_Best"] else None,
            "Old_Inaccurate": categories["Old_Worst"][0] if categories["Old_Worst"] else None,
            "Worst_Overall": categories["General_Fail"][0] if categories["General_Fail"] else None
        }

        # Generate Grad-CAM for these candidates
        for cat_name, data in candidates.items():
            if not data: continue
            
            img_data = data["sample"]["img"]
            true_age = data["sample"]["true_age"]
            pred_age = data["pred"]
            
            # Run Grad-CAM
            input_img = np.expand_dims(img_data, axis=0)
            heatmap = make_gradcam_heatmap(input_img, model, LAYER_NAMES[model_name])
            
            if heatmap is None: continue
            
            # Save Image
            title = f"{model_name} | {cat_name}\nTrue: {true_age:.1f} | Pred: {pred_age:.1f} | Err: {abs(pred_age-true_age):.1f}"
            filename = os.path.join(CONFIG["OUTPUT_DIR"], f"{model_name}_{cat_name}.png")
            
            save_and_display_gradcam(None, heatmap, img_data, title, filename)
            print(f"Saved visualization: {filename}")

if __name__ == "__main__":
    ensure_dataset(CONFIG["DATA_DIR"])
    main()