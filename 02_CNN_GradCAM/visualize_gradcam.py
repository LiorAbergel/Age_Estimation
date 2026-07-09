"""Grad-CAM gallery for the CNN age regressors (paper Fig. 5, Sec. 4.3).

Loads a trained backbone from the CNN-ensemble experiment (Experiment 01) and
uses Gradient-weighted Class Activation Mapping to show which handwriting
regions drive the predicted age. The paper (Fig. 5) reports Grad-CAM for
InceptionV3, the best-performing individual CNN in cross-validation, using its
final inception block (``mixed10``); that is the default here. The script can
also visualise any of the other trained CNN backbones for comparison.

Rather than emitting a couple of fixed figures, this script produces a browsable
gallery so the best examples can be picked by hand:

* It sweeps several conv layers at different resolutions. Grad-CAM at
  EfficientNet's final layer (~13x13 for a 400px patch) is coarse and tends to
  light up patch edges; mid-resolution layers (~25x25, ~50x50) usually localise
  onto strokes. ``--layers auto`` picks the coarsest few distinct resolutions.
* Example images are chosen at the IMAGE level from the committed
  ``test_image_predictions.csv`` -- the best (most accurate) and worst
  (least accurate) writers per age group. For each chosen image we Grad-CAM the
  single patch whose prediction is closest to that image's page-level
  prediction, i.e. a representative tile rather than one cherry-picked to match
  the label. Individual 3-panel figures plus a montage contact sheet per layer.

Two design points that matter for correctness:

1. Preprocessing is normalisation to [0, 1] (``/255.0``), matching exactly how
   the backbones were trained in ``01_CNN_Ensemble/train_cnn_ensemble.py``. We
   deliberately do NOT apply each architecture's canonical preprocessing (e.g.
   EfficientNet expects [0, 255]); the goal is to reproduce what the trained
   model actually saw, not to "fix" the pipeline.

2. The training script nests the backbone inside the functional model
   (``x = base_model(inputs)``), so the target conv layer lives inside a nested
   sub-model. Grad-CAM therefore builds the gradient model against the nested
   backbone and re-applies the regression head (GAP -> Dropout -> Dense) to
   recover the true scalar-age output.

Usage (run from the repository root):
    python 02_CNN_GradCAM/visualize_gradcam.py                       # InceptionV3, auto layers, K=5
    python 02_CNN_GradCAM/visualize_gradcam.py --top-k 8
    python 02_CNN_GradCAM/visualize_gradcam.py --layers mixed10,mixed9
    python 02_CNN_GradCAM/visualize_gradcam.py --model InceptionV3,EfficientNetV2M  # compare two
    python 02_CNN_GradCAM/visualize_gradcam.py --model all
"""
import os
# NOTE: the trained weights were saved with Keras 3 (native .keras format), so
# we must load them with Keras 3 -- do NOT set TF_USE_LEGACY_KERAS here, or
# tf_keras (Keras 2) fails with "cannot import keras.src.models.functional".

import argparse
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Model, load_model

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from download_dataset import ensure_dataset

# --- Configuration ---
CONFIG = {
    "PATCH_SIZE": (400, 400),
    "STRIDE": 200,
    "STANDARD_SIZE": 800,
    "THR": 0.0054,            # drop near-blank patches (matches training pipeline)
    "BATCH_SIZE": 32,
    "DATA_DIR": "./data",
    "CSV_PATH": "./data/NewAgeSplit.csv",
    # Trained weights from Experiment 01 (CNN ensemble).
    "MODELS_DIR": "./models/experiment_01",
    "OUTPUT_DIR": "./results/gradcam_analysis",
}

# Age groups, matching the paper's error analysis (Sec. 4.4 / Fig. 6).
AGE_BINS = ["<=15", "16-25", "26-50", ">50"]


def _in_colab():
    """True when running on Google Colab, whose local disk is ephemeral."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _apply_colab_paths():
    """On Colab, load weights (and write outputs) from Google Drive."""
    from pathlib import Path
    drive_root = Path("/content/drive")
    try:
        if not (drive_root / "MyDrive").exists():
            from google.colab import drive
            print("Colab detected: mounting Google Drive to load trained models...")
            drive.mount(str(drive_root))
    except Exception as exc:
        print(f"WARNING: could not mount Google Drive ({exc}).")
    persist_base = drive_root / "MyDrive" / "Age_Estimation" / "01_CNN_Ensemble"
    CONFIG["MODELS_DIR"] = str(persist_base / "models")
    CONFIG["OUTPUT_DIR"] = str(persist_base / "gradcam_analysis")
    print(f"Loading models from Google Drive: {CONFIG['MODELS_DIR']}")


# --- Preferred last-conv target layer per architecture ---
# Used when the model runs with the default single layer; the auto ladder
# ignores this and discovers layers by resolution instead.
LAYER_NAMES = {
    "ResNet50": "conv5_block3_out",
    "InceptionV3": "mixed10",
    "InceptionResNetV2": "conv_7b_ac",
    "DenseNet121": "relu",
    "EfficientNetV2M": "top_activation",
}


# --- Data Processing (mirrors the training pipeline) ---
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
        new_h, new_w = calculate_resized_dimensions(
            h, w, CONFIG["PATCH_SIZE"][0], CONFIG["STRIDE"], CONFIG["STANDARD_SIZE"])
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
    ids = tf.fill([tf.shape(patches)[0]], row['File'])

    # Drop near-blank (background) patches, matching training so we never pick a
    # blank tile as a visualisation example.
    patch_means = tf.reduce_mean(patches, axis=[1, 2, 3])
    mask = patch_means > CONFIG["THR"]
    patches = tf.boolean_mask(patches, mask)
    labels = tf.boolean_mask(labels, mask)
    ids = tf.boolean_mask(ids, mask)

    return patches, labels, ids


def create_test_dataset(data_dir, labels_df, image_ids=None):
    subset_df = labels_df[labels_df['Set'] == 'test'].reset_index(drop=True)
    if image_ids is not None:
        subset_df = subset_df[subset_df['File'].isin(image_ids)].reset_index(drop=True)
    target_dir = os.path.join(data_dir, 'test')

    ds = tf.data.Dataset.from_tensor_slices(dict(subset_df))
    ds = ds.map(lambda row: process_image(row, target_dir), num_parallel_calls=tf.data.AUTOTUNE)

    # Flatten (batch-of-patches) -> (patch), then re-batch for efficient inference.
    ds = ds.flat_map(lambda patches, labels, ids: tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(patches),
         tf.data.Dataset.from_tensor_slices(labels),
         tf.data.Dataset.from_tensor_slices(ids))
    ))

    return ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)


# --- Backbone / layer discovery ---
def get_backbone_and_head(model):
    """Return (nested backbone sub-model, list of head layers after it).

    The training script builds ``Model(inputs, outputs)`` with the pretrained
    backbone nested as a single functional layer, followed by the regression
    head (GlobalAveragePooling2D -> Dropout -> Dense).
    """
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            return layer, model.layers[i + 1:]
    raise ValueError("No nested backbone sub-model found; unexpected architecture.")


def _layer_hw(layer):
    """Spatial (H, W) of a layer's output, or None if not a static 4-D tensor."""
    try:
        shape = layer.output.shape
    except Exception:
        return None
    if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
        return (int(shape[1]), int(shape[2]))
    return None


def build_layer_ladder(backbone, requested, max_resolutions=3):
    """Return an ordered list of ``(layer_name, (h, w))`` to sweep.

    ``requested`` is either an explicit comma-separated list of layer names or
    ``"auto"``. In auto mode we group all 4-D conv layers by output resolution,
    keep the deepest layer at each resolution (the stage's output activation),
    and return the coarsest ``max_resolutions`` distinct resolutions -- i.e. the
    final layer plus a couple of higher-resolution earlier layers.
    """
    if requested and requested.lower() != "auto":
        valid = {layer.name for layer in backbone.layers}
        ladder = []
        for name in (n.strip() for n in requested.split(",") if n.strip()):
            if name in valid:
                ladder.append((name, _layer_hw(backbone.get_layer(name))))
            else:
                print(f"  Requested layer '{name}' not in backbone; skipping.")
        if ladder:
            return ladder
        print("  No requested layers matched; falling back to auto ladder.")

    # Floor the ladder at the backbone's final feature-map size. Anything
    # smaller lives off the main path -- e.g. 1x1 squeeze-and-excitation layers,
    # whose Grad-CAM map collapses to a scalar and is useless for visualisation.
    floor_area = 1
    out_hw = _layer_hw(backbone)  # backbone is itself a Model with a 4-D output
    if out_hw is not None:
        floor_area = out_hw[0] * out_hw[1]

    # Auto: deepest layer per distinct (valid) resolution, then coarsest few.
    res_to_layer = {}
    for layer in backbone.layers:
        hw = _layer_hw(layer)
        if hw is not None and hw[0] * hw[1] >= floor_area and min(hw) >= 2:
            res_to_layer[hw] = layer.name  # later (deeper) layer overwrites -> stage output
    resolutions = sorted(res_to_layer, key=lambda hw: hw[0] * hw[1])  # coarsest first
    chosen = resolutions[:max_resolutions]
    return [(res_to_layer[hw], hw) for hw in chosen]


# --- Grad-CAM (build once per layer, compute per patch) ---
def build_grad_model(model, target_name):
    """Grad-model mapping backbone input -> (target activations, backbone output).

    Also returns the head layers so the true scalar age can be reconstructed.
    """
    backbone, head_layers = get_backbone_and_head(model)
    grad_model = Model(
        backbone.inputs,
        [backbone.get_layer(target_name).output, backbone.output],
    )
    return grad_model, head_layers


def compute_heatmap(grad_model, head_layers, img_array):
    """Grad-CAM heatmap of the predicted scalar age for a single patch.

    Works for any target layer: the gradient flows from the reconstructed head
    back through the remaining backbone to the target activations.
    """
    img = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img, training=False)
        tape.watch(conv_output)
        x = base_output
        for layer in head_layers:  # GAP -> Dropout (no-op) -> Dense
            x = layer(x, training=False)
        score = x[:, 0]

    grads = tape.gradient(score, conv_output)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.squeeze(conv_output @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    if heatmap.ndim != 2:
        # e.g. a 1x1 (squeeze-and-excitation) target layer has no spatial map.
        return None
    return heatmap


# --- Rendering ---
def make_overlay(heatmap, original_img, alpha=0.4):
    """Return a uint8 RGB overlay of the jet-coloured heatmap on the patch."""
    heatmap_u8 = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]        # (256, 3) in [0, 1]
    jet_heatmap = jet_colors[heatmap_u8]           # (h, w, 3) in [0, 1]

    jet_img = Image.fromarray(np.uint8(255 * jet_heatmap)).resize(
        (original_img.shape[1], original_img.shape[0]), Image.Resampling.BILINEAR)
    jet_heatmap = np.asarray(jet_img, dtype=np.float32)

    superimposed = jet_heatmap * alpha + original_img * 255.0 * (1 - alpha)
    return np.clip(superimposed, 0, 255).astype(np.uint8)


def save_three_panel(heatmap, original_img, title, filename):
    """Save the original patch, raw heatmap (with attention colorbar), and overlay."""
    overlay = make_overlay(heatmap, original_img)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title("Original Patch")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title("Attention Heatmap")
    ax2.axis('off')
    # Colorbar legend to the right of the heatmap: what colour means more attention.
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("Attention", rotation=90)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title(title)
    ax3.axis('off')

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_montage(cells, k, suptitle, path):
    """Contact sheet: rows = age bins, columns = best-K then worst-K overlays.

    ``cells[bin][mode]`` is a list of ``(record, overlay)`` tuples.
    """
    bins = list(cells.keys())
    n_rows, n_cols = len(bins), 2 * k
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.5 * n_rows), squeeze=False)
    for r, bin_name in enumerate(bins):
        for mode, col_offset in (("best", 0), ("worst", k)):
            items = cells[bin_name][mode]
            for c in range(k):
                ax = axes[r][col_offset + c]
                ax.axis('off')
                if c < len(items):
                    rec, overlay = items[c]
                    ax.imshow(overlay)
                    ax.set_title(f"{mode} #{c + 1}\nT{rec['true_age']:.0f} "
                                 f"P{rec['pred']:.0f} E{rec['error']:.0f}", fontsize=7)
        # Row label on the far left.
        axes[r][0].text(-0.35, 0.5, bin_name, transform=axes[r][0].transAxes,
                        fontsize=11, va='center', ha='right', rotation=90)
    fig.suptitle(f"{suptitle}   (left {k} = best | right {k} = worst)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Candidate selection: best/worst IMAGES per age group (from the CSV) ---
def bin_for_age(age):
    if age <= 15:
        return "<=15"
    if age <= 25:
        return "16-25"
    if age <= 50:
        return "26-50"
    return ">50"


def select_images_from_csv(preds_df, model_col, k):
    """Pick the k most- and least-accurate test IMAGES per age group.

    Selection uses the committed image-level predictions (``model_col`` column of
    ``test_image_predictions.csv``), never the label -- so an "accurate" example
    is a writer the model genuinely predicts well, not a patch cherry-picked to
    match the true age.
    """
    df = preds_df[["ImageID", "TrueAge", model_col]].dropna().copy()
    df["err"] = (df[model_col] - df["TrueAge"]).abs()
    df["bin"] = df["TrueAge"].apply(bin_for_age)

    selection = {b: {"best": [], "worst": []} for b in AGE_BINS}
    for b in AGE_BINS:
        sub = df[df["bin"] == b]
        for mode, part in (("best", sub.nsmallest(k, "err")),
                           ("worst", sub.nlargest(k, "err"))):
            for rank, (_, row) in enumerate(part.iterrows(), 1):
                selection[b][mode].append({
                    "id": row["ImageID"],
                    "true": float(row["TrueAge"]),
                    "image_pred": float(row[model_col]),
                    "image_err": float(row["err"]),
                    "bin": b,
                    "mode": mode,
                    "rank": rank,
                })
    return selection


def gather_representative_patches(model, data_dir, labels_df, image_pred_lookup):
    """For each chosen image, keep the patch whose prediction is closest to the
    image-level prediction -- the tile most representative of the model's
    behaviour for that writer (not the tile that best matches the true age)."""
    ds = create_test_dataset(data_dir, labels_df, image_ids=set(image_pred_lookup))
    rep = {}
    for patches, _labels, ids in tqdm(ds, desc="Representative patches"):
        preds = model.predict(patches, verbose=0).flatten()
        id_strs = ids.numpy()
        for i, pred in enumerate(preds):
            image_id = id_strs[i].decode("utf-8")
            target = image_pred_lookup.get(image_id)
            if target is None:
                continue
            dist = abs(float(pred) - target)
            if image_id not in rep or dist < rep[image_id]["dist"]:
                rep[image_id] = {"img": patches[i].numpy(),
                                 "patch_pred": float(pred), "dist": dist}
    return rep


def _safe_id(image_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(image_id))


# --- Main Logic ---
def process_model(model_name, models_dir, output_dir, data_dir, labels_df, preds_df,
                  layers_arg, top_k):
    if model_name not in preds_df.columns:
        print(f"Skipping {model_name}: no prediction column in the CSV.")
        return

    model_path = os.path.join(models_dir, f"{model_name}_best_model.keras")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Select example IMAGES from the committed predictions -- never the label.
    selection = select_images_from_csv(preds_df, model_name, top_k)
    lookup = {rec["id"]: rec["image_pred"]
              for b in AGE_BINS for mode in ("best", "worst")
              for rec in selection[b][mode]}
    if not lookup:
        print(f"No images selected for {model_name}.")
        return

    print(f"\nProcessing {model_name}...")
    model = load_model(model_path)
    backbone, _ = get_backbone_and_head(model)

    ladder = build_layer_ladder(backbone, layers_arg)
    print("Sweeping layers:")
    for name, hw in ladder:
        res = f"{hw[0]}x{hw[1]}" if hw else "?"
        print(f"  {name:<32} ({res})")

    rep = gather_representative_patches(model, data_dir, labels_df, lookup)

    print("\nSelected images (per age group; image-level predictions):")
    for b in AGE_BINS:
        for mode in ("best", "worst"):
            for rec in selection[b][mode]:
                patch_pred = rep.get(rec["id"], {}).get("patch_pred", float("nan"))
                print(f"  [{b:>5} {mode:<5} #{rec['rank']}] id={rec['id']:<28} "
                      f"true={rec['true']:.1f} img_pred={rec['image_pred']:.1f} "
                      f"err={rec['image_err']:.1f} patch_pred={patch_pred:.1f}")

    for layer_name, hw in ladder:
        grad_model, head_layers = build_grad_model(model, layer_name)
        layer_dir = os.path.join(output_dir, f"{model_name}_{layer_name}")
        os.makedirs(layer_dir, exist_ok=True)

        cells = {b: {"best": [], "worst": []} for b in AGE_BINS}
        for b in AGE_BINS:
            for mode in ("best", "worst"):
                for rec in selection[b][mode]:
                    r = rep.get(rec["id"])
                    if r is None:
                        print(f"  No patch for {rec['id']} ({b}/{mode}); skipping.")
                        continue
                    heatmap = compute_heatmap(grad_model, head_layers,
                                              np.expand_dims(r["img"], axis=0))
                    if heatmap is None:
                        print(f"  Gradient was None for {b}/{mode} #{rec['rank']}; skipping.")
                        continue
                    title = (f"{model_name} | {layer_name} | {b} {mode} #{rec['rank']}\n"
                             f"True: {rec['true']:.1f} | ImgPred: {rec['image_pred']:.1f} "
                             f"(err {rec['image_err']:.1f}) | PatchPred: {r['patch_pred']:.1f}")
                    fname = os.path.join(
                        layer_dir, f"{b}_{mode}_{rec['rank']}_{_safe_id(rec['id'])}.png")
                    save_three_panel(heatmap, r["img"], title, fname)
                    cell_rec = {"true_age": rec["true"], "pred": rec["image_pred"],
                                "error": rec["image_err"]}
                    cells[b][mode].append((cell_rec, make_overlay(heatmap, r["img"])))

        res = f"{hw[0]}x{hw[1]}" if hw else "?"
        montage_path = os.path.join(output_dir, f"montage_{model_name}_{layer_name}.png")
        save_montage(cells, top_k, f"{model_name} · {layer_name} ({res})", montage_path)
        print(f"Saved layer '{layer_name}': individual figures in {layer_dir}/ "
              f"and montage {montage_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM gallery for CNN age regressors (paper Fig. 5).")
    parser.add_argument("--model", default="InceptionV3",
                        help="Backbone(s) to visualise: a single name, a comma-separated "
                             "list (e.g. 'InceptionV3,EfficientNetV2M'), or 'all'. "
                             "Default: InceptionV3 (the model reported in the paper).")
    parser.add_argument("--layers", default="auto",
                        help="'auto' (coarsest few resolutions) or a comma-separated "
                             "list of backbone layer names. Default: auto.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of best and worst patches per age group. Default: 5.")
    parser.add_argument("--models-dir", default=None,
                        help=f"Weights directory (default: {CONFIG['MODELS_DIR']}; "
                             "auto-redirected to Google Drive on Colab).")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output directory (default: {CONFIG['OUTPUT_DIR']}).")
    parser.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    parser.add_argument("--csv-path", default=CONFIG["CSV_PATH"])
    parser.add_argument("--preds-csv",
                        default="01_CNN_Ensemble/predictions/test_image_predictions.csv",
                        help="Image-level predictions CSV used to select example images.")
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"])
    args = parser.parse_args()

    if _in_colab() and (args.models_dir is None or args.output_dir is None):
        _apply_colab_paths()
    if args.models_dir is not None:
        CONFIG["MODELS_DIR"] = args.models_dir
    if args.output_dir is not None:
        CONFIG["OUTPUT_DIR"] = args.output_dir
    CONFIG["DATA_DIR"] = args.data_dir
    CONFIG["CSV_PATH"] = args.csv_path
    CONFIG["BATCH_SIZE"] = args.batch_size

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    print("Loading test data...")
    ensure_dataset(CONFIG["DATA_DIR"])
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f"Error: CSV not found at {CONFIG['CSV_PATH']}.")
        return
    labels_data = pd.read_csv(CONFIG["CSV_PATH"])
    if not os.path.exists(args.preds_csv):
        print(f"Error: predictions CSV not found at {args.preds_csv}.")
        return
    preds_df = pd.read_csv(args.preds_csv)

    if args.model.lower() == "all":
        model_names = list(LAYER_NAMES)
    else:
        model_names = [m.strip() for m in args.model.split(",") if m.strip()]

    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    for model_name in model_names:
        process_model(model_name, CONFIG["MODELS_DIR"], CONFIG["OUTPUT_DIR"],
                      CONFIG["DATA_DIR"], labels_data, preds_df, args.layers, args.top_k)


if __name__ == "__main__":
    main()
