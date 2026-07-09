# Age Estimation from Offline Handwriting

This repository contains the code for the paper:

> **Age Estimation from Offline Handwriting: A Case Study Using Regression-Based Modeling**
> Lior Abergel, Maor Merling, Marina Litvak, Irina Rabaev

## Dataset

> **Dataset Download:** [https://zenodo.org/records/14996257](https://zenodo.org/records/14996257)

This study uses the **HHD dataset** (Rabaev et al., 2020), which contains 885 handwriting samples from writers aged 8 to 60.

> I. Rabaev, B. K. Barakat, A. Churkin, and J. El-Sana, "The HHD dataset," in *Proc. 16th Int. Conf. Front. Handwriting Recognit. (ICFHR)*, 2020.

### Automatic Download

The dataset is downloaded automatically when running any experiment. It is fetched from the open-access Zenodo record, so **no account or credentials are required** (only the Python standard library plus `pandas`).

To download the dataset manually:

```bash
python download_dataset.py
```

The data will be placed in the `data/` directory with the following structure:

```
data/
  NewAgeSplit.csv    # Metadata file
  train/             # Training images (.tif format)
  val/               # Validation images
  test/              # Test images
```

The `NewAgeSplit.csv` file maps each image to its metadata:
- `File`: image filename
- `AgeGroup`: age group (1--4)
- `Set`: data split (train/val/test)
- `Age`: exact age in years
- `WriterNumber`: unique writer identifier

## Installation

The dataset downloads automatically from Zenodo (no extra package required). Install the appropriate framework dependencies:

**TensorFlow environment** (CNN, ViT, and Grad-CAM experiments):
```bash
pip install -r requirements_tf.txt
```

**PyTorch environment** (DiT experiments):
```bash
pip install -r requirements_torch.txt
```

**Note:** The `keras_cv_attention_models` library used by the ViT experiments (`04_ViT_Ensemble`, `05_ViT_CrossVal`) depends on legacy Keras. The training and reproduction scripts set `TF_USE_LEGACY_KERAS=1` internally where needed, so no manual configuration is required.

## Experiments

### Mapping to Paper

The table below maps each experiment folder to the corresponding paper sections and tables.

| Folder | Paper Reference | Description |
|--------|----------------|-------------|
| `03_CNN_CrossVal` | Table 2 (top) | CNN individual models, 5-fold stratified group CV |
| `05_ViT_CrossVal` | Table 2 (middle) | ViT individual models, 5-fold stratified group CV |
| `07_DiT_CrossVal` | Table 2 (bottom) | DiT individual models, 5-fold stratified group CV |
| `01_CNN_Ensemble` | Table 3 (top) | CNN ensembles on official HHD split |
| `04_ViT_Ensemble` | Table 3 (middle), Table 4 | ViT ensembles on official HHD split; hybrid InceptionV3 + MobileViT-XXS ensemble (`reproduce_hybrid.py`) |
| `06_DiT_Ensemble` | Table 3 (bottom) | DiT ensembles on official HHD split |
| `02_CNN_GradCAM` | Figure 5 | Grad-CAM visualizations (InceptionV3) |
| `08_Significance` | "Ensemble Models" section | Paired bootstrap significance tests between ensembles |

### Running an Experiment

Each experiment is self-contained in its folder. Run scripts **from the repository root**:

```bash
python <experiment_folder>/<script_name>.py
```

For example:
```bash
python 03_CNN_CrossVal/train_cnn_cv.py
```

### Reproducing Results (no training required)

Every experiment folder ships a `reproduce_results.py` that recomputes all reported
metrics from committed predictions — no GPU needed:

```bash
# Fast path (default) — recompute metrics from the committed predictions
python 01_CNN_Ensemble/reproduce_results.py

# Full path — regenerate predictions from model weights (requires GPU + Zenodo weights).
# Every folder uses the same interface: --mode {auto,fast,full} (default: auto).
python 01_CNN_Ensemble/reproduce_results.py --mode full
python 06_DiT_Ensemble/reproduce_results.py --mode full
```

Results for each experiment are documented in `results/results.md` within the
experiment folder, including the full training configuration and all evaluation
metrics. See each folder's `README.md` for a per-experiment guide.

### Key Hyperparameters

All experiments share a common preprocessing pipeline:
- Image resizing to 800px height (aspect ratio preserved)
- Patch extraction: 400x400 pixels, stride 200
- Pixel normalization to [0, 1]
- Augmentation: rotation (+/-15 degrees), zoom (up to 10%), Gaussian noise, brightness adjustment, and contrast adjustment
- Empty patch filtering: intensity threshold = 0.0054

**CNN experiments (`01_CNN_Ensemble`, `03_CNN_CrossVal`):**
- Models: ResNet50, InceptionV3, DenseNet121, InceptionResNetV2, EfficientNetV2M
- Pretrained weights: ImageNet-1K
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 128
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**ViT experiments (`04_ViT_Ensemble`, `05_ViT_CrossVal`):**
- Models: SwinV2-Tiny (256x256), MobileViT-XXS (256x256), ConvNeXtV2-Tiny (224x224), TinyViT-11M (224x224)
- Pretrained weights: ImageNet-1K (all four models; `pretrained="imagenet"` in `keras_cv_attention_models`)
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 128 (`04_ViT_Ensemble` and `05_ViT_CrossVal`)
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**DiT experiments (`06_DiT_Ensemble`, `07_DiT_CrossVal`):**
- Models: DiT-Base, DiT-Large, DiT-Base (RVL-CDIP), DiT-Large (RVL-CDIP)
- Pretrained weights: IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants)
- Input: 224x224 via BeitImageProcessor
- Regression head: Global mean-pooling (patch tokens), Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Physical batch size: 128 (Base variants) / 16 (Large variants)
- Effective patch batch size: 128 patches/step in both `06_DiT_Ensemble` and `07_DiT_CrossVal` via gradient accumulation where needed
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

## Results Summary

The numbers below are produced under a unified training protocol (1e-3/1e-4,
50/10 epochs, MSE) shared across all architectures, and match the results
reported in the paper. Each experiment's `results/results.md` holds the full
metric set.

Best individual models, 5-fold stratified group cross-validation (Table 2), mean ± std:

| Model Family | Best Individual Model | MAE (years) | R² |
|-------------|-----------------------|-------------|----|
| CNN | InceptionV3 | 4.67 ± 0.48 | 0.20 ± 0.12 |
| ViT | MobileViT-XXS | 4.69 ± 0.22 | 0.18 ± 0.09 |
| DiT | DiT-Base (RVL-CDIP) | 5.01 ± 0.53 | 0.19 ± 0.11 |

Best ensemble per family on the official HHD split (Table 3), plus the hybrid
ensemble (Table 4):

| Configuration | MAE (years) | R² |
|--------------|-------------|----|
| CNN: Best 4 / Full Ensemble (Grid Search) | 2.73 | 0.07–0.08 |
| ViT: Best 3 Ensemble (Grid Search) | 2.77 | 0.15 |
| DiT: Best 2 Ensemble (Grid Search) | 3.17 | 0.17 |
| Hybrid: InceptionV3 + MobileViT-XXS (mean) | 2.81 | 0.07 |

CNN ensembles achieve the lowest MAE (2.73), closely followed by ViT ensembles (2.77),
which offer the best RMSE (5.57) and R² (0.22) among all evaluated configurations.
In cross-validation, CNN and lightweight ViT individual models are comparable and both
edge out DiT. A paired bootstrap test (`08_Significance/`) confirms that the CNN ensemble
significantly outperforms its strongest single backbone and that both the CNN and ViT
ensembles significantly outperform the best DiT ensemble.

## Repository Structure

Every experiment folder is self-contained and follows the same layout: a per-folder
`README.md`, the training script, a `reproduce_results.py`, committed `predictions/`,
generated `reproduction_output/`, and a `results/results.md` with the full metric set.

> **Note on code organization.** Each experiment folder is intentionally standalone:
> shared building blocks (patch extraction, augmentation, metrics, ensemble weighting)
> are duplicated across folders by design so that any single experiment can be run,
> read, and reproduced in isolation without a shared package dependency. The only
> shared module is `download_dataset.py` at the repository root.

```
Age_Estimation/
  01_CNN_Ensemble/                         # CNN ensembles (Table 3 top)
    README.md                              # Folder guide (contents, models, how to run)
    train_cnn_ensemble.py                  # Training pipeline (5 CNN backbones + ensembles)
    reproduce_results.py                   # Reproduce results (CSV fast path / Zenodo weights)
    predictions/                           # Per-image predictions (test & val)
    reproduction_output/                   # Generated CSVs from reproduce_results.py
    results/
      results.md                           #   Training config + full metrics (matches the paper)
  02_CNN_GradCAM/                          # Grad-CAM visualizations (Figure 5)
    visualize_gradcam.py
    results.md
  03_CNN_CrossVal/                         # CNN 5-fold stratified group CV (Table 2 top)
  04_ViT_Ensemble/                         # ViT ensembles (Table 3 middle, Table 4)
  05_ViT_CrossVal/                         # ViT 5-fold stratified group CV (Table 2 middle)
  06_DiT_Ensemble/                         # DiT experiments + ensembles (Table 3 bottom)
  07_DiT_CrossVal/                         # DiT 5-fold stratified group CV (Table 2 bottom)
  08_Significance/                         # Paired bootstrap significance tests
    paired_bootstrap.py
    README.md
  data/                                    # Downloaded automatically from Zenodo
  download_dataset.py                      # Dataset download utility
  requirements_tf.txt                      # TensorFlow dependencies (CNN, ViT, Grad-CAM)
  requirements_torch.txt                   # PyTorch dependencies (DiT)
  LICENSE
```

> `03_CNN_CrossVal`, `04_ViT_Ensemble`, `05_ViT_CrossVal`, `06_DiT_Ensemble`, and
> `07_DiT_CrossVal` follow the same layout shown expanded for `01_CNN_Ensemble` above.
> CV folders name their predictions file `oof_predictions.csv`. `02_CNN_GradCAM` and
> `08_Significance` are utility folders that keep a flat layout.

## Citation

This paper is currently under submission to the ICDAR Workshop on Document
Analysis of Low-Resource Languages (DALL) 2026 and is not yet published. A
full citation (with venue, pages, and DOI) will be added here once available.

```
Lior Abergel, Maor Merling, Marina Litvak, Irina Rabaev.
"Age Estimation from Offline Handwriting: A Case Study Using Regression-Based Modeling."
Under submission, 2026.
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
