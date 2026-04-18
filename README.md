# Age Estimation from Offline Handwriting

This repository contains the code for the paper:

> **Age Estimation from Offline Handwriting: A Case Study Using Regression-Based Modeling**
> Irina Rabaev, Lior Abergel, Maor Merling, Marina Litvak
> *Pattern Recognition Letters*, 2026

## Dataset

This study uses the **HHD dataset** (Rabaev et al., 2020), which contains 885 handwriting samples from writers aged 8 to 60.

> I. Rabaev, B. K. Barakat, A. Churkin, and J. El-Sana, "The HHD dataset," in *Proc. 16th Int. Conf. Front. Handwriting Recognit. (ICFHR)*, 2020.

The dataset is available on Kaggle: **https://www.kaggle.com/datasets/liorabergel/hhd-age**

### Automatic Download

The dataset is downloaded automatically when running any experiment. On first run, you will be prompted for your Kaggle credentials. This requires the `kagglehub` package:

```bash
pip install kagglehub
```

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

Install `kagglehub` for automatic dataset download, then install the appropriate framework dependencies:

```bash
pip install kagglehub
```

**TensorFlow environment** (experiments 01--08):
```bash
pip install -r requirements_tf.txt
```

**PyTorch environment** (experiments 09--10):
```bash
pip install -r requirements_torch.txt
```

**Note:** ViT experiments (06, 07) require setting `TF_USE_LEGACY_KERAS=1` before running, as the `keras_cv_attention_models` library depends on legacy Keras.

## Experiments

### Mapping to Paper

The table below maps each experiment folder to the corresponding paper sections and tables.

| Folder | Paper Reference | Description |
|--------|----------------|-------------|
| `05_SOTA_CNN_CrossVal` | Table 2 (top) | CNN individual models, 5-fold stratified group CV |
| `07_ViT_CrossVal` | Table 2 (middle) | ViT individual models, 5-fold stratified group CV |
| `10_DiT_CrossVal` | Table 2 (bottom) | DiT individual models, 5-fold stratified group CV |
| `03_SOTA_CNN_Ensemble_Augmented` | Table 3 (top) | CNN ensembles on official HHD split |
| `09_DiT` | Table 3 (bottom), Table 4 | DiT ensembles and hybrid ensemble on official HHD split |
| `03_SOTA_CNN_Ensemble_Augmented_Grad_CAM` | Figure 4, Section 4.3 | Grad-CAM visualizations |
| `01_Baseline_CNN` | -- | Development: custom baseline CNN |
| `02_SOTA_CNN_Ensemble` | -- | Development: initial transfer learning |
| `04_SOTA_Scaling_Scheduler` | -- | Development: label scaling and LR scheduling |
| `06_ViT` | -- | Development: initial ViT experiments |
| `08_CNN_Tuning_Augmented` | -- | Development: hyperparameter tuning |

### Running an Experiment

Each experiment is self-contained in its folder. To run:

```bash
cd <experiment_folder>
python <script_name>.py
```

For example:
```bash
cd 05_SOTA_CNN_CrossVal
python train_sota_cv.py
```

Results for each experiment are documented in `results.md` within the experiment folder, including the full training configuration and all evaluation metrics.

### Key Hyperparameters

All experiments share a common preprocessing pipeline:
- Image resizing to 800px height (aspect ratio preserved)
- Patch extraction: 400x400 pixels, stride 200
- Pixel normalization to [0, 1]
- Augmentation: rotation (+/-15 degrees), zoom (up to 10%), brightness/contrast adjustment, Gaussian noise
- Empty patch filtering: intensity threshold = 0.0054

**CNN experiments (03, 05):**
- Models: ResNet50, InceptionV3, DenseNet121, InceptionResNetV2, EfficientNetV2M
- Pretrained weights: ImageNet-1K
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 128
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**ViT experiments (07):**
- Models: SwinV2-Tiny (256x256), MobileViT-XXS (256x256), ConvNeXtV2-Tiny (224x224), TinyViT-11M (224x224)
- Pretrained weights: ImageNet-1K (all four models; `pretrained="imagenet"` in `keras_cv_attention_models`)
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 64
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**DiT experiments (09, 10):**
- Models: DiT-Base, DiT-Large, DiT-Base (RVL-CDIP), DiT-Large (RVL-CDIP)
- Pretrained weights: IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants)
- Input: 224x224 via BeitImageProcessor
- Training: 15 epochs frozen backbone + 30 epochs fine-tuning
- Batch size: 4 (experiment 09), dynamic with gradient accumulation + FP16 (experiment 10)
- Optimizer: AdamW (1e-4 frozen, 1e-5 fine-tuning)

## Results Summary

Results are from 5-fold stratified group cross-validation (Table 2 in the paper), reported as mean ± std.
See each experiment's `results.md` for full metrics and ensemble configurations.

| Model Family | Best Individual Model | MAE (years) | R² |
|-------------|-----------------------|-------------|----|
| CNN | ResNet50 | 5.41 ± 0.78 | 0.10 ± 0.06 |
| ViT | MobileViT-XXS | 7.96 ± 0.53 | −0.05 ± 0.09 |
| DiT | DiT-Base (RVL-CDIP) | 5.85 ± 0.84 | 0.32 ± 0.12 |

Best results on the official HHD split (Table 3 in the paper):

| Configuration | MAE (years) | R² |
|--------------|-------------|----|
| CNN: Best 3 Ensemble, Grid Search | 2.85 | 0.14 |
| DiT-Base (RVL-CDIP), individual | 2.35 | 0.51 |

## Repository Structure

```
Age_Estimation/
  01_Baseline_CNN/              # Baseline custom CNN
    train_baseline_cnn.py
    results.md                  # Training config + evaluation metrics
  02_SOTA_CNN_Ensemble/         # Transfer learning with 5 SOTA CNNs
    train_sota_ensemble.py
    results.md
  03_SOTA_CNN_Ensemble_Augmented/          # Advanced augmentation + weighted ensembles
    train_sota_augmented.py
    results.md
  03_SOTA_CNN_Ensemble_Augmented_Grad_CAM/ # Grad-CAM visualizations
    visualize_gradcam.py
  04_SOTA_Scaling_Scheduler/    # Label scaling + LR scheduling
    train_sota_scaling.py
    results.md
  05_SOTA_CNN_CrossVal/         # CNN 5-fold stratified group CV
    train_sota_cv.py
    results.md
  06_ViT/                       # Initial ViT experiments
    train_vit.py
    results.md
  07_ViT_CrossVal/              # ViT 5-fold stratified group CV
    train_vit_cv.py
    results.md
  08_CNN_Tuning_Augmented/      # Hyperparameter tuning
    train_cnn_tuning.py
    results.md
  09_DiT/                       # Document Image Transformer experiments
    train_dit.py
    results.md
  10_DiT_CrossVal/              # DiT 5-fold stratified group CV
    train_dit_cv.py
    results.md
  data/                         # Downloaded automatically via kagglehub
  download_dataset.py           # Dataset download utility
  requirements_tf.txt           # TensorFlow dependencies (experiments 01–08)
  requirements_torch.txt        # PyTorch dependencies (experiments 09–10)
  LICENSE
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
