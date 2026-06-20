# Age Estimation from Offline Handwriting

This repository contains the code for the paper:

> **Age Estimation from Offline Handwriting: A Case Study Using Regression-Based Modeling**
> Irina Rabaev, Lior Abergel, Maor Merling, Marina Litvak

## Dataset

> **Dataset Download:** [https://www.kaggle.com/datasets/liorabergel/hhd-age](https://www.kaggle.com/datasets/liorabergel/hhd-age)

This study uses the **HHD dataset** (Rabaev et al., 2020), which contains 885 handwriting samples from writers aged 8 to 60.

> I. Rabaev, B. K. Barakat, A. Churkin, and J. El-Sana, "The HHD dataset," in *Proc. 16th Int. Conf. Front. Handwriting Recognit. (ICFHR)*, 2020.

The dataset was originally hosted at `https://www.cs.bgu.ac.il/~berat/data/hhd_dataset.zip` (no longer available). It is now publicly available on Kaggle: **https://www.kaggle.com/datasets/liorabergel/hhd-age**

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

**TensorFlow environment** (CNN, ViT, and Grad-CAM experiments):
```bash
pip install -r requirements_tf.txt
```

**PyTorch environment** (DiT experiments):
```bash
pip install -r requirements_torch.txt
```

**Note:** The ViT experiments (`04_ViT`, `05_ViT_CrossVal`) require setting `TF_USE_LEGACY_KERAS=1` before running, as the `keras_cv_attention_models` library depends on legacy Keras.

## Experiments

### Mapping to Paper

The table below maps each experiment folder to the corresponding paper sections and tables.

| Folder | Paper Reference | Description |
|--------|----------------|-------------|
| `03_CNN_CrossVal` | Table 2 (top) | CNN individual models, 5-fold stratified group CV |
| `05_ViT_CrossVal` | Table 2 (middle) | ViT individual models, 5-fold stratified group CV |
| `07_DiT_CrossVal` | Table 2 (bottom) | DiT individual models, 5-fold stratified group CV |
| `01_CNN_Ensemble` | Table 3 (top) | CNN ensembles on official HHD split |
| `06_DiT` | Table 3 (bottom), Table 4 | DiT ensembles and hybrid ensemble on official HHD split |
| `02_CNN_GradCAM` | Figure 5, Section 4.3 | Grad-CAM visualizations |

`04_ViT` is a development experiment (initial ViT exploration on the official HHD split) and is not reported in the paper.

### Running an Experiment

Each experiment is self-contained in its folder. Run scripts **from the repository root**:

```bash
python <experiment_folder>/<script_name>.py
```

For example:
```bash
python 03_CNN_CrossVal/train_sota_cv.py
```

Results for each experiment are documented in `results.md` within the experiment folder, including the full training configuration and all evaluation metrics.

### Key Hyperparameters

All experiments share a common preprocessing pipeline:
- Image resizing to 800px height (aspect ratio preserved)
- Patch extraction: 400x400 pixels, stride 200
- Pixel normalization to [0, 1]
- Augmentation: rotation (+/-15 degrees), zoom (up to 10%), Gaussian noise, and brightness and/or contrast (varies by experiment)
- Empty patch filtering: intensity threshold = 0.0054

**CNN experiments (`01_CNN_Ensemble`, `03_CNN_CrossVal`):**
- Models: ResNet50, InceptionV3, DenseNet121, InceptionResNetV2, EfficientNetV2M
- Pretrained weights: ImageNet-1K
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 128
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**ViT experiments (`04_ViT`, `05_ViT_CrossVal`):**
- Models: SwinV2-Tiny (256x256), MobileViT-XXS (256x256), ConvNeXtV2-Tiny (224x224), TinyViT-11M (224x224)
- Pretrained weights: ImageNet-1K (all four models; `pretrained="imagenet"` in `keras_cv_attention_models`)
- Regression head: GlobalAveragePooling2D, Dropout (0.5), Dense (1, linear)
- Training: 50 epochs frozen backbone + 10 epochs fine-tuning
- Batch size: 64
- Optimizer: Adam (1e-3 frozen, 1e-4 fine-tuning)

**DiT experiments (`06_DiT`, `07_DiT_CrossVal`):**
- Models: DiT-Base, DiT-Large, DiT-Base (RVL-CDIP), DiT-Large (RVL-CDIP)
- Pretrained weights: IIT-CDIP (Base/Large); RVL-CDIP fine-tuned (RVL-CDIP variants)
- Input: 224x224 via BeitImageProcessor
- Training: 15 epochs frozen backbone + 30 epochs fine-tuning
- Batch size: 4 (`06_DiT`), dynamic with gradient accumulation + FP16 (`07_DiT_CrossVal`)
- Optimizer: AdamW (1e-4 frozen, 1e-5 fine-tuning)

## Results Summary

Results are from 5-fold stratified group cross-validation (Table 2 in the paper), reported as mean ± std.
See each experiment's `results.md` for full metrics and ensemble configurations.

| Model Family | Best Individual Model | MAE (years) | R² |
|-------------|-----------------------|-------------|----|
| CNN | ResNet50 | 5.41 ± 0.78 | 0.10 ± 0.06 |
| ViT | MobileViT-XXS | 6.00 ± 0.76 | 0.08 ± 0.05 |
| DiT | DiT-Large (RVL-CDIP) | 3.47 ± 0.54 | 0.46 ± 0.15 |

Best results on the official HHD split (Table 3 in the paper):

| Configuration | MAE (years) | R² |
|--------------|-------------|----|
| CNN: Best 3 Ensemble (MAE-based) | 2.86 | 0.14 |
| DiT: Best 2 Ensemble (Grid Search) | 2.33 | 0.42 |

## Repository Structure

```
Age_Estimation/
  01_CNN_Ensemble/                         # CNN ensembles with augmentation (Table 3 top)
    train_cnn_ensemble.py                  # Training pipeline (5 CNN backbones + ensembles)
    reproduce_results.py                   # Reproduce results.md (CSV fast path / Zenodo weights)
    results.md                             # Training config + evaluation metrics
  02_CNN_GradCAM/                          # Grad-CAM visualizations (Figure 5)
    visualize_gradcam.py
    results.md
  03_CNN_CrossVal/                         # CNN 5-fold stratified group CV (Table 2 top)
    train_sota_cv.py
    results.md
  04_ViT/                                  # ViT initial exploration (development, not in paper)
    train_vit.py
    results.md
  05_ViT_CrossVal/                         # ViT 5-fold stratified group CV (Table 2 middle)
    train_vit_cv.py
    results.md
  06_DiT/                                  # DiT experiments + ensembles (Table 3 bottom, Table 4)
    train_dit.py
    results.md
  07_DiT_CrossVal/                         # DiT 5-fold stratified group CV (Table 2 bottom)
    train_dit_cv.py
    results.md
  data/                                    # Downloaded automatically via kagglehub
  download_dataset.py                      # Dataset download utility
  requirements_tf.txt                      # TensorFlow dependencies (CNN, ViT, Grad-CAM)
  requirements_torch.txt                   # PyTorch dependencies (DiT)
  LICENSE
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
