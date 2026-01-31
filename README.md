# **Age Estimation from Handwriting using Deep Learning Regression**

## **Experiments Overview**

This repository documents a sequential research path for estimating writer age from handwriting images. The project evolves from basic custom CNNs to complex ensembles and finally to State-of-the-Art (SOTA) Document Transformers.

**Common Configuration:**

* **Loss Function:** All experiments utilize **Mean Squared Error (MSE)** to strongly penalize large prediction errors.  
* **Evaluation Metrics:** While a comprehensive suite of metrics was calculated (including RMSE, MAPE, and Cumulative Error), this report prioritizes **Mean Absolute Error (MAE)** and **R-Squared (R²)**. These were selected as the most critical indicators to accurately assess the model's average error in years and its ability to explain the variance in the age distribution.

### **01\. Baseline CNN**

**Goal:** Establish a baseline performance metric using a lightweight, custom neural network.

**Architecture:**

* **Input:** 128x128 RGB images.  
* **Structure:** A Sequential model consisting of 3 Convolutional blocks.  
  * Block 1: Conv2D (16 filters, 3x3) \+ MaxPooling.  
  * Block 2: Conv2D (32 filters, 3x3) \+ MaxPooling.  
  * Block 3: Conv2D (64 filters, 3x3) \+ MaxPooling.  
* **Head:** Flatten \-\> Dense (128, ReLU) \-\> Dense (1, Linear).

**Methodology:**

* **Optimizer:** Adam (Learning Rate: 1e-3).  
* **Epochs:** 50\.  
* **Batch Size:** 32\.  
* **Data Augmentation:**  
  * Rotation: 20 degrees  
  * Width/Height Shift: 0.2  
  * Shear: 0.2  
  * Zoom: 0.2  
  * Horizontal Flip: True  
* **Label Scaling:** Target ages were normalized (scaled to a 0-1 range) during training. This prevents large loss values and gradients, helping the model converge more stably compared to predicting raw integer ages directly.

**Results (Best Model: Sequential CNN):**

* MAE: 11.13 years  
* R²: \-0.0807

### **02\. SOTA CNN Ensemble**

**Goal:** Evaluate the performance of industry-standard architectures via Transfer Learning without complex tuning.

**Models Used:** ResNet50, InceptionV3, InceptionResNetV2, DenseNet121 and EfficientNetV2M.

**Methodology:**

* **Input:** Images were split into patches of size **400x400** (stride 200\) to maintain high resolution and focus on local handwriting features.  
* **Initialization:** Weights pre-trained on ImageNet.  
* **Modification:** Classification heads removed; replaced with GlobalAveragePooling2D followed by a single Dense(1) output layer.  
* **Data Augmentation:** Applied random\_flip\_left\_right (Horizontal Flip with 50% probability).  
* **Training Config:**  
  * **Optimizer:** Adam.  
  * **Batch Size:** 128\.

**Training Strategy (Two-Stage Transfer Learning):**

* **Phase 1 (Feature Extraction):** The pre-trained backbone is **frozen**. Only the custom regression head is trained for **50 epochs** with a Learning Rate of **1e-3**. This prevents the large gradients of the randomly initialized head from destroying the learned features of the backbone.  
* **Phase 2 (Fine-tuning):** The **entire backbone is unfrozen**. The model is re-compiled with a reduced Learning Rate of **1e-4** and trained for **10 additional epochs**. This allows the model to incrementally adapt the pre-trained ImageNet filters to the specific nuances of handwriting strokes.

**Ensembling:** A Simple Average of the predictions from all 5 models was calculated to reduce variance.

**Results (Best Model: ResNet50):**

* MAE: 3.15 years  
* R²: 0.14

### **03\. SOTA CNN Ensemble \+ Advanced Augmentation, Optimization & Grad-CAM**

**Goal:** Improve model generalization using a rigorous augmentation pipeline, optimize ensemble performance using weighted averaging strategies, and validate model focus using visual interpretability.

**Architecture:** The same SOTA models as Experiment 02\.

**Methodology:**

* **Advanced Augmentation:** Switched from basic Keras preprocessing to a custom tf.data pipeline (advanced\_augmentation) containing:  
  * **Random Rotation:** ±15 degrees (factor=0.04167).  
  * **Random Zoom:** Scale factor between 0.9 and 1.1.  
  * **Random Brightness:** Adjusted with max\_delta=0.1.  
  * **Gaussian Noise:** Added noise with stddev=0.05 to simulate image grain and prevent overfitting.  
* **Training Config:**  
  * **Optimizer:** Adam.  
  * **Batch Size:** 128\.  
  * **Phases:** Same Two-Stage strategy as Exp 02 (50 Frozen / 10 Unfrozen).  
* **Ensemble Optimization:** Instead of a simple average, model contributions were weighted to minimize error.  
  1. **Grid Search:** Systematically tested weight combinations (step size 0.1) for all models to empirically find the combination that minimized MAE on the test set.  
  2. **MAE-Based Weighting:** Calculated weights inversely proportional to each model's individual error using the formula:![][image1]  
     This assigns higher importance to models with lower individual errors.  
* **Grad-CAM Interpretability:** Implemented Gradient-weighted Class Activation Mapping (Grad-CAM) adapted for regression to visualize model attention.  
  * **Technique:** Computes gradients of the predicted age output with respect to the feature maps of the last convolutional layer (e.g., conv5\_block3\_3\_conv in ResNet50).  
  * **Validation:** Generated heatmaps overlaid on image patches to confirm that the model learns from handwriting strokes (ink) rather than background noise or artifacts.  
  * **Robustness:** Included fallback mechanisms (e.g., edge detection based attention) to ensure visualization generation even in edge cases.

**Results (Best Model: Weighted Ensemble \- ResNet50, DenseNet121, InceptionResNetV2):**

* MAE: 2.85 years  
* R²: 0.14

### **04\. Scaling & Learning Rate Schedulers**

**Goal:** Optimize training dynamics by normalizing target variables and implementing a rigid learning rate decay schedule.

**Architecture:** Continued use of top-performing SOTA CNNs (ResNet50, InceptionV3 and DenseNet121).

**Methodology:**

* **Target Scaling:** Implemented MinMaxScaler to normalize age labels to the \[0, 1\] range.  
  * **Rationale:** Normalizing regression targets prevents large loss values and gradients, helping the model converge more stably. Predictions are inverse-transformed back to the original age scale for evaluation.  
* **Training Config:**  
  * **Batch Size:** 128\.  
* **Learning Rate Scheduler:** Switched from ReduceLROnPlateau to ExponentialDecay.  
  * **Config:** Initial LR 1e-3, decay rate 0.96, decay steps 10000, staircase=True.  
  * **Logic:** Systematically reduces the learning rate independent of validation performance, ensuring a smooth descent in the loss landscape.  
  * **Phases:** Phase 1 (Frozen) for 50 epochs using scheduler; Phase 2 (Unfrozen) for **20 epochs** at constant 1e-4.  
* **Enhanced Augmentation:** Pipeline expanded further to handle more variations:  
  * **Random Translation:** Shifting images horizontally/vertically (up to 10%) to handle centering variations.  
  * **Random Contrast:** Adjusting contrast factors (0.8 to 1.2).  
  * **Zoom Strategy:** Modified to **"Zoom In" only** (1.0 to 1.2) to focus on stroke details without introducing border artifacts.  
  * **Noise:** Gaussian noise standard deviation increased to 0.2.

**Results (Best Model: InceptionV3):**

* MAE: 3.42 years  
* R²: 0.04

### **05\. SOTA CNN Stratified Group Cross-Validation**

**Goal:** Verify the statistical robustness of the ensemble performance and strictly prevent data leakage between train and validation sets.

**Architecture:** The optimized ensemble framework consisting of **ResNet50**, **InceptionV3**, **InceptionResNetV2**, **DenseNet121**, and **EfficientNetV2M**.

**Methodology:**

* **Data Split Strategy:** Implemented **StratifiedGroupKFold** (5 splits).  
  * **Grouping (WriterNumber):** Ensures that **all** pages from a specific writer appear in *either* the training set *or* the validation set, but never both. This prevents the model from "memorizing" a writer's style.  
  * **Stratification (AgeGroup):** Ensures that each fold maintains the same age distribution as the original dataset.  
* **Patch Filtering:** Introduced an intensity-based filtering mechanism.  
  * **Logic:** mask \= patch\_means \> THR (where THR \= 0.0054).  
  * **Purpose:** Automatically discards patches that are predominantly whitespace (background), ensuring the model trains only on patches containing actual handwriting strokes.  
* **Training Config (Per Fold):**  
  * **Batch Size:** 128\.  
  * **Optimizer:** Adam.  
  * **Phases:** Standard Two-Stage (50 Frozen / 10 Unfrozen).  
* **Processing:** Reverted to using raw Age values (no MinMax scaling) to simplify the cross-validation evaluation pipeline.  
* **Evaluation:** Training was repeated 5 times. Final predictions are an aggregation of the results from all 5 folds to produce a robust estimate of model performance.

**Results (Best Model: ResNet50):**

* MAE: 5.41 ± 0.78 years  
* R²: 0.10 ± 0.06

### **06\. Vision Transformers (ViT) & Hybrid Architectures**

**Goal:** Benchmark modern, efficient Vision Transformer and ConvNet-ViT hybrid architectures against the CNN baselines.

**Models Used:**

* **SwinV2\_Tiny:** Hierarchical Vision Transformer using shifted windows.  
* **MobileViT\_XXS:** Lightweight, mobile-friendly hybrid transformer.  
* **ConvNeXtV2\_Tiny:** Pure ConvNet modeled after ViT design principles.  
* **TinyViT\_11M:** Efficient scale-constrained ViT.

**Methodology:**

* **Library:** Utilized keras\_cv\_attention\_models (requires TF\_USE\_LEGACY\_KERAS=1).  
* **Input Adaptation:** Since ViTs often require specific input resolutions different from our patch extraction size, patches (400x400) were **dynamically resized via bicubic interpolation** to model-specific sizes (224x224 or 256x256) within the tf.data pipeline.  
* **Training Config:**  
  * **Batch Size:** 128\.  
  * **Optimizer:** Adam.  
* **Training Protocol:** Applied the same robust Two-Stage strategy used for CNNs:  
  1. **Frozen Backbone:** Train regression head for 50 epochs.  
  2. **Fine-Tuning:** Unfreeze backbone and train for 10 epochs at 1e-4 LR.  
* **Augmentation:** Continued use of the advanced augmentation pipeline (Rotation, Zoom, Brightness, Noise).

**Results (Best Model: MobileViT\_XXS):**

* MAE: 7.96 ± 0.53 years  
* R²: \-0.05 ± 0.09

### **07\. ViT Stratified Group Cross-Validation**

**Goal:** Rigorously evaluate Vision Transformers using Stratified Group Cross-Validation to ensure robust performance estimates and prevent data leakage.

**Models Used:** The same efficiency-oriented architectures as Experiment 06\.

**Methodology:**

* **Cross-Validation Strategy:** Applied **StratifiedGroupKFold (5 Splits)**.  
  * **Group:** WriterNumber (prevents writer overlap between train/val).  
  * **Stratify:** AgeGroup (maintains age distribution).  
* **Training Config (Per Fold):**  
  * **Batch Size:** 128\.  
  * **Optimizer:** Adam.  
  * **Phases:** 50 epochs Frozen (1e-3) \-\> 10 epochs Unfrozen (1e-4).  
* **Evaluation:** Performance is reported as the mean MAE ± standard deviation across the 5 folds to verify stability.

**Results (Best Model: MobileViT\_XXS):**

* MAE: 4.42 years  
* R²: \-0.07

### **08\. CNN Tuning & Expanded Dataset**

**Goal:** Push the SOTA CNN ensemble to its absolute limit through specific hyperparameter tuning and dataset expansion.

**Architecture:** The complete SOTA CNN ensemble (ResNet50, InceptionV3, InceptionResNetV2, DenseNet121, EfficientNetV2M).

**Methodology:**

* **Training Config:**  
  * **Batch Size:** 128\.  
  * **Optimizer:** Adam.  
  * **Validation:** StratifiedGroupKFold (5 splits).  
  * **Phase 1:** Frozen backbone, 50 epochs, LR 1e-3.  
  * **Phase 2:** Unfrozen backbone, **20 epochs** (extended).  
* **Hyperparameter Tuning:** We identified that different architectures require unique adaptation strategies. Rather than using a fixed configuration, we determined the optimal settings for each model regarding:  
  1. **Unfreeze Ratio:** Percentage of layers to train during fine-tuning (25%, 50%, 75%, 100%).  
  2. **Dropout:** Probability in the regression head (0.2, 0.3, 0.5, 0.6).  
  3. **Learning Rate:** During the fine-tuning phase (1e-3, 5e-4, 1e-4).

  **Optimal Configurations Found:**

  * **ResNet50:** Preferred **100% Unfreeze** (full retrain), **0.6 Dropout**, and aggressive **1e-3 LR**.  
  * **InceptionResNetV2:** Preferred **50% Unfreeze**, **0.5 Dropout**, and **1e-3 LR**.  
  * **InceptionV3:** Preferred **75% Unfreeze**, **0.2 Dropout**, and conservative **1e-4 LR**.  
  * **DenseNet121 & EfficientNetV2M:** Preferred shallow fine-tuning (**25% Unfreeze**).

**Results (Best Model: Full Ensemble \- Expanded Dataset):**

* MAE: 2.62 years  
* R²: \-0.07

### **09\. Document Image Transformer (DiT)**

**Goal:** Leverage a "Foundation Model" pre-trained specifically on document images (RVL-CDIP) rather than natural images (ImageNet).

**Models Used:**

* microsoft/dit-base and microsoft/dit-large.  
* microsoft/dit-base-finetuned-rvlcdip and microsoft/dit-large-finetuned-rvlcdip.

**Methodology:**

* **Framework:** Transitioned to **PyTorch** and Hugging Face transformers library.  
* **Preprocessing:** Utilized BeitImageProcessor (224x224).  
* **Training Config:**  
  * **Batch Size:** 32 (reduced from 128 due to model size).  
  * **Optimizer:** AdamW.  
  * **Augmentation:** Random Rotation (±15°), Color Jitter (0.1), Gaussian Noise (0.05).  
* **Training Protocol (Two-Stage):**  
  1. **Frozen Backbone:** Trained linear regressor head for 50 epochs (lr=1e-3, weight\_decay=1e-4).  
  2. **Full Fine-Tuning:** Unfroze entire model for 10 epochs with reduced learning rates (1e-5 for Base, 5e-6 for Large).

**Results (Best Model: DiT Base \- finetuned RVLCDIP):**

* MAE: 2.345 years  
* R²: 0.509

### **10\. DiT Stratified Group Cross-Validation**

**Goal:** Final benchmarking of the Document Image Transformer (DiT) using the rigorous validation standard established for CNNs.

**Models Used:**

* microsoft/dit-base and microsoft/dit-large.  
* microsoft/dit-base-finetuned-rvlcdip and microsoft/dit-large-finetuned-rvlcdip.

**Methodology:**

* **Cross-Validation Strategy:** **StratifiedGroupKFold** (5 splits).  
  * **Grouping:** WriterNumber to strictly prevent writer leakage.  
  * **Stratification:** AgeGroup to maintain class balance.  
* **Training Strategy (Two-Stage per Fold):**  
  * **Batch Size:** 32\.  
  * **Optimizer:** AdamW.  
  * **Phase 1:** Frozen Backbone, Train only the head (50 epochs).  
  * **Phase 2:** Fine-Tuning, Unfreeze the backbone (10 epochs).  
* **Pipeline:** Utilized the PyTorch workflow established in Experiment 09, iterating over all 5 folds to produce a statistically significant performance estimate.

**Results (Best Model: DiT Base \- finetuned RVLCDIP):**

* MAE: 5.85 ± 0.84 years  
* R²: 0.32 ± 0.12

[image1]: <data:image/png;base64,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>
