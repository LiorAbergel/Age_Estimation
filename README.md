# Age Estimation from Handwriting using Deep Learning

This project explores age estimation from Hebrew handwriting samples using deep learning models. We conducted a series of experiments on both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), gradually enhancing model performance with dataset expansion, cross-validation, Grad-CAM analysis, and hyperparameter tuning.

## 📂 Project Structure

### 🔹 `Age_Recognition_Handwriting_Regression_V2.5 (1).ipynb`
Initial experiment using CNNs like ResNet50 and InceptionV3 on the HHD dataset. This served as a baseline for performance evaluation.

### 🔹 `Age_Recognition_Handwriting_Regression_V2.5_Expanded.ipynb`
Introduces an expanded version of the dataset with 166 new handwriting samples, increasing diversity across age groups. This experiment evaluates the impact of data diversity on model accuracy.

### 🔹 `Age_Recognition_Handwriting_Regression_V2_5_Grad_CAM (3).ipynb`
Implements Grad-CAM visualizations for interpretability. Provides insight into which regions of handwriting images influenced the model’s age predictions.

### 🔹 `Age_Recognition_Handwriting_Regression_V4_Cross_Validation (2).ipynb`
Applies stratified k-fold cross-validation to better assess model robustness across the dataset. Includes evaluation metrics such as MAE, RMSE, R², and error distributions.

### 🔹 `Age_Recognition_Handwriting_Regression_V5_ViT (3).ipynb`
First implementation of Vision Transformers (ViTs) such as SwinV2, MobileViT, and BEiT for age regression, without cross-validation.

### 🔹 `Age_Recognition_Handwriting_Regression_V6_ViT_No-CV.ipynb`
Additional experimentation with more ViT variants and configurations, focusing on training stability and error metrics.

### 🔹 `Age_Recognition_Handwriting_Regression_V7_HP_Tuning.ipynb`
Performs hyperparameter tuning (dropout, learning rate, unfreeze ratio) on CNNs using Keras Tuner to optimize performance.

## 📊 Dataset
- **Base Dataset**: HHD (Hebrew Handwriting Dataset), 882 samples
- **Expanded Dataset**: Additional 166 samples for a total of 1,048
- **Age Range**: 8–91 years

## 🧠 Models Explored
- **CNNs**: ResNet50, InceptionV3, InceptionResNetV2, DenseNet121, EfficientNetV2M
- **ViTs**: SwinV2, MobileViT, BEiT, TinyViT, FasterViT

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- MAPE (%)
- Accuracy within ±2, ±5, ±10 years
- Max, Median, and Min Error

## 🔍 Interpretability
Grad-CAM was used to visualize attention maps and highlight areas the model focused on during prediction, especially useful in identifying potential model biases.

## 📌 Citation
If you use this work, please cite our IGS 2025 poster presentation or our forthcoming submission to *Pattern Recognition Letters*.
