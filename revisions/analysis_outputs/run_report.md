# Camera-ready revision analysis report

## Inputs

- labels: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\data\NewAgeSplit.csv` (FOUND)
- cnn_test_predictions: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\01_CNN_Ensemble\predictions\test_image_predictions.csv` (FOUND)
- cnn_val_predictions: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\01_CNN_Ensemble\predictions\val_image_predictions.csv` (FOUND)
- vit_test_predictions: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\04_ViT\predictions\test_image_predictions.csv` (MISSING)
- vit_val_predictions: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\04_ViT\predictions\val_image_predictions.csv` (MISSING)
- dit_predictions_directory: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06` (MISSING)
- cv_predictions_directory: `C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\05_ViT_CrossVal\new_results` (FOUND)

## Metadata

```json
{
  "CNN_Best3Grid_val_mae": 3.1426240418375033,
  "CNN_Best3Grid_weights": {
    "DenseNet121": 0.5,
    "InceptionResNetV2": 0.4,
    "ResNet50": 0.1
  },
  "SEED": 42,
  "bootstrap_resamples": 10000,
  "figure6_outputs": [
    "C:\\Users\\liora\\OneDrive\\Documents\\Important\\School\\Bachelors\\Age Estimation Project\\Age_Estimation\\revisions\\analysis_outputs\\figure6_age_group_mae.png",
    "C:\\Users\\liora\\OneDrive\\Documents\\Important\\School\\Bachelors\\Age Estimation Project\\Age_Estimation\\revisions\\analysis_outputs\\figure6_age_group_mae.pdf"
  ]
}
```

## Blocked notes

- BLOCKED: missing official ViT test prediction file: C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\04_ViT\predictions\test_image_predictions.csv
- BLOCKED: missing DiT test prediction file for DiT-Base: C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-base_test_preds.csv
- BLOCKED: missing DiT test prediction file for DiT-Large: C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-large_test_preds.csv
- BLOCKED: missing DiT test prediction file for DiT-Base (RVL-CDIP): C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-base-finetuned-rvlcdip_test_preds.csv
- BLOCKED: missing DiT test prediction file for DiT-Large (RVL-CDIP): C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-large-finetuned-rvlcdip_test_preds.csv
- BLOCKED: missing DiT val prediction file for DiT-Base: C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-base_val_preds.csv
- BLOCKED: missing DiT val prediction file for DiT-Large: C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-large_val_preds.csv
- BLOCKED: missing DiT val prediction file for DiT-Base (RVL-CDIP): C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-base-finetuned-rvlcdip_val_preds.csv
- BLOCKED: missing DiT val prediction file for DiT-Large (RVL-CDIP): C:\Users\liora\OneDrive\Documents\Important\School\Bachelors\Age Estimation Project\Age_Estimation\results\experiment_06\microsoft__dit-large-finetuned-rvlcdip_val_preds.csv
- BLOCKED: cannot compute DiT_Best2Grid; missing DiT official-split val/test predictions
- BLOCKED: age-group table/figure missing DiT_Best2Grid
- BLOCKED: Figure 6 cannot include ViT until official-split ViT predictions are available
- BLOCKED: paired bootstrap DiT_Best2Grid vs best single DiT needs DiT official predictions
- BLOCKED: paired bootstrap DiT_Best2Grid vs CNN_Best3Grid needs both systems
- BLOCKED: paired bootstrap best single DiT vs CNN_Best3Grid needs best single DiT
