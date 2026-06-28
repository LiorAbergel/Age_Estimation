# Camera-ready revision analysis

This folder contains code-only revisions for the camera-ready analysis tasks. It does not edit the paper.

## Run

```powershell
python revisions/analyze_camera_ready.py
```

For a quick smoke test without the 10,000-resample bootstrap:

```powershell
python revisions/analyze_camera_ready.py --skip-bootstrap
```

## Inputs

Defaults are resolved relative to the repository root:

- `data/NewAgeSplit.csv`
- `01_CNN_Ensemble/predictions/test_image_predictions.csv`
- `01_CNN_Ensemble/predictions/val_image_predictions.csv`
- `04_ViT/predictions/test_image_predictions.csv`
- `04_ViT/predictions/val_image_predictions.csv`
- `results/experiment_06/*_{val,test}_preds.csv`
- `05_ViT_CrossVal/new_results/*_fold*_preds.csv`

Missing inputs produce explicit `BLOCKED:` notes in `revisions/analysis_outputs/run_report.md`.

## Outputs

The script writes raw and paper-ready outputs under `revisions/analysis_outputs/`, including:

- `test_distribution.csv`
- `single_models_official_split.csv` and `.tex`
- `bootstrap_cis.csv` and `.tex`
- `paired_bootstrap.csv` and `.tex`
- `error_by_age_group.csv` and `.tex`
- `binned_classification.csv` and `.tex`
- `cv_per_fold.csv` and `.tex`
- `per_sample_errors.csv`
- `figure6_age_group_mae.png`
- `figure6_age_group_mae.pdf`

Figure 6 includes available CNN, DiT, and ViT official-split systems. ViT is included automatically once `04_ViT/predictions/test_image_predictions.csv` exists or a custom path is passed with `--vit-test-preds`.
