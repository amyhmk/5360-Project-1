# 5360-Project-1
Nadeem Ayoub and Aaron Yocom Group 8 for 5360 Group Project 1

# Credit Card Fraud Analysis

## Purpose

This project explores a 10,000-transaction credit-card dataset and compares fraud-classification models. The notebook covers data checks, exploratory analysis, preprocessing, cross-validation, model tuning, threshold selection, SHAP explanations, interaction analysis, and feature-ablation checks.

The target is `is_fraud`. The identifier `transaction_id` is excluded from model features. The dataset contains 151 fraud cases (1.51%), so precision-recall metrics are more informative than accuracy alone.

## Project files

- `fraud_project.ipynb` - analysis and modeling workflow
- `credit_card_fraud_10k.csv` - input data; keep it beside the notebook
- `fraud_detection_presentation.pptx` - five-slide presentation

## Setup

The notebook was validated with Python 3.13.14. From this folder, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel numpy==2.3.4 pandas==3.0.5 scikit-learn==1.9.1 matplotlib==3.10.7 seaborn==0.13.2 optuna==5.0.0 shap==0.54.1
```

## Run the notebook

1. Start Jupyter from this project folder:

   ```powershell
   jupyter lab
   ```

2. Open `fraud_project.ipynb` and select the project's Python 3.13 environment as its kernel.
3. Restart the kernel and run all cells from top to bottom. The CSV path is resolved relative to the notebook.

The workflow uses a fixed random seed and a stratified train/test split. Optuna tuning and cross-validation can take several minutes; Step 7 also reruns cross-validation for interaction and feature-ablation analyses.

## Results at a glance

The selected model is a tuned `HistGradientBoostingClassifier`. Its reported training 5-fold CV PR AUC is 0.9979. The final threshold is 0.9989, selected on an internal validation split to be the highest threshold meeting at least 90% recall there. The final thresholded test result was precision 1.00, recall 0.833, F1 0.909, and accuracy 0.9975 (25 frauds caught, 5 missed, and no false positives).

## Important interpretation caveats

- Fraud is rare (1.51%), and the near-perfect CV scores are unusually high for real fraud data. Check how labels and features were generated and verify every feature would be available at decision time.
- Optuna selected parameters using the same 5-fold CV folds used to report tuned-model CV metrics; this can make the reported CV score optimistic.
- The test set was inspected for split summaries, model evaluation, and SHAP examples in different steps. It was not used for fitting or threshold/model selection, but it is no longer an untouched one-time holdout.
- The high-recall threshold was unstable: out-of-fold recall at the selected threshold was about 70%. Use a fresh holdout or later-period data, ideally with customer/time separation, before relying on performance estimates.
- Feature ablation is a useful diagnostic, not proof that data leakage is absent. The available columns do not include a customer/account ID or full transaction timestamp, so a group- or time-based split requires additional source data.
