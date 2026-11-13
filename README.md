# Credit Card Fraud Analysis

## Purpose

This project explores a 10,000-transaction credit-card dataset and compares fraud-classification models. The notebook covers data checks, exploratory analysis, preprocessing, cross-validation, model tuning, threshold selection, SHAP explanations, interaction analysis, and feature-ablation checks.

The target is `is_fraud`. The identifier `transaction_id` is excluded from model features. The dataset contains 151 fraud cases (1.51%), so precision-recall metrics are more informative than accuracy alone.

## Project files

- `fraud_project.ipynb` - analysis and modeling workflow
- `credit_card_fraud_10k.csv` - input data; keep it beside the notebook
- `fraud_detection_presentation_final.pptx` - five-slide presentation

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

The workflow uses a fixed random seed and a stratified train/test split. Step 6 runs nested cross-validation with Optuna searches inside the outer training folds; this and Step 7's interaction and feature-ablation CV can take several minutes.

## Results at a glance

The selected model is a tuned `HistGradientBoostingClassifier`. In five-fold nested training CV, Optuna tuning and F2 threshold calibration are repeated inside each outer training fold. The mean outer-fold PR AUC was 0.9885; pooled outer-fold precision was 84.9%, recall 97.5%, and F2 94.7%, with 3 missed frauds and 21 false positives across 121 training frauds.

For the final fit, the F2-maximizing threshold from training-only out-of-fold scores was 0.8090. The associated threshold-calibration scores were precision 99.1%, recall 95.9%, and F2 96.5%; these are not an unbiased estimate because the final hyperparameters were tuned on the same training partition. At the fixed threshold, the exploratory test check caught 30/30 frauds with no false positives (all reported test metrics were 100%).

## Important interpretation caveats

- Fraud is rare (1.51%), and the strong CV scores are unusual for real fraud data. Check how labels and features were generated and verify every feature would be available at decision time.
- Nested outer CV reduces the optimism from tuning and evaluation on the same folds, but the final threshold was calibrated on OOF scores after tuning the final hyperparameters on all training data. Treat its OOF metrics as calibration results, not an unbiased estimate.
- The test set was inspected in earlier steps, and the decision to focus on F2 followed seeing the earlier test misses. The updated test result is post hoc and is not an independent final estimate, even though this run did not use test data to choose the model or threshold.
- Use a fresh holdout or later-period data, ideally with customer/time separation, before relying on performance estimates or claiming zero missed fraud.
- Feature ablation is a useful diagnostic, not proof that data leakage is absent. The available columns do not include a customer/account ID or full transaction timestamp, so a group- or time-based split requires additional source data.
