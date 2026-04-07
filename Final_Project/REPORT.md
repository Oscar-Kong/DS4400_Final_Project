# Decoding Musical Time — Methods, Ethics, and Limitations

This document supports the DS 4400 final project. **Numerical results** are produced by `run_experiment.py` and written to [`outputs/metrics.csv`](outputs/metrics.csv) and [`outputs/RESULTS_SUMMARY.md`](outputs/RESULTS_SUMMARY.md) after you run the pipeline.

## Data and ethics

- **Primary data:** UCI **YearPredictionMSD**: 90 numeric timbre-related features and release year. See [`DATA.md`](DATA.md) for URLs, schema, and split policy.
- **Why not Spotify as core training data:** Platform terms create uncertainty for redistributing or training on API-sourced content; a public benchmark avoids that compliance risk while staying aligned with the research question (year from audio descriptors).

## Task and preprocessing

- **Task:** Regression — predict **release year** from the feature vector.
- **Cleaning:** Drop rows with missing labels or features; drop duplicate feature rows; optional `--sample-fraction` for quick runs.
- **Scaling:** `StandardScaler` fit **only** on the training portion appropriate to each experiment (random: train+val before test; time-aware: all years ≤ 2000 before predicting years > 2000).

## Splits and leakage

1. **Random split:** ~70% / ~15% / ~15% train / validation / test (`random_state=42`). Tree hyperparameters are tuned on a subsample of the training split; **Ridge** uses internal 5-fold CV on train+val. Final models are fit on train+val combined; metrics are on the held-out test set.
2. **Time-aware split:** Train on **year ≤ 2000**, test on **year > 2000**. Inner tuning uses **year ≤ 1995** vs **1996–2000** so hyperparameter choices are not fit on the final test era. This reduces **temporal leakage** compared to a purely random split: random mixing makes “nearby years” easier to interpolate.

Comparing the two splits shows how much measured performance is an artifact of evaluation design versus generalization to **later** music.

## Models

| Model | Role |
|--------|------|
| **Linear regression** | OLS baseline; interpretable coefficients on scaled features. |
| **RidgeCV** | L2 regularization for correlated timbre features; α chosen by CV. |
| **Random forest** | Nonlinearities and interactions; feature importances for interpretation. |
| **HistGradientBoostingRegressor** | Strong tabular regressor (histogram GBDT); tuned hyperparameters. |
| **GradientBoostingRegressor** | Classic gradient boosting (sklearn); tuned on a subsample for runtime. |

## Evaluation metrics

- **MSE, RMSE, MAE, R²** on held-out data. **MAE / RMSE in years** are the most interpretable headlines.
- **Post-hoc decade analysis:** True and predicted years are binned to decades; we report **decade accuracy** and save confusion-style tables for **HGBR** predictions (descriptive only, not a trained classifier).

## Artifacts (after running the code)

- `outputs/metrics.csv` — all models × both splits.
- `outputs/figures/residuals_*.png` — residuals for Ridge and HGBR.
- `outputs/feature_importance_rf_*.csv`, `outputs/ridge_coef_*.csv` — interpretability.
- `outputs/decade_confusion_*_hgbr.csv` — post-hoc decade confusion tables.

## Limitations

- **Overlap between adjacent eras** in feature space makes perfect separation unrealistic; errors should be interpreted in that light.
- **MSD features** are summaries, not raw audio; conclusions are about **this feature set**, not all possible descriptors.
- **Time-aware split** is one plausible policy; different cutoffs or rolling windows would yield different numbers.
- **Hyperparameter search** uses subsamples for tree models to keep runtime manageable; full-grid search on the entire training set could change results slightly.

## How to run

```bash
cd Final_Project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiment.py
# Quick test:
python run_experiment.py --sample-fraction 0.05
```

First run downloads the UCI zip into `data_cache/` (or pass `--data-home` to override the cache directory).
