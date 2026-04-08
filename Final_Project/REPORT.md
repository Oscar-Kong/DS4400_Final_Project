# Decoding Musical Time: Methods, Evaluation, and Reporting Notes

This document explains the final experiment design used by `run_experiment.py`. Report-ready numbers are written to `outputs/metrics.csv`, `outputs/baseline_metrics.csv`, and `outputs/RESULTS_SUMMARY.md`.

## Main story

- Primary task: regression on release year.
- Secondary task: decade classification for interpretation and rubric support.
- Primary benchmark to report: random split results.
- Stronger leakage check to report: blocked holdout results.
- Stress test to report carefully: future extrapolation results. Treat this as a hard generalization check, not the main headline table.

## Regression models

The project compares four regression models:

| Model | Purpose |
| --- | --- |
| `linear_regression` | Simple baseline on scaled features |
| `ridge_regression` | Regularized linear model with alpha selected on validation data |
| `random_forest` | Nonlinear ensemble with feature importance estimates |
| `hist_gradient_boosting` | Strong tree boosting baseline for tabular data |

The old sklearn `GradientBoostingRegressor` path has been removed because it did not materially improve the story or the runtime.

## Evaluation design

### Random split

- About 70% train, 15% validation, 15% test.
- Hyperparameters are selected using the explicit validation split.
- Final models are fit on train plus validation and scored on the held-out test set.

### Blocked holdout

- Validation window: 1971-1980
- Test window: 1981-1995
- Final training pool: all years except the test window

This split is useful because it removes the exact test years from training without forcing the model to extrapolate only beyond the maximum observed train year.

### Future extrapolation stress test

- Training pool: years up to 2000
- Tuning split: years up to 1995 versus 1996-2000
- Test pool: years after 2000

This is intentionally difficult. It should be discussed as a stress test because it asks the model to predict later music from earlier music only.

## Baselines

Every regression split is compared against naive baselines:

- training-mean predictor
- training-median predictor
- last-seen-year predictor for the future extrapolation split

A learned model should beat these baselines before the result is presented as meaningful.

## Classification notes

- Decade classification is run only on splits where every test decade label appears in training.
- Use confusion matrices and ROC curves from those supported splits in the final report.
- Do not use the future extrapolation classification result as a headline comparison if the test set contains unseen decade labels.

## Recommended reporting

- Put the random split regression table first.
- Put the blocked holdout regression table second.
- Use the future extrapolation table as a limitations or robustness subsection.
- Include baseline metrics alongside the learned models.
- Use feature importance, ridge coefficients, residual plots, and decade confusion patterns to interpret what the models are learning.
