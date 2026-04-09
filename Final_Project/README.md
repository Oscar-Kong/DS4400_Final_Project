# Decoding Musical Time

This is our DS 4400 final project. The goal is simple: use the UCI YearPredictionMSD dataset to predict a song's release year from its audio summary features.

The final version of the project is regression-only. We tried to keep the pipeline straightforward and easy to defend:

- target: release year
- dataset: UCI YearPredictionMSD
- models: linear regression, ridge regression, random forest, hist gradient boosting
- evaluation: random split, blocked holdout, and a future extrapolation stress test

## Setup

Run everything from `Final_Project/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the experiment

Full run:

```powershell
python run_experiment.py
```

If you just want a faster check that the pipeline still works:

```powershell
python run_experiment.py --sample-fraction 0.05
```

## What gets written

The script writes generated results to `outputs/`.

The files we use the most are:

- `outputs/metrics.csv` for the main regression results
- `outputs/baseline_metrics.csv` for the naive baselines
- `outputs/RESULTS_SUMMARY.md` for a quick readable summary of the last run
- `outputs/split_metadata.json` for split sizes and year ranges
- `outputs/figures/` for EDA plots, residual plots, model comparisons, and decade-binned heatmaps

## Notes

- `outputs/` and `data_cache/` are ignored by git because they are generated.
- The decade confusion heatmaps come from binning regression predictions into decades after the fact. They are there to help interpret the regression results.
