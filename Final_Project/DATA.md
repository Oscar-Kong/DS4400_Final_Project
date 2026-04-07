# Data: YearPredictionMSD

## Source

- **Dataset:** Year Prediction Million Song Dataset (MSD-derived tabular timbre features).
- **Download:** UCI archive [YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD); code caches `YearPredictionMSD.txt` under `data_cache/` (see [`src/data.py`](src/data.py) for the zip URL).

## License and use

Research-friendly public benchmark; suitable for ML coursework. This project does **not** use Spotify API data as the primary training source (policy uncertainty); MSD/UCI-style data is used instead.

## Schema

- **Samples:** ~515,345 tracks in the official file (row count may drop slightly after duplicate removal).
- **Target:** Release year, typically **1922–2011** per UCI documentation (confirm in `outputs/data_meta.json` after a run).
- **Features:** **90** numeric attributes (`f00`–`f89` in code): timbre averages and covariances as distributed by UCI.

## Train / holdout policy (code)

1. **Random split:** 70% train, 15% validation, 15% test (`random_state=42`). Used for hyperparameter tuning, then scaler + models refit on train+validation, metrics on held-out test.
2. **Time-aware split:** **Train** if `year <= 2000`, **test** if `year > 2000`. Tuning uses an inner temporal slice: fit scaler and models on `year <= 1995`, validate on `1996 <= year <= 2000`; final models use scaler fit on **all** `year <= 2000` and evaluation on `year > 2000`.

## Preprocessing

- Drop rows with missing target or features.
- Optional subsampling for quick runs: `--sample-fraction` (see `run_experiment.py`).
- **StandardScaler** (z-score): always **fit on training data only** for each split regime, then applied to validation/test.
