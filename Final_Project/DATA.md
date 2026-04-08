# Data: YearPredictionMSD

## Source

- Dataset: Year Prediction Million Song Dataset (MSD-derived tabular timbre features).
- Download: UCI archive [YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). The code caches `YearPredictionMSD.txt` under `data_cache/`.

## License and use

This is a research-friendly public benchmark and is appropriate for coursework. The project does not use Spotify API data as the main training source.

## Schema

- Samples: about 515,345 tracks in the official file before cleaning.
- Target: release year, typically 1922-2011.
- Features: 90 numeric attributes (`f00`-`f89`) distributed with the UCI benchmark.

## Preprocessing

- Drop rows with missing target or missing features.
- Deduplicate exact duplicate rows only. The pipeline does not drop rows just because the feature vector repeats with a different target year.
- Optional subsampling for quick runs via `--sample-fraction`.
- Standardize features with `StandardScaler` for linear models only, fitting the scaler on the appropriate training partition for each split.

## Split policy used in code

1. Random split
   Train / validation / test is approximately 70% / 15% / 15% with `random_state=42`.
2. Blocked holdout
   Validation years are 1971-1980, test years are 1981-1995, and the final training pool uses every year outside the test window.
3. Future extrapolation stress test
   Train on `year <= 2000`, test on `year > 2000`, and tune on `year <= 1995` versus `1996-2000`.

## Output artifacts

- `outputs/data_meta.json`: overall dataset size and sampled run metadata.
- `outputs/split_metadata.json`: split sizes and year ranges.
- `outputs/baseline_metrics.csv`: naive benchmark metrics for each split.
- `outputs/classification_support.csv`: whether each split has full decade-label support for classification.
