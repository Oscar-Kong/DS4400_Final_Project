# Data notes - YearPredictionMSD

## Source

We are using the UCI **YearPredictionMSD** benchmark:

[https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD)

It is a tabular dataset built from the Million Song Dataset. We are not working with raw audio files here.

When `run_experiment.py` runs, `src/data.py` downloads the archive if needed and stores it under `data_cache/`.

## Why we used this dataset

This was the safest choice for the class project.

- it is public
- it is stable
- it is already set up for supervised learning
- it avoids the mess of API limits or data-use questions that would come with pulling a custom Spotify dataset

## What each row looks like

Each row is one song with:

- one target: release year
- 90 numeric features named `f00` through `f89`

After cleaning, the dataset we used has **515,131 rows**. The year range is **1922 to 2011**.

## Cleaning

The cleaning steps are simple:

- drop rows with missing year values
- drop rows with missing feature values
- drop only exact duplicate rows

We did **not** remove rows just because two songs had similar features. If two songs look similar in the feature space but came out in different years, that is part of the actual problem.

## Preprocessing

We only scale features where it makes sense:

- linear regression and ridge use `StandardScaler`
- tree-based models use the raw numeric features

The scaler is always fit on the training side of a split, never on the test side.

## Splits used in the code

1. **Random split**
   About 70 / 15 / 15 train, validation, and test.

2. **Blocked holdout**
   Validation years are 1971-1980, test years are 1981-1995, and the training pool is everything outside the test block.

3. **Future extrapolation**
   Train on years up to 2000, tune on up to 1995 versus 1996-2000, and test on years after 2000.

## Files worth checking after a run

- `outputs/data_meta.json`
- `outputs/split_metadata.json`
- `outputs/metrics.csv`
- `outputs/baseline_metrics.csv`
- `outputs/RESULTS_SUMMARY.md`
- `outputs/figures/`

If something looks off, rerun `python run_experiment.py` from `Final_Project/` and use the newly generated outputs.
