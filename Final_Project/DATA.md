# Data writeup — YearPredictionMSD

## Where I got it

I’m using the **Year Prediction Million Song Dataset** version that UCI hosts — it’s basically tabular features derived from the Million Song Dataset, not raw audio files. Link: [YearPredictionMSD on UCI](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD).

When I run the project, `src/data.py` downloads the zip once (if I don’t already have it) and unpacks `YearPredictionMSD.txt` into `data_cache/` next to the project.

## Why I didn’t use Spotify as my main dataset

I didn’t want to risk terms-of-use / redistribution issues with API data for a class project. This UCI bundle is public and already packaged for ML homework, so it was an easier fit.

## What’s in each row

- There are on the order of **~515k** tracks in the official file before I clean it (exact count lands in `outputs/data_meta.json` after a run).
- **Label:** release year (the homework problem is to predict this). Years are mostly in the **1922–2011** range for this benchmark.
- **Features:** **90** numeric columns. In my code they’re named `f00` … `f89`. They’re timbre-related summaries that came with the dataset — I’m not engineering new audio features from scratch.

## What I do to clean it

- Drop rows if the year or any feature is missing.  
- Drop **exact duplicate** feature rows (same 90 numbers). I’m **not** dropping rows just because two different years share similar feature values — that would be sketchy.  
- If I’m debugging or my laptop is struggling, I sometimes run with `--sample-fraction` in `run_experiment.py` (e.g. 0.05) just to get faster feedback; the writeup for the final should use the full data unless the professor says otherwise.  
- `StandardScaler` (z-scores) is only for the **linear / ridge** models, and I always fit the scaler on **training data for that split** — I never fit it on the test set.

## Splits my code actually uses

1. **Random:** ~70 / 15 / 15 train / val / test, `random_state=42`.  
2. **Blocked holdout:** val = **1971–1980**, test = **1981–1995**, train = everything else except the test window.  
3. **Future stress test:** train **≤ 2000**, tune inner slice **≤ 1995** vs **1996–2000**, test **> 2000**.

## Files the pipeline writes (so I can find them later)

- `outputs/data_meta.json` — how many rows I used, year min/max, etc.  
- `outputs/split_metadata.json` — sizes and year ranges per split. Baseline scores live in `outputs/baseline_metrics.csv`.
- `outputs/classification_support.csv` — whether decade classification was allowed for each split.

If something looks weird, I re-run `python run_experiment.py` from the `Final_Project` folder and trust the newer outputs.
