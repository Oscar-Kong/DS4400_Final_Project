# DS 4400 final project — what I ran and how I’m using the numbers

Hey — this is my notes file for the final so I don’t forget what the code is doing. All the actual tables get dumped when I run `run_experiment.py` into `outputs/metrics.csv`, `outputs/baseline_metrics.csv`, and `outputs/RESULTS_SUMMARY.md`.

## What I’m trying to predict

I’m mostly doing **regression**: predict **release year** from the 90 timbre-style features in YearPredictionMSD.

I also trained some **decade classifiers** (same data, but labels binned by decade) because it helped me talk about confusion matrices and ROC in the report, and it made the patterns easier to explain in words.

**How I’m planning to present it in the video/writeup**

- I’ll lead with the **random split** — it’s the cleanest “normal ML” story.
- Then I’ll show the **blocked holdout** split as a stricter check (test years aren’t in training).
- I’ll show the **future extrapolation** split last and be honest that it’s brutal — it’s mostly there to show the model can look good when years are mixed randomly but struggle when the task is “old music → predict newer music.”

## Models I used (regression)

I picked four regressors — enough to compare linear vs regularized linear vs trees, without adding a fifth model that was slow and didn’t change my conclusions:

| What I called it in the CSV | Why I included it |
| --- | --- |
| `linear_regression` | Simple baseline. If I can’t beat this, something’s wrong. |
| `ridge_regression` | Same idea as linear but with L2; features are correlated so this usually helps. I picked the strength of regularization on the validation slice. |
| `random_forest` | Nonlinear + I can read off feature importances for the writeup. |
| `hist_gradient_boosting` | Strong default for tabular data; this was usually my best or close to it. |

I dropped the older sklearn `GradientBoostingRegressor` from my pipeline because it was slower and didn’t really change what I wanted to say for the project.

## How I split the data

### Random split

Roughly **70% train / 15% validation / 15% test**, fixed seed (`42`) so my results are reproducible.

I tune hyperparameters on **train vs validation**, then I **refit on train+validation** and only then score **test** (so the test set stays honest).

### Blocked holdout

- Validation years: **1971–1980**
- Test years: **1981–1995**
- Training pool: everything **except** the test window

I liked this split because the model never sees the exact test years during training, but I’m not yet forcing it to extrapolate past the newest year in the dataset like the “future” split does.

### Future extrapolation (stress test)

- Train: **year ≤ 2000**
- For tuning: **≤ 1995** vs **1996–2000**
- Test: **year > 2000**

This one is supposed to look bad (or at least worse). I’m using it in the “limitations / why is this hard” part of the report, not as my main accuracy brag.

## Baselines

I compare against dumb predictors so I’m not impressed by numbers that are barely better than guessing the average year:

- always predict the **mean** year from training  
- always predict the **median** year from training  
- for the future split only, also **“last year seen in training”** as another sanity check  

If my real models don’t beat these, I shouldn’t oversell them.

## Decade classification (extra)

The script only runs full decade classification when **every decade label in the test set also shows up in training** — otherwise sklearn would be trying to predict classes it never saw, which isn’t the point of what I’m doing here.

- I’m using the **confusion matrices** and **ROC (one-vs-rest)** plots from the splits where that check passes.
- If the **future extrapolation** split skips classification, that’s expected; I’ll explain in the video that newer decades weren’t in the training labels.

## What I’d put in the final writeup / slides

- Random split table first, then blocked holdout, then future split as a “hard mode” discussion.  
- Show **baselines** next to real models.  
- Use **feature importances**, **ridge coefficients**, **residual plots**, and **decade confusion** screenshots to explain what the model is actually learning and where it messes up.

That’s it — the code is the source of truth; this file is just how I’m framing it for class.
