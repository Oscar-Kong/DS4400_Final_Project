# DS 4400 project notes

The actual numbers come from:

- `outputs/metrics.csv`
- `outputs/baseline_metrics.csv`
- `outputs/RESULTS_SUMMARY.md`

## What we are predicting

The final project is **regression only**. The target is the song's release year.

That is the cleanest framing for this dataset. Year is ordered and continuous, so it makes more sense to predict the year directly than to force everything into decade labels.

For the report, we still like using a decade heatmap because it is easy to read. That heatmap now comes from binning the regression predictions and true years into decades after the model runs. So the project stays regression-only, but we still get a useful visual for interpretation.

## Models

We kept four regression models:

| Name in results | Why it is there |
| --- | --- |
| `linear_regression` | Basic baseline |
| `ridge_regression` | Linear model with regularization for correlated features |
| `random_forest` | Nonlinear tree ensemble with feature importances |
| `hist_gradient_boosting` | Strong tabular model and usually the best overall |

We dropped the extra gradient boosting path because it was just adding code and runtime without changing the story much.

## Splits

### Random split

This is the standard benchmark: about 70% train, 15% validation, 15% test with a fixed seed.

This is the result we would lead with, because it answers the most basic question: is there usable signal in these features or not?

### Blocked holdout

- validation years: 1971-1980
- test years: 1981-1995
- training pool: everything except the test window

This is a tougher check than the random split, but it is not a pure train-on-past, test-on-future setup. It is better to describe it as a withheld time block.

### Future extrapolation

- train on years up to 2000
- tune on years up to 1995 versus 1996-2000
- test on years after 2000

This is the stress test. It is supposed to be hard, and the results should be discussed that way.

## Baselines

We compare every split against simple baselines:

- training mean year
- training median year
- last year seen in training for the future split

That matters because a model is not very interesting if it cannot beat a dumb constant predictor.

## How we would present the results

I would keep the story simple:

1. Start with the random split.
2. Show the blocked holdout to make the point that evaluation design matters.
3. Show the future extrapolation split as the hard case.
4. Put the baselines next to the learned models so the comparison stays honest.
5. Use feature importances, ridge coefficients, residual plots, and the random-split decade heatmap to explain what the models are doing.

## Main takeaway

The project works best when it stays focused on one clear claim:

- these audio features do contain temporal signal
- that signal is useful for held-out prediction
- true forward extrapolation is much harder than ordinary random-split evaluation
