#!/usr/bin/env python3
"""
Run full YearPredictionMSD regression experiment (random + time-aware splits).

Usage (from Final_Project/):
  . .venv/bin/activate
  pip install -r requirements.txt
  python run_experiment.py
  python run_experiment.py --sample-fraction 0.05   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from src.analysis import decade_posthoc, plot_residuals, ridge_coef_table
from src.data import load_year_prediction_msd
from src.eval import regression_metrics
from src.splits import random_train_val_test, time_aware_masks

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent / "outputs"
N_SEARCH = 80_000  # max rows for randomized hyperparam search


def _subsample(X: np.ndarray, y: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n_max:
        return X, y
    idx = RNG.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    Xs, ys = _subsample(X_train, y_train, N_SEARCH)
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [12, 20, 28, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.3, 0.5, "sqrt"],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(Xs, ys)
    return RandomForestRegressor(random_state=42, n_jobs=-1, **search.best_params_)


def tune_hist_gbrt(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> HistGradientBoostingRegressor:
    Xs, ys = _subsample(X_train, y_train, N_SEARCH)
    param_dist = {
        "learning_rate": [0.03, 0.06, 0.1],
        "max_depth": [4, 6, 8],
        "max_iter": [150, 250],
        "min_samples_leaf": [15, 30],
        "l2_regularization": [0.0, 0.1],
    }
    base = HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=14,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(Xs, ys)
    best = search.best_params_.copy()
    return HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        **best,
    )


def tune_gradient_boosting_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GradientBoostingRegressor:
    """Classic GBR on a subsample (slower than HGBR on full data)."""
    Xs, ys = _subsample(X_train, y_train, min(N_SEARCH, 40_000))
    param_dist = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
    }
    base = GradientBoostingRegressor(random_state=42)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=8,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(Xs, ys)
    return GradientBoostingRegressor(random_state=42, **search.best_params_)


def run_random_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[list[dict], dict[str, object]]:
    (
        _i_tr,
        _i_va,
        _i_te,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = random_train_val_test(
        pd.DataFrame(X, columns=feature_names),
        pd.Series(y),
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    scaler_final = StandardScaler()
    X_fit_s = scaler_final.fit_transform(X_combined)
    X_test_s = scaler_final.transform(X_test)

    rows: list[dict] = []
    artifacts: dict[str, object] = {}

    lr = LinearRegression()
    lr.fit(X_fit_s, y_combined)
    pred = lr.predict(X_test_s)
    m = regression_metrics(y_test, pred)
    rows.append({"split": "random", "model": "linear_regression", **m})
    artifacts["lr"] = lr

    ridge = RidgeCV(alphas=np.logspace(-3, 5, 40), cv=5)
    ridge.fit(X_fit_s, y_combined)
    pred = ridge.predict(X_test_s)
    m = regression_metrics(y_test, pred)
    rows.append(
        {
            "split": "random",
            "model": "ridge_cv",
            **m,
            "note": f"alpha={ridge.alpha_:.4g}",
        }
    )
    artifacts["ridge"] = ridge

    rf = tune_random_forest(X_train, y_train)
    rf.fit(X_fit_s, y_combined)
    pred = rf.predict(X_test_s)
    m = regression_metrics(y_test, pred)
    rows.append({"split": "random", "model": "random_forest", **m})
    artifacts["rf"] = rf

    hgbr = tune_hist_gbrt(X_train, y_train)
    hgbr.fit(X_fit_s, y_combined)
    pred = hgbr.predict(X_test_s)
    m = regression_metrics(y_test, pred)
    rows.append({"split": "random", "model": "hist_gradient_boosting", **m})
    artifacts["hgbr"] = hgbr

    gbr = tune_gradient_boosting_sklearn(X_train, y_train)
    gbr.fit(X_fit_s, y_combined)
    pred = gbr.predict(X_test_s)
    m = regression_metrics(y_test, pred)
    rows.append({"split": "random", "model": "gradient_boosting", **m})
    artifacts["gbr"] = gbr

    artifacts["y_test"] = y_test
    artifacts["X_test_s"] = X_test_s
    artifacts["scaler"] = scaler_final
    artifacts["feature_names"] = feature_names

    return rows, artifacts


def run_time_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[list[dict], dict[str, object]]:
    masks = time_aware_masks(y, train_year_max=2000, tune_train_year_max=1995)
    tr = masks["train_pool"]
    te = masks["test_pool"]
    tt = masks["tune_train"]
    tv = masks["tune_val"]

    X_train_pool = X[tr]
    y_train_pool = y[tr]
    X_test = X[te]
    y_test = y[te]

    X_tune_tr = X[tt]
    y_tune_tr = y[tt]
    X_tune_va = X[tv]
    y_tune_va = y[tv]

    if len(X_test) < 100 or len(X_train_pool) < 1000:
        raise RuntimeError(
            "Time-aware split produced too few samples; check year range in data."
        )

    scaler_tune = StandardScaler()
    X_ttr_s = scaler_tune.fit_transform(X_tune_tr)
    X_tva_s = scaler_tune.transform(X_tune_va)

    scaler_final = StandardScaler()
    X_fit_s = scaler_final.fit_transform(X_train_pool)
    X_test_s = scaler_final.transform(X_test)

    rows: list[dict] = []
    artifacts: dict[str, object] = {}

    def tune_rf_time() -> RandomForestRegressor:
        Xs, ys = _subsample(X_ttr_s, y_tune_tr, N_SEARCH)
        param_dist = {
            "n_estimators": [100, 200],
            "max_depth": [12, 20, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.3, 0.5, "sqrt"],
        }
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=10,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(Xs, ys)
        return RandomForestRegressor(
            random_state=42, n_jobs=-1, **search.best_params_
        )

    def tune_hgbr_time() -> HistGradientBoostingRegressor:
        Xs, ys = _subsample(X_ttr_s, y_tune_tr, N_SEARCH)
        param_dist = {
            "learning_rate": [0.03, 0.06, 0.1],
            "max_depth": [4, 6, 8],
            "max_iter": [150, 250],
            "min_samples_leaf": [15, 30],
            "l2_regularization": [0.0, 0.1],
        }
        base = HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=12,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(Xs, ys)
        best = search.best_params_.copy()
        return HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            **best,
        )

    lr = LinearRegression()
    lr.fit(X_fit_s, y_train_pool)
    pred = lr.predict(X_test_s)
    rows.append({"split": "time_aware", "model": "linear_regression", **regression_metrics(y_test, pred)})
    artifacts["lr"] = lr

    ridge = RidgeCV(alphas=np.logspace(-3, 5, 40), cv=5)
    ridge.fit(X_fit_s, y_train_pool)
    pred = ridge.predict(X_test_s)
    rows.append(
        {
            "split": "time_aware",
            "model": "ridge_cv",
            **regression_metrics(y_test, pred),
            "note": f"alpha={ridge.alpha_:.4g}",
        }
    )
    artifacts["ridge"] = ridge

    rf = tune_rf_time()
    rf.fit(X_fit_s, y_train_pool)
    pred = rf.predict(X_test_s)
    rows.append({"split": "time_aware", "model": "random_forest", **regression_metrics(y_test, pred)})
    artifacts["rf"] = rf

    hgbr = tune_hgbr_time()
    hgbr.fit(X_fit_s, y_train_pool)
    pred = hgbr.predict(X_test_s)
    rows.append(
        {"split": "time_aware", "model": "hist_gradient_boosting", **regression_metrics(y_test, pred)}
    )
    artifacts["hgbr"] = hgbr

    gbr = tune_gradient_boosting_sklearn(X_tune_tr, y_tune_tr)
    gbr.fit(X_fit_s, y_train_pool)
    pred = gbr.predict(X_test_s)
    rows.append(
        {"split": "time_aware", "model": "gradient_boosting", **regression_metrics(y_test, pred)}
    )
    artifacts["gbr"] = gbr

    artifacts["y_test"] = y_test
    artifacts["X_test_s"] = X_test_s
    artifacts["feature_names"] = feature_names

    return rows, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional fraction of data for quick runs (e.g. 0.05).",
    )
    parser.add_argument(
        "--data-home",
        type=str,
        default=None,
        help="Optional directory for UCI YearPredictionMSD cache (default: ./data_cache).",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    X_df, y_series, feature_names = load_year_prediction_msd(
        data_home=args.data_home,
        sample_fraction=args.sample_fraction,
        random_state=42,
    )
    X = X_df.values
    y = y_series.values

    meta = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "year_min": float(np.min(y)),
        "year_max": float(np.max(y)),
        "sample_fraction": args.sample_fraction,
    }
    (OUT / "data_meta.json").write_text(json.dumps(meta, indent=2))

    all_rows: list[dict] = []
    random_rows, random_art = run_random_split(X, y, feature_names)
    all_rows.extend(random_rows)

    time_rows, time_art = run_time_split(X, y, feature_names)
    all_rows.extend(time_rows)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)

    for name, art in [("random", random_art), ("time_aware", time_art)]:
        y_te = art["y_test"]
        fn = art["feature_names"]
        pred_ridge = art["ridge"].predict(art["X_test_s"])
        plot_residuals(
            y_te,
            pred_ridge,
            title=f"{name} / ridge_cv",
            out_path=fig_dir / f"residuals_ridge_{name}.png",
        )
        pred_h = art["hgbr"].predict(art["X_test_s"])
        plot_residuals(
            y_te,
            pred_h,
            title=f"{name} / hist_gradient_boosting",
            out_path=fig_dir / f"residuals_hgbr_{name}.png",
        )

        rf: RandomForestRegressor = art["rf"]
        imp = pd.DataFrame(
            {"feature": fn, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)
        imp.head(20).to_csv(
            OUT / f"feature_importance_rf_{name}.csv", index=False
        )

        rc = ridge_coef_table(art["ridge"], fn, top_k=20)
        rc.to_csv(OUT / f"ridge_coef_{name}.csv", index=False)

        _, dacc = decade_posthoc(y_te, pred_h)
        (OUT / f"decade_accuracy_hgbr_{name}.txt").write_text(
            f"decade_bin_accuracy\t{dacc:.4f}\n"
        )

    ctab_r, acc_r = decade_posthoc(
        random_art["y_test"], random_art["hgbr"].predict(random_art["X_test_s"])
    )
    ctab_r.to_csv(OUT / "decade_confusion_random_hgbr.csv")
    ctab_t, acc_t = decade_posthoc(
        time_art["y_test"], time_art["hgbr"].predict(time_art["X_test_s"])
    )
    ctab_t.to_csv(OUT / "decade_confusion_time_hgbr.csv")

    summary_lines = [
        "# Auto-generated result summary",
        "",
        f"Rows used: {meta['n_samples']}",
        "",
        "```",
        metrics_df.to_string(index=False),
        "```",
        "",
        f"Decade bin accuracy (HGBR, random): {acc_r:.4f}",
        f"Decade bin accuracy (HGBR, time-aware): {acc_t:.4f}",
    ]
    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(summary_lines))

    print(metrics_df.to_string(index=False))
    print("\nWrote outputs to", OUT)


if __name__ == "__main__":
    main()
