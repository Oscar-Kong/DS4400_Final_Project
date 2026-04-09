#!/usr/bin/env python3
"""
Main script for my DS 4400 final — trains models on YearPredictionMSD and writes
tables/plots under `outputs/`.

From `Final_Project/`:
  python -m venv .venv
  source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  python run_experiment.py
  python run_experiment.py --sample-fraction 0.05   # faster while debugging
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

from src.analysis import decade_posthoc, plot_residuals, ridge_coef_table
from src.data import load_year_prediction_msd
from src.eda import run_eda
from src.eval import regression_metrics
from src.splits import (
    blocked_holdout_masks,
    future_extrapolation_masks,
    random_train_val_test,
)
from src.visualize import (
    plot_decade_confusion_heatmap,
    plot_feature_importance_bars,
    plot_model_comparison,
    plot_predicted_vs_actual,
)

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent / "outputs"
N_TUNE_TRAIN = 80_000
N_TUNE_VAL = 40_000
RIDGE_ALPHAS = np.logspace(-3, 5, 40)
RF_PARAM_DIST = {
    "n_estimators": [100, 200],
    "max_depth": [12, 20, 28, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.3, 0.5, "sqrt"],
}
HGBR_PARAM_DIST = {
    "learning_rate": [0.03, 0.06, 0.1],
    "max_depth": [4, 6, 8],
    "max_iter": [150, 250, 350],
    "min_samples_leaf": [15, 30],
    "l2_regularization": [0.0, 0.1],
}


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n_max:
        return X, y
    idx = RNG.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def _year_bounds(y: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).ravel()
    return {"min": int(np.min(y)), "max": int(np.max(y))}


def _baseline_rows(
    split_name: str,
    y_train_ref: np.ndarray,
    y_test: np.ndarray,
    include_last_seen: bool = False,
) -> list[dict]:
    y_train_ref = np.asarray(y_train_ref).ravel()
    y_test = np.asarray(y_test).ravel()

    rows: list[dict] = []
    baseline_specs = [
        ("baseline_train_mean", float(np.mean(y_train_ref))),
        ("baseline_train_median", float(np.median(y_train_ref))),
    ]
    if include_last_seen:
        baseline_specs.append(("baseline_last_seen_year", float(np.max(y_train_ref))))

    for model_name, constant in baseline_specs:
        pred = np.full_like(y_test, fill_value=constant, dtype=float)
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                **regression_metrics(y_test, pred),
                "note": f"constant={constant:.3f}",
            }
        )
    return rows


def _select_ridge_alpha(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
) -> float:
    best_alpha = float(RIDGE_ALPHAS[0])
    best_mae = float("inf")

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=float(alpha))
        model.fit(X_train_s, y_train)
        pred = model.predict(X_val_s)
        mae = mean_absolute_error(y_val, pred)
        if mae < best_mae:
            best_mae = float(mae)
            best_alpha = float(alpha)

    return best_alpha


def _select_params(
    estimator_factory,
    param_dist: dict[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_iter: int,
) -> dict[str, object]:
    X_train_s, y_train_s = _subsample(X_train, y_train, N_TUNE_TRAIN)
    X_val_s, y_val_s = _subsample(X_val, y_val, N_TUNE_VAL)

    best_params: dict[str, object] | None = None
    best_mae = float("inf")

    for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=42):
        model = estimator_factory(**params)
        model.fit(X_train_s, y_train_s)
        pred = model.predict(X_val_s)
        mae = mean_absolute_error(y_val_s, pred)
        if mae < best_mae:
            best_mae = float(mae)
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Parameter search failed to evaluate any candidate.")
    return best_params


def _tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    return _select_params(
        estimator_factory=lambda **params: RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params,
        ),
        param_dist=RF_PARAM_DIST,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iter=12,
    )


def _tune_hist_gbrt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    return _select_params(
        estimator_factory=lambda **params: HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=False,
            **params,
        ),
        param_dist=HGBR_PARAM_DIST,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iter=14,
    )


def _split_metadata(
    split_name: str,
    y_tune_train: np.ndarray,
    y_tune_val: np.ndarray,
    y_train_ref: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    return {
        "split": split_name,
        "tune_train_rows": int(len(y_tune_train)),
        "tune_val_rows": int(len(y_tune_val)),
        "train_rows": int(len(y_train_ref)),
        "test_rows": int(len(y_test)),
        "tune_train_years": _year_bounds(y_tune_train),
        "tune_val_years": _year_bounds(y_tune_val),
        "train_years": _year_bounds(y_train_ref),
        "test_years": _year_bounds(y_test),
    }


def _run_explicit_validation_split(
    split_name: str,
    X_tune_train: np.ndarray,
    y_tune_train: np.ndarray,
    X_tune_val: np.ndarray,
    y_tune_val: np.ndarray,
    X_train_ref: np.ndarray,
    y_train_ref: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    include_last_seen_baseline: bool = False,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    # Loose floors so `--sample-fraction` smoke runs work; full UCI data is far larger.
    if len(X_test) < 25 or len(X_train_ref) < 150 or len(X_tune_val) < 15:
        raise RuntimeError(
            f"Split '{split_name}' produced too few samples for a stable run."
        )

    scaler_final = StandardScaler()
    X_train_ref_s = scaler_final.fit_transform(X_train_ref)
    X_test_s = scaler_final.transform(X_test)
    X_tune_train_s = scaler_final.transform(X_tune_train)
    X_tune_val_s = scaler_final.transform(X_tune_val)

    rows: list[dict] = []
    baseline_rows = _baseline_rows(
        split_name=split_name,
        y_train_ref=y_train_ref,
        y_test=y_test,
        include_last_seen=include_last_seen_baseline,
    )

    lr = LinearRegression()
    lr.fit(X_train_ref_s, y_train_ref)
    pred = lr.predict(X_test_s)
    rows.append({"split": split_name, "model": "linear_regression", **regression_metrics(y_test, pred)})

    ridge_alpha = _select_ridge_alpha(X_tune_train_s, y_tune_train, X_tune_val_s, y_tune_val)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_train_ref_s, y_train_ref)
    pred = ridge.predict(X_test_s)
    rows.append(
        {
            "split": split_name,
            "model": "ridge_regression",
            **regression_metrics(y_test, pred),
            "note": f"alpha={ridge_alpha:.4g}",
        }
    )

    rf_params = _tune_random_forest(X_tune_train, y_tune_train, X_tune_val, y_tune_val)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X_train_ref, y_train_ref)
    pred = rf.predict(X_test)
    rows.append(
        {
            "split": split_name,
            "model": "random_forest",
            **regression_metrics(y_test, pred),
            "note": json.dumps(rf_params, sort_keys=True),
        }
    )

    hgbr_params = _tune_hist_gbrt(X_tune_train, y_tune_train, X_tune_val, y_tune_val)
    hgbr = HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=False,
        **hgbr_params,
    )
    hgbr.fit(X_train_ref, y_train_ref)
    pred = hgbr.predict(X_test)
    rows.append(
        {
            "split": split_name,
            "model": "hist_gradient_boosting",
            **regression_metrics(y_test, pred),
            "note": json.dumps(hgbr_params, sort_keys=True),
        }
    )

    artifacts = {
        "lr": lr,
        "ridge": ridge,
        "rf": rf,
        "hgbr": hgbr,
        "X_train_raw": X_train_ref,
        "X_test_raw": X_test,
        "X_test_scaled": X_test_s,
        "y_train": y_train_ref,
        "y_test": y_test,
        "feature_names": feature_names,
        "split_meta": _split_metadata(
            split_name=split_name,
            y_tune_train=y_tune_train,
            y_tune_val=y_tune_val,
            y_train_ref=y_train_ref,
            y_test=y_test,
        ),
    }
    return rows, baseline_rows, artifacts


def run_random_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[list[dict], list[dict], dict[str, object]]:
    (
        _idx_train,
        _idx_val,
        _idx_test,
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

    X_train_ref = np.vstack([X_train, X_val])
    y_train_ref = np.concatenate([y_train, y_val])

    return _run_explicit_validation_split(
        split_name="random",
        X_tune_train=X_train,
        y_tune_train=y_train,
        X_tune_val=X_val,
        y_tune_val=y_val,
        X_train_ref=X_train_ref,
        y_train_ref=y_train_ref,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )


def run_masked_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    masks: dict[str, np.ndarray],
    split_name: str,
    include_last_seen_baseline: bool = False,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    train_pool = masks["train_pool"]
    test_pool = masks["test_pool"]
    tune_train = masks["tune_train"]
    tune_val = masks["tune_val"]

    return _run_explicit_validation_split(
        split_name=split_name,
        X_tune_train=X[tune_train],
        y_tune_train=y[tune_train],
        X_tune_val=X[tune_val],
        y_tune_val=y[tune_val],
        X_train_ref=X[train_pool],
        y_train_ref=y[train_pool],
        X_test=X[test_pool],
        y_test=y[test_pool],
        feature_names=feature_names,
        include_last_seen_baseline=include_last_seen_baseline,
    )


def _write_summary(
    meta: dict[str, object],
    metrics_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    sample_fraction = meta["sample_fraction"]
    sample_note = "full dataset" if sample_fraction is None else f"sample_fraction={sample_fraction}"

    summary_lines = [
        "# Results from my last `run_experiment.py` run",
        "",
        "Auto-saved so I can paste tables into my report without re-copying numbers.",
        "",
        f"- Rows after loading/cleaning: **{meta['n_samples']}**",
        f"- Run: **{sample_note}**",
        "",
        "## Regression (my four models + splits)",
        "```",
        metrics_df.to_string(index=False),
        "```",
        "",
        "## Baselines (mean / median predictors — for comparison)",
        "```",
        baseline_df.to_string(index=False),
        "```",
    ]

    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(summary_lines))


def _cleanup_previous_outputs(out_dir: Path) -> None:
    managed_patterns = [
        "baseline_metrics.csv",
        "classification_metrics.csv",
        "classification_report_*.txt",
        "classification_skipped_*.txt",
        "classification_support.csv",
        "data_meta.json",
        "decade_accuracy_hgbr_*.txt",
        "decade_confusion_*.csv",
        "feature_importance_rf_*.csv",
        "metrics.csv",
        "RESULTS_SUMMARY.md",
        "ridge_coef_*.csv",
        "split_metadata.json",
    ]
    figure_patterns = [
        "confusion_matrix_*.png",
        "decade_confusion_heatmap_*.png",
        "eda_*.png",
        "feature_importance_bar_*.png",
        "model_comparison_*.png",
        "predicted_vs_actual_*.png",
        "residuals_*.png",
        "roc_curves_*.png",
    ]

    for pattern in managed_patterns:
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)

    fig_dir = out_dir / "figures"
    if fig_dir.exists():
        for pattern in figure_patterns:
            for path in fig_dir.glob(pattern):
                path.unlink(missing_ok=True)


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
    _cleanup_previous_outputs(OUT)

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

    print("Running EDA ...")
    run_eda(X_df, y_series, OUT)
    print("EDA complete; figures saved.")

    split_runs = [
        ("random", lambda: run_random_split(X, y, feature_names)),
        (
            "blocked_holdout",
            lambda: run_masked_split(
                X,
                y,
                feature_names,
                blocked_holdout_masks(y),
                split_name="blocked_holdout",
            ),
        ),
        (
            "future_extrapolation",
            lambda: run_masked_split(
                X,
                y,
                feature_names,
                future_extrapolation_masks(y, train_year_max=2000, tune_train_year_max=1995),
                split_name="future_extrapolation",
                include_last_seen_baseline=True,
            ),
        ),
    ]

    regression_rows: list[dict] = []
    baseline_rows: list[dict] = []
    artifacts_by_split: dict[str, dict[str, object]] = {}

    for split_name, runner in split_runs:
        print(f"Running split: {split_name} ...")
        split_rows, split_baselines, split_artifacts = runner()
        regression_rows.extend(split_rows)
        baseline_rows.extend(split_baselines)
        artifacts_by_split[split_name] = split_artifacts

    metrics_df = pd.DataFrame(regression_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    baseline_df.to_csv(OUT / "baseline_metrics.csv", index=False)

    split_metadata = {
        split_name: artifacts["split_meta"]
        for split_name, artifacts in artifacts_by_split.items()
    }

    for split_name, artifacts in artifacts_by_split.items():
        y_test = artifacts["y_test"]
        feature_names_split = artifacts["feature_names"]

        ridge_pred = artifacts["ridge"].predict(artifacts["X_test_scaled"])
        plot_residuals(
            y_test,
            ridge_pred,
            title=f"{split_name} / ridge_regression",
            out_path=fig_dir / f"residuals_ridge_{split_name}.png",
        )

        hgbr_pred = artifacts["hgbr"].predict(artifacts["X_test_raw"])
        plot_residuals(
            y_test,
            hgbr_pred,
            title=f"{split_name} / hist_gradient_boosting",
            out_path=fig_dir / f"residuals_hgbr_{split_name}.png",
        )

        rf = artifacts["rf"]
        rf_importance = pd.DataFrame(
            {
                "feature": feature_names_split,
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        rf_importance.head(20).to_csv(
            OUT / f"feature_importance_rf_{split_name}.csv",
            index=False,
        )

        ridge_coef_table(artifacts["ridge"], feature_names_split, top_k=20).to_csv(
            OUT / f"ridge_coef_{split_name}.csv",
            index=False,
        )

        ctab, dacc = decade_posthoc(y_test, hgbr_pred)
        ctab.to_csv(OUT / f"decade_confusion_{split_name}_hgbr.csv")
        (OUT / f"decade_accuracy_hgbr_{split_name}.txt").write_text(
            f"decade_bin_accuracy\t{dacc:.4f}\n"
        )

        plot_feature_importance_bars(rf, feature_names_split, split_name, OUT)
        plot_predicted_vs_actual(y_test, hgbr_pred, "hgbr", split_name, OUT)
        plot_decade_confusion_heatmap(ctab, dacc, split_name, OUT)

    print("Generating regression comparison plots ...")
    plot_model_comparison(metrics_df, OUT)

    combined_metadata = {
        "data": meta,
        "splits": split_metadata,
    }
    (OUT / "split_metadata.json").write_text(json.dumps(combined_metadata, indent=2))

    _write_summary(meta, metrics_df, baseline_df)

    print(metrics_df.to_string(index=False))
    print("\nBaseline metrics")
    print(baseline_df.to_string(index=False))
    print("\nWrote outputs to", OUT)


if __name__ == "__main__":
    main()
