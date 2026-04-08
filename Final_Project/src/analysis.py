"""Plots, feature importance, post-hoc decade analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import RocCurveDisplay

ModelWithCoef = Ridge | RidgeCV


def feature_year_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Pearson correlation of each feature with release year (univariate signal)."""
    yv = pd.to_numeric(y, errors="coerce")
    corrs = X.apply(lambda col: col.corr(yv))
    out = pd.DataFrame({"feature": X.columns, "corr_year": corrs})
    out["abs_corr"] = out["corr_year"].abs()
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def plot_eda_year_distribution(y: np.ndarray | pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y).ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Release year (label)")
    ax.set_ylabel("Count")
    ax.set_title("YearPredictionMSD: distribution of release years")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eda_feature_correlation_bars(
    corr_table: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = corr_table.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(sub["corr_year"].values >= 0, "tab:blue", "tab:red")
    ax.barh(sub["feature"], sub["corr_year"], color=colors)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Pearson r (feature vs year)")
    ax.set_title(f"Top {top_k} features by |correlation| with year")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eda_selected_feature_histograms(
    X: pd.DataFrame,
    feature_cols: list[str],
    out_path: Path,
) -> None:
    """Histograms for a small set of columns (rubric: selective feature distributions)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(feature_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False,
    )
    for i, col in enumerate(feature_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.hist(
            X[col].values,
            bins=40,
            color="slategray",
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    for j in range(len(feature_cols), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decade_confusion_heatmap(
    ctab: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    """Confusion-style table for post-hoc decade bins (true × predicted counts)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = ctab.values.astype(float)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(ctab.columns, rotation=45, ha="right")
    ax.set_yticklabels(ctab.index)
    ax.set_xlabel("Predicted decade")
    ax.set_ylabel("True decade")
    ax.set_title(title)
    vmax = mat.max() if mat.size else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="w" if v > vmax * 0.55 and vmax > 0 else "k",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_median_year_split(
    y_true: np.ndarray,
    scores: np.ndarray,
    year_threshold: float,
    out_path: Path,
    title: str,
) -> None:
    """
    ROC for a binary view of the task: year >= threshold vs < threshold.

    Auxiliary visualization for coursework rubrics; regression metrics remain primary.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    y_bin = (y_true >= year_threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_bin, scores, ax=ax, name="model")
    ax.set_title(title + f"\n(positive class: year ≥ {year_threshold:.0f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, resid, alpha=0.15, s=4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("Predicted year")
    axes[0].set_ylabel("Residual (true − pred)")
    axes[0].set_title(f"{title}: residuals vs prediction")

    axes[1].scatter(y_true, resid, alpha=0.15, s=4)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("True year")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{title}: residuals vs true year")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def interpretation_summary(
    feature_names: list[str],
    rf_importances: np.ndarray,
    corr_table: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> dict[str, object]:
    """Structured hints for writeups: important features, errors, task difficulty."""
    imp = pd.DataFrame({"feature": feature_names, "importance": rf_importances})
    imp = imp.sort_values("importance", ascending=False)
    top_imp = imp.head(12)["feature"].tolist()
    top_corr = corr_table.head(12)["feature"].tolist()
    by_mae = metrics_df.sort_values("mae").head(8)
    return {
        "feature_representation": (
            "Inputs are 90 numeric timbre summary statistics (f00–f89) from "
            "YearPredictionMSD; values are standardized with StandardScaler fit "
            "only on training data for each split regime."
        ),
        "feature_selection": (
            "All 90 features are used for models. Univariate correlations with year "
            "and random-forest importances highlight which coordinates carry the "
            "most signal; highly correlated timbre dimensions motivate ridge and "
            "tree ensembles."
        ),
        "most_relevant_features": {
            "by_random_forest_importance_top12": top_imp,
            "by_abs_correlation_with_year_top12": top_corr,
        },
        "why_models_make_errors": (
            "Release year is not fully determined by coarse timbre summaries; "
            "adjacent years overlap in feature space. Residual plots show "
            "heteroscedasticity and systematic bias under temporal shift (time-aware "
            "split). Large max_error / heavy tails indicate outlier eras or styles."
        ),
        "why_task_is_challenging": (
            "Label is a continuous year from audio proxies, not causal features; "
            "distribution shift (random vs time-aware evaluation) reveals spurious "
            "temporal cues that help under random mixing but hurt on future years."
        ),
        "metrics_table_best_mae": by_mae.to_dict(orient="records"),
    }


def decade_posthoc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Bin true and predicted years into decades; return confusion-style table and accuracy."""
    def decade(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        return (np.floor(y / 10) * 10).astype(int)

    dt = decade(y_true)
    dp = decade(y_pred)
    acc = float(np.mean(dt == dp))
    labels = sorted(np.unique(np.concatenate([dt, dp])))
    mat = pd.crosstab(
        pd.Series(dt, name="true_decade"),
        pd.Series(dp, name="pred_decade"),
    ).reindex(index=labels, columns=labels, fill_value=0)
    return mat, acc


def ridge_coef_table(
    model: ModelWithCoef,
    feature_names: list[str],
    top_k: int = 15,
) -> pd.DataFrame:
    coef = np.asarray(model.coef_).ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs_coef"] = np.abs(df["coef"])
    return df.sort_values("abs_coef", ascending=False).head(top_k).drop(
        columns=["abs_coef"]
    )
