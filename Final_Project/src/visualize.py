"""Report-ready visualizations: model comparison, feature importance bars,
predicted vs actual, decade confusion heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = [
    "linear_regression",
    "ridge_regression",
    "random_forest",
    "hist_gradient_boosting",
]
SPLIT_ORDER = ["random", "blocked_holdout", "future_extrapolation"]


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar charts comparing RMSE and MAE across models and splits."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("rmse", "mae"):
        pivot = metrics_df.pivot(index="model", columns="split", values=metric)
        ordered_models = [name for name in MODEL_ORDER if name in pivot.index]
        ordered_splits = [name for name in SPLIT_ORDER if name in pivot.columns]
        pivot = pivot.reindex(index=ordered_models, columns=ordered_splits)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pivot))
        width = 0.25

        cols = list(pivot.columns)
        for i, col in enumerate(cols):
            offset = (i - (len(cols) - 1) / 2) * width
            bars = ax.bar(x + offset, pivot[col], width, label=col)
            for bar, val in zip(bars, pivot[col]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric.upper()} (years)")
        ax.set_title(f"Model Comparison — {metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.legend(title="Split Strategy")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"model_comparison_{metric}.png", dpi=150)
        plt.close(fig)


def plot_feature_importance_bars(
    rf_model,
    feature_names: list[str],
    split_name: str,
    out_dir: Path,
    top_k: int = 20,
) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    fig_dir = out_dir / "figures"
    imp = pd.Series(rf_model.feature_importances_, index=feature_names)
    imp = imp.nlargest(top_k).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"Top {top_k} Features — Random Forest ({split_name})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"feature_importance_bar_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    out_dir: Path,
) -> None:
    """Scatter plot of predicted vs actual year with diagonal reference."""
    fig_dir = out_dir / "figures"
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.15, s=4, color="#4C72B0")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True Year")
    ax.set_ylabel("Predicted Year")
    ax.set_title(f"Predicted vs Actual — {model_name} ({split_name})")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(fig_dir / f"predicted_vs_actual_{model_name}_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_decade_confusion_heatmap(
    ctab: pd.DataFrame,
    accuracy: float,
    split_name: str,
    out_dir: Path,
) -> None:
    """Heatmap of the decade-level confusion table."""
    fig_dir = out_dir / "figures"

    # Normalize rows to percentages
    row_sums = ctab.sum(axis=1)
    ctab_pct = ctab.div(row_sums, axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ctab_pct.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ctab_pct.columns)))
    ax.set_yticks(range(len(ctab_pct.index)))
    ax.set_xticklabels(ctab_pct.columns, rotation=45, ha="right")
    ax.set_yticklabels(ctab_pct.index)
    ax.set_xlabel("Predicted Decade")
    ax.set_ylabel("True Decade")
    ax.set_title(f"Decade Confusion — HGBR ({split_name})\nAccuracy: {accuracy:.1%}")

    # Annotate cells
    for i in range(len(ctab_pct.index)):
        for j in range(len(ctab_pct.columns)):
            val = ctab_pct.values[i, j]
            count = ctab.values[i, j]
            if count > 0:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}\n({count})", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Row proportion")
    fig.tight_layout()
    fig.savefig(fig_dir / f"decade_confusion_heatmap_{split_name}.png", dpi=150)
    plt.close(fig)
