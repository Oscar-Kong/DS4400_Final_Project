"""Plots, feature importance, post-hoc decade analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


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
    model: Ridge,
    feature_names: list[str],
    top_k: int = 15,
) -> pd.DataFrame:
    coef = np.asarray(model.coef_).ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs_coef"] = np.abs(df["coef"])
    return df.sort_values("abs_coef", ascending=False).head(top_k).drop(
        columns=["abs_coef"]
    )
