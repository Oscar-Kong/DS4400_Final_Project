"""Exploratory data analysis: distributions, correlations, PCA, summary stats."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_eda(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
) -> None:
    """Run all EDA analyses and save outputs to *out_dir*."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _year_distribution(y, fig_dir)
    _summary_stats(X, y, out_dir)
    _correlation_with_target(X, y, fig_dir)
    _top_feature_distributions(X, y, fig_dir)
    _pca_scatter(X, y, fig_dir)
    _feature_vs_year_scatter(X, y, fig_dir)


def _year_distribution(y: pd.Series, fig_dir: Path) -> None:
    """Histogram of release years."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(y.min(), y.max() + 2, 1)
    ax.hist(y, bins=bins, edgecolor="black", linewidth=0.3, color="#4C72B0")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Songs")
    ax.set_title("Distribution of Release Years")
    ax.axvline(2000, color="red", linestyle="--", linewidth=1, label="Time-split cutoff (2000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_year_distribution.png", dpi=150)
    plt.close(fig)


def _summary_stats(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    """Summary statistics for all features and the target."""
    stats = X.describe().T
    stats.index.name = "feature"

    target_stats = y.describe().to_frame("year").T
    target_stats.index.name = "feature"

    combined = pd.concat([target_stats, stats])
    combined.to_csv(out_dir / "eda_summary_stats.csv")


def _correlation_with_target(
    X: pd.DataFrame,
    y: pd.Series,
    fig_dir: Path,
) -> None:
    """Bar chart of Pearson correlation between each feature and release year."""
    corr = X.corrwith(y).sort_values()
    top_n = 20

    # Top 10 positive + top 10 negative
    top_pos = corr.tail(top_n // 2)
    top_neg = corr.head(top_n // 2)
    selected = pd.concat([top_neg, top_pos])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in selected.values]
    ax.barh(selected.index, selected.values, color=colors)
    ax.set_xlabel("Pearson Correlation with Release Year")
    ax.set_title(f"Top {top_n} Features by Correlation with Year")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_correlation_with_year.png", dpi=150)
    plt.close(fig)

    # Full feature-feature correlation heatmap (top 20 most correlated with year)
    top_features = corr.abs().nlargest(20).index.tolist()
    corr_matrix = X[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_title("Feature-Feature Correlation (Top 20 by |corr with year|)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _top_feature_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    fig_dir: Path,
) -> None:
    """Histograms for the top 8 features most correlated with year."""
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top_8 = corr.head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, feat in zip(axes.ravel(), top_8):
        ax.hist(X[feat], bins=50, edgecolor="black", linewidth=0.3, color="#4C72B0", alpha=0.8)
        r = X[feat].corr(y)
        ax.set_title(f"{feat} (r={r:.3f})", fontsize=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
    fig.suptitle("Distributions of Top 8 Features (by |correlation with year|)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_top_feature_distributions.png", dpi=150)
    plt.close(fig)


def _pca_scatter(X: pd.DataFrame, y: pd.Series, fig_dir: Path) -> None:
    """PCA scatter plot (first 2 components), colored by decade."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    decades = (np.floor(y.values / 10) * 10).astype(int)
    unique_decades = sorted(np.unique(decades))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(unique_decades), vmax=max(unique_decades))

    # Subsample for readability if dataset is large
    n = len(components)
    if n > 20000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=20000, replace=False)
        components_plot = components[idx]
        decades_plot = decades[idx]
    else:
        components_plot = components
        decades_plot = decades

    sc = ax.scatter(
        components_plot[:, 0],
        components_plot[:, 1],
        c=decades_plot,
        cmap=cmap,
        norm=norm,
        alpha=0.3,
        s=4,
    )
    fig.colorbar(sc, ax=ax, label="Decade")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA of Audio Features (colored by decade)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_pca_scatter.png", dpi=150)
    plt.close(fig)

    # Also save explained variance for all components
    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", markersize=3)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8, label="90% variance")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=0.8, label="95% variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_pca_variance.png", dpi=150)
    plt.close(fig)


def _feature_vs_year_scatter(
    X: pd.DataFrame,
    y: pd.Series,
    fig_dir: Path,
) -> None:
    """Scatter plots of top 4 features vs release year."""
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top_4 = corr.head(4).index.tolist()

    # Subsample for readability
    n = len(X)
    if n > 15000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=15000, replace=False)
    else:
        idx = np.arange(n)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, feat in zip(axes, top_4):
        r = X[feat].corr(y)
        ax.scatter(y.iloc[idx], X[feat].iloc[idx], alpha=0.1, s=2, color="#4C72B0")
        ax.set_xlabel("Release Year")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} vs Year (r={r:.3f})")
    fig.suptitle("Top Features vs Release Year", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_feature_vs_year.png", dpi=150)
    plt.close(fig)
