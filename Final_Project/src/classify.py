"""Decade classification on decade-binned labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_decade(y: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(y).ravel() / 10) * 10).astype(int)


def run_decade_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    split_name: str,
    out_dir: Path,
) -> list[dict]:
    """Train classifiers on decade labels and generate visualizations.

    Returns a list of metric dicts for each classifier.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d_train = _to_decade(y_train)
    d_test = _to_decade(y_test)

    # Scale once inside the classification pipeline.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest_clf": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting_clf": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, random_state=42
        ),
    }

    rows: list[dict] = []
    best_model = None
    best_acc = -1.0

    for name, clf in classifiers.items():
        clf.fit(X_tr_s, d_train)
        pred = clf.predict(X_te_s)
        acc = accuracy_score(d_test, pred)
        f1 = f1_score(d_test, pred, average="weighted", zero_division=0)
        rows.append({
            "split": split_name,
            "classifier": name,
            "accuracy": float(acc),
            "weighted_f1": float(f1),
        })
        if acc > best_acc:
            best_acc = acc
            best_model = (name, clf)

    # Confusion matrix for the best classifier
    best_name, best_clf = best_model
    pred_best = best_clf.predict(X_te_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        d_test, pred_best, ax=ax, cmap="Blues",
        xticks_rotation=45, values_format="d",
    )
    ax.set_title(f"Confusion Matrix — {best_name} ({split_name})\nAccuracy: {best_acc:.1%}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_matrix_{split_name}.png", dpi=150)
    plt.close(fig)

    # ROC curves (one-vs-rest) for the best classifier
    if hasattr(best_clf, "predict_proba"):
        proba = best_clf.predict_proba(X_te_s)

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(best_clf.classes_):
            truth = (d_test == cls).astype(int)
            if truth.sum() == 0 or truth.sum() == len(truth):
                continue
            RocCurveDisplay.from_predictions(
                truth,
                proba[:, i],
                name=f"Decade {cls}",
                ax=ax,
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
        ax.set_title(f"ROC Curves (One-vs-Rest) — {best_name} ({split_name})")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"roc_curves_{split_name}.png", dpi=150)
        plt.close(fig)

    # Save classification report
    report = classification_report(d_test, pred_best, zero_division=0)
    (out_dir / f"classification_report_{split_name}.txt").write_text(
        f"Classifier: {best_name}\n\n{report}"
    )

    return rows
