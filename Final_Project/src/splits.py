"""Random and time-aware train/validation/test splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    ~70% / ~15% / ~15% split (test_size and val_size are fractions of full data).
    First hold out test, then split remainder into train/val.
    """
    idx = np.arange(len(X))
    idx_rem, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_rem, test_size=rel_val, random_state=random_state
    )
    return (
        idx_train,
        idx_val,
        idx_test,
        X.iloc[idx_train].values,
        X.iloc[idx_val].values,
        X.iloc[idx_test].values,
        y.iloc[idx_train].values,
        y.iloc[idx_val].values,
        y.iloc[idx_test].values,
    )


def time_aware_masks(
    y: np.ndarray,
    train_year_max: int = 2000,
    tune_train_year_max: int = 1995,
) -> dict[str, np.ndarray]:
    """
    Temporal split: train pool year <= train_year_max, test year > train_year_max.
    Inner tuning: train_tune year <= tune_train_year_max,
                  val_tune tune_train_year_max < year <= train_year_max.
    """
    y = np.asarray(y).ravel()
    train_pool = y <= train_year_max
    test_pool = y > train_year_max
    tune_train = train_pool & (y <= tune_train_year_max)
    tune_val = train_pool & (y > tune_train_year_max)
    return {
        "train_pool": train_pool,
        "test_pool": test_pool,
        "tune_train": tune_train,
        "tune_val": tune_val,
    }
