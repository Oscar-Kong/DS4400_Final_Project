"""Load and clean YearPredictionMSD from the UCI Machine Learning Repository."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
    "YearPredictionMSD.txt.zip"
)
ZIP_NAME = "YearPredictionMSD.txt.zip"
TXT_NAME = "YearPredictionMSD.txt"
YEAR_MIN, YEAR_MAX = 1900, 2025


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data_cache"


def _ensure_uci_txt(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / TXT_NAME
    if txt_path.exists():
        return txt_path
    zip_path = cache_dir / ZIP_NAME
    urlretrieve(UCI_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(TXT_NAME, path=cache_dir)
    return txt_path


def load_year_prediction_msd(
    data_home: str | None = None,
    sample_fraction: float | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load YearPredictionMSD (first column = year, next 90 columns = features).

    Parameters
    ----------
    data_home
        Directory for cached UCI zip/txt. Defaults to ``Final_Project/data_cache``.
    sample_fraction
        Optional subsample fraction in (0, 1) for quick experiments.
    """
    cache = Path(data_home) if data_home else _default_cache_dir()
    txt = _ensure_uci_txt(cache)

    # Single file, comma-separated, no header (year, 90 features)
    df = pd.read_csv(txt, header=None)
    if df.shape[1] != 91:
        raise ValueError(f"Expected 91 columns (year + 90 features), got {df.shape[1]}")

    y = pd.to_numeric(df.iloc[:, 0], errors="coerce").astype(float)
    X = df.iloc[:, 1:91].apply(pd.to_numeric, errors="coerce")
    X.columns = [f"f{i:02d}" for i in range(90)]

    valid = y.notna() & (y >= YEAR_MIN) & (y <= YEAR_MAX)
    valid &= X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    dup = X.duplicated(keep="first")
    if dup.any():
        X = X.loc[~dup].reset_index(drop=True)
        y = y.loc[~dup].reset_index(drop=True)

    feature_names = list(X.columns)

    if sample_fraction is not None:
        if not (0 < sample_fraction < 1):
            raise ValueError("sample_fraction must be in (0, 1)")
        rng = np.random.default_rng(random_state)
        n = max(1, int(round(len(X) * sample_fraction)))
        n = min(n, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y, feature_names
