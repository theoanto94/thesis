"""Data loading, temporal split, and target/feature construction.

Loads the ``claims`` table from DuckDB, builds the structural feature matrix
``X`` (no cost columns => no target leakage), derives every target rate
(total + per-category) as ``sum(cost cols) / insured_years``, and splits
temporally into train (<=2020) and test (>=2021).

The test years (2021-2023) are *never* used to fit anything. Imputation
statistics (lat/long medians) are computed on the training rows only and
applied to both splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """Train/test feature matrices, all target rates, and the sample weight.

    Attributes
    ----------
    X_train, X_test:
        Structural feature frames (columns == ``config.FEATURES``).
    y_train, y_test:
        DataFrames of target *rates* (euros per insured year); one column per
        target key in ``config.TARGETS`` (``"total"`` + 23 categories).
    w_train, w_test:
        ``insured_years`` per row — the sample weight for fitting and metrics.
    meta_train, meta_test:
        Identifier columns (year, postal_code_3, gementee, gementee_code,
        age_group, gender) kept for disaggregated evaluation, NOT used as
        model inputs beyond what already lives in ``X``.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    w_train: pd.Series
    w_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame


_META_COLUMNS = [
    "year", "postal_code_3", "gementee", "gementee_code", "age_group", "gender",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw() -> pd.DataFrame:
    """Read the full ``claims`` table into a DataFrame (read-only)."""
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        # Load features + cost columns + the weight + every identifier column
        # used for disaggregation/metadata (some, e.g. gementee_code, are no
        # longer model features but are still needed by the dashboard).
        needed = sorted(
            set(config.FEATURES) | set(config.COST_COLUMNS)
            | {config.WEIGHT_COLUMN} | set(_META_COLUMNS)
        )
        cols = ", ".join(needed)
        df = con.execute(f"SELECT {cols} FROM claims").fetchdf()
    finally:
        con.close()
    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every target rate = sum(cost cols) / insured_years."""
    iy = df[config.WEIGHT_COLUMN].to_numpy(dtype=float)
    rates: dict[str, np.ndarray] = {}
    for key, (_label, cost_cols) in config.TARGETS.items():
        numerator = df[cost_cols].sum(axis=1).to_numpy(dtype=float)
        rates[key] = numerator / iy
    return pd.DataFrame(rates, index=df.index)


def _prepare_features(df: pd.DataFrame, lat_med: float, lon_med: float) -> pd.DataFrame:
    """Return a clean structural feature frame with nulls handled.

    Categorical nulls (gementee_code) become ``UNKNOWN``; lat/long nulls are
    imputed with the supplied *training* medians.
    """
    X = df[config.FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna(config.UNKNOWN_CATEGORY).astype(str)
    X["latitude"] = X["latitude"].fillna(lat_med)
    X["longitude"] = X["longitude"].fillna(lon_med)
    X["number_of_insured"] = X["number_of_insured"].astype(float)
    X["insured_years"] = X["insured_years"].astype(float)
    X["year"] = X["year"].astype(int)
    return X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset() -> Dataset:
    """Load claims, build targets/features, and split temporally.

    Returns a :class:`Dataset`. Imputation medians are fit on train rows only.
    """
    df = _load_raw()

    is_train = df["year"].isin(list(config.TRAIN_YEARS))
    is_test = df["year"].isin(list(config.TEST_YEARS))

    # Impute statistics from the training rows ONLY.
    lat_med = float(df.loc[is_train, "latitude"].median())
    lon_med = float(df.loc[is_train, "longitude"].median())

    X = _prepare_features(df, lat_med, lon_med)
    y = _build_targets(df)
    w = df[config.WEIGHT_COLUMN].astype(float)
    meta = df[_META_COLUMNS].copy()

    ds = Dataset(
        X_train=X.loc[is_train].reset_index(drop=True),
        X_test=X.loc[is_test].reset_index(drop=True),
        y_train=y.loc[is_train].reset_index(drop=True),
        y_test=y.loc[is_test].reset_index(drop=True),
        w_train=w.loc[is_train].reset_index(drop=True),
        w_test=w.loc[is_test].reset_index(drop=True),
        meta_train=meta.loc[is_train].reset_index(drop=True),
        meta_test=meta.loc[is_test].reset_index(drop=True),
    )
    _assert_no_leakage(ds)
    return ds


def _assert_no_leakage(ds: Dataset) -> None:
    """Fail loudly if the split or feature set violates the leakage rules."""
    # 1. No cost column may appear among the features.
    leaked = sorted(set(config.COST_COLUMNS) & set(ds.X_train.columns))
    assert not leaked, f"Cost columns leaked into features: {leaked}"

    # 2. Train years are a subset of 2011-2020; test years of 2021-2023.
    train_years = set(ds.meta_train["year"].unique())
    test_years = set(ds.meta_test["year"].unique())
    assert train_years <= set(config.TRAIN_YEARS), f"Bad train years: {train_years}"
    assert test_years <= set(config.TEST_YEARS), f"Bad test years: {test_years}"
    assert train_years.isdisjoint(test_years), "Train/test years overlap"

    # 3. Weights must be strictly positive (rates are well-defined).
    assert (ds.w_train > 0).all() and (ds.w_test > 0).all(), "Non-positive weight"

    # 4. No residual nulls in the feature matrices.
    assert not ds.X_train.isnull().any().any(), "Null in X_train"
    assert not ds.X_test.isnull().any().any(), "Null in X_test"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load_dataset()
    print("Features:", list(ds.X_train.columns))
    print(f"Train rows: {len(ds.X_train):,}  | years {sorted(ds.meta_train['year'].unique())}")
    print(f"Test rows:  {len(ds.X_test):,}  | years {sorted(ds.meta_test['year'].unique())}")
    print(f"Targets ({ds.y_train.shape[1]}): {list(ds.y_train.columns)}")
    tot = ds.y_train["total"]
    wmean = np.average(tot, weights=ds.w_train)
    print(f"Total cost rate (train): weighted mean €{wmean:,.0f}, "
          f"min €{tot.min():,.0f}, max €{tot.max():,.0f}")
    print("Leakage assertions passed.")
