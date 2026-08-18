"""Data loading, temporal split, and target/feature construction.

Builds the structural feature matrix (no cost columns, so no target leakage),
derives every target rate as sum(cost cols) / insured_years, and splits at
2020/2021. Nothing is fitted on the test years — the lat/long imputation
medians come from the training rows and are applied to both splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Dataset:
    """Train/test feature matrices, all target rates, and the sample weight.

    ``y`` holds one column of rates (euros per insured year) per key in
    ``config.TARGETS``; ``w`` is insured_years; ``meta`` carries the identifier
    columns for disaggregated evaluation and is not a model input.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    w_train: pd.Series
    w_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame


def _load_raw() -> pd.DataFrame:
    """Read the full ``claims`` table into a DataFrame (read-only)."""
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        needed = sorted(
            set(config.FEATURES) | set(config.COST_COLUMNS)
            | {config.WEIGHT_COLUMN} | set(config.META_COLUMNS)
        )
        cols = ", ".join(needed)
        df = con.execute(f"SELECT {cols} FROM claims").fetchdf()
    finally:
        con.close()
    return df


def cost_rate(df: pd.DataFrame, cost_cols: list[str]) -> np.ndarray:
    """Rate in euros per insured-year: ``sum(cost_cols) / insured_years``.

    The only definition of a cost rate in the project; the denominator is
    always insured_years, never number_of_insured.
    """
    numerator = df[cost_cols].sum(axis=1).to_numpy(dtype=float)
    return numerator / df[config.WEIGHT_COLUMN].to_numpy(dtype=float)


def total_rate(df: pd.DataFrame) -> np.ndarray:
    """Total real cost rate — the primary RQ2 target."""
    return cost_rate(df, config.COST_COLUMNS)


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every target rate = sum(cost cols) / insured_years."""
    rates = {
        key: cost_rate(df, cost_cols)
        for key, (_label, cost_cols) in config.TARGETS.items()
    }
    return pd.DataFrame(rates, index=df.index)


def prepare_features(df: pd.DataFrame, lat_med: float, lon_med: float,
                     extra_numeric: list[str] = ()) -> pd.DataFrame:
    """Return a clean structural feature frame with nulls handled.

    lat/long nulls take the supplied *training* medians. *extra_numeric*
    appends further float columns; the temporal track passes its t-1 lag there
    so both tracks clean their inputs the same way.
    """
    extra_numeric = list(extra_numeric)
    X = df[config.FEATURES + extra_numeric].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna(config.UNKNOWN_CATEGORY).astype(str)
    X["latitude"] = X["latitude"].fillna(lat_med)
    X["longitude"] = X["longitude"].fillna(lon_med)
    X["number_of_insured"] = X["number_of_insured"].astype(float)
    X["insured_years"] = X["insured_years"].astype(float)
    X["year"] = X["year"].astype(int)
    for col in extra_numeric:
        X[col] = X[col].astype(float)
    return X


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    """Load claims, build targets/features, and split temporally.

    Cached, since run_all drives four stages that each want the dataset and
    rebuilding re-reads ~380k rows. Treat the returned frames as read-only.
    """
    df = _load_raw()

    is_train = df["year"].isin(list(config.TRAIN_YEARS))
    is_test = df["year"].isin(list(config.TEST_YEARS))

    # Training rows only.
    lat_med = float(df.loc[is_train, "latitude"].median())
    lon_med = float(df.loc[is_train, "longitude"].median())

    X = prepare_features(df, lat_med, lon_med)
    y = _build_targets(df)
    w = df[config.WEIGHT_COLUMN].astype(float)
    meta = df[config.META_COLUMNS].copy()

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
    leaked = sorted(set(config.COST_COLUMNS) & set(ds.X_train.columns))
    assert not leaked, f"Cost columns leaked into features: {leaked}"

    train_years = set(ds.meta_train["year"].unique())
    test_years = set(ds.meta_test["year"].unique())
    assert train_years <= set(config.TRAIN_YEARS), f"Bad train years: {train_years}"
    assert test_years <= set(config.TEST_YEARS), f"Bad test years: {test_years}"
    assert train_years.isdisjoint(test_years), "Train/test years overlap"

    # Strictly positive weights keep the rates well-defined.
    assert (ds.w_train > 0).all() and (ds.w_test > 0).all(), "Non-positive weight"

    assert not ds.X_train.isnull().any().any(), "Null in X_train"
    assert not ds.X_test.isnull().any().any(), "Null in X_test"


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
