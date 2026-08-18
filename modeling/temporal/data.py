"""Panel construction, the validated t-1 lag, and the treatment-aware split.

The panel sits at the (entity x year) grain, with rate_total and its one-year
lag. The lag comes from an exact self-merge on (entity, year - 1) and is NULL
where that year is absent — a groupby().shift() would silently hand back the
previous *row* across year gaps, and ~500 entities have them.

The lagged realized rate is a deliberate feature: last year's outcome is known
at prediction time, so it is history rather than leakage. The rules that do
apply are asserted in _assert_temporal_ok.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config as base
from ..data import _load_raw, prepare_features, total_rate
from . import config as tcfg

RATE_COL = "rate_total"


def add_lag_features(
    df: pd.DataFrame, value_cols: dict[str, str], max_depth: int
) -> pd.DataFrame:
    """Append exact year-validated lag columns to an entity-year frame.

    ``<lag_name>_lag<k>`` at year t is the entity's value at t-k, or NaN when
    that exact year is unobserved. Needs a unique (entity, year) grain.
    """
    key_cols = tcfg.ENTITY_KEYS + ["year"]
    source = df[key_cols + list(value_cols.values())]
    for k in range(1, max_depth + 1):
        right = source.copy()
        right["year"] = right["year"] + k     # value of year t-k lands at year t
        right.columns = key_cols + [f"{name}_lag{k}" for name in value_cols]
        df = df.merge(right, on=key_cols, how="left", validate="one_to_one")
    return df


@lru_cache(maxsize=1)
def build_panel() -> pd.DataFrame:
    """Full claims panel with the total rate and its validated t-1 lag.

    One row per entity-year. The raw per-category cost columns are kept for
    the forecast's constant-composition allocation. Cached, so treat the
    returned frame as read-only.
    """
    df = _load_raw()
    df[RATE_COL] = total_rate(df)

    df = df.sort_values(tcfg.ENTITY_KEYS + ["year"], kind="mergesort").reset_index(drop=True)
    return add_lag_features(df, {RATE_COL: RATE_COL}, max_depth=1)


def entity_index(df: pd.DataFrame) -> pd.MultiIndex:
    """MultiIndex of the entity key columns for row alignment."""
    return pd.MultiIndex.from_frame(df[tcfg.ENTITY_KEYS])


@lru_cache(maxsize=1)
def eval_entities() -> pd.MultiIndex:
    """Entities observed in every year of ``tcfg.EVAL_HISTORY_YEARS``.

    Membership guarantees a complete lag for every scored row and an unbroken
    recursive chain, so every protocol and treatment scores identical rows.
    """
    panel = build_panel()
    window = panel[panel["year"].isin(list(tcfg.EVAL_HISTORY_YEARS))]
    counts = window.groupby(tcfg.ENTITY_KEYS, sort=False)["year"].nunique()
    full = counts[counts == len(tcfg.EVAL_HISTORY_YEARS)]
    return pd.MultiIndex.from_tuples(full.index, names=tcfg.ENTITY_KEYS)


def rates_for_year(panel: pd.DataFrame, year: int) -> pd.Series:
    """Actual total rates of one year, indexed by entity."""
    rows = panel[panel["year"] == year]
    return pd.Series(rows[RATE_COL].to_numpy(), index=entity_index(rows))


def train_medians(panel: pd.DataFrame, train_end: int) -> tuple[float, float]:
    """Lat/long imputation medians fit on years <= *train_end* only."""
    train = panel[panel["year"] <= train_end]
    return float(train["latitude"].median()), float(train["longitude"].median())


def prepare_X(rows: pd.DataFrame, lat_med: float, lon_med: float) -> pd.DataFrame:
    """Clean feature frame: structural features + the t-1 lag.

    Deliberately Phase 2's cleaner, so the two tracks cannot drift apart. Lag
    values must already be complete — recursive callers overwrite them first.
    """
    return prepare_features(rows, lat_med, lon_med, extra_numeric=[tcfg.LAG_COL])


@dataclass(frozen=True)
class TemporalDataset:
    """Training split for one COVID treatment; evaluation is recursive-only.

    The imputation medians are exposed so recursive and forecast feature
    assembly stays consistent with training.
    """

    treatment: str
    train_end: int
    X_train: pd.DataFrame
    y_train: pd.Series
    w_train: pd.Series
    meta_train: pd.DataFrame
    lat_med: float
    lon_med: float


def load_train(treatment: str, train_end: int = tcfg.TRAIN_END) -> TemporalDataset:
    """Build the training set for one COVID *treatment*.

    Rows need a present t-1 lag, so the first panel year and gapped entities
    drop out. The excluded COVID years stay in the panel as lag history; they
    are just never the thing being predicted.
    """
    panel = build_panel()
    lat_med, lon_med = train_medians(panel, train_end)

    years = tcfg.train_target_years(treatment, train_end)
    is_train = panel["year"].isin(years) & panel[tcfg.LAG_COL].notna()
    rows = panel.loc[is_train]

    ds = TemporalDataset(
        treatment=treatment,
        train_end=train_end,
        X_train=prepare_X(rows, lat_med, lon_med).reset_index(drop=True),
        y_train=rows[RATE_COL].astype(float).reset_index(drop=True),
        w_train=rows[base.WEIGHT_COLUMN].astype(float).reset_index(drop=True),
        meta_train=rows[base.META_COLUMNS].copy().reset_index(drop=True),
        lat_med=lat_med,
        lon_med=lon_med,
    )
    _assert_temporal_ok(ds)
    return ds


def _assert_temporal_ok(ds: TemporalDataset) -> None:
    """Fail loudly if the training split or feature set is malformed."""
    # Only the validated lag may carry cost-derived signal.
    leaked = sorted(set(base.COST_COLUMNS) & set(ds.X_train.columns))
    assert not leaked, f"Raw cost columns leaked into features: {leaked}"
    expected = set(base.FEATURES) | {tcfg.LAG_COL}
    assert set(ds.X_train.columns) == expected, "Unexpected feature columns"

    # Under no_covid the COVID years may not appear at all.
    declared = set(tcfg.train_target_years(ds.treatment, ds.train_end))
    train_years = set(ds.meta_train["year"].unique())
    assert train_years == declared, (
        f"Train years {sorted(train_years)} != declared "
        f"{sorted(declared)} for treatment {ds.treatment!r}"
    )

    assert (ds.w_train > 0).all(), "Non-positive train weight"
    assert not ds.X_train.isnull().any().any(), "Null in X_train"
    assert not ds.y_train.isnull().any(), "Null in y_train"


if __name__ == "__main__":
    panel = build_panel()
    cov = panel[tcfg.LAG_COL].notna()
    print(f"Panel rows: {len(panel):,} | rows with a t-1 lag: {cov.sum():,} "
          f"({cov.mean():.1%})")
    hist = tcfg.EVAL_HISTORY_YEARS
    print(f"Eval entities (full {min(hist)}-{max(hist)}): {len(eval_entities()):,}")
    for treatment in tcfg.TREATMENTS:
        ds = load_train(treatment)
        rate = np.average(ds.y_train, weights=ds.w_train)
        print(f"  {treatment}: train {len(ds.X_train):,} rows "
              f"(years {sorted(ds.meta_train['year'].unique())}) | "
              f"weighted mean rate €{rate:,.0f}")
    print("Temporal assertions passed.")
