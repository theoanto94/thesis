"""Panel construction, the validated t-1 lag, and the treatment-aware split.

Builds the claims panel at the ``(postal_code_3, age_group, gender)`` x
``year`` grain with the single modelled target ``rate_total`` (total real
cost per insured-year) and its one-year lag ``rate_total_lag1``, derived via
an exact self-merge on ``(entity, year - 1)``: the lag is the entity's
observation exactly one year earlier, NULL if that year is absent. (A naive
``groupby().shift()`` would silently return the previous *row* across year
gaps — ~500 entities have them.)

Unlike the Phase 2 loader (``modeling.data``), the lagged *realized* rate is
a deliberate feature here — the previous year's outcome is known at
prediction time, so this is legitimate history, not leakage. The temporal
leakage rules that DO apply are asserted in :func:`_assert_temporal_ok`:
no raw (current-year) cost column may enter the features, training target
years must match the declared treatment exactly, and imputation statistics
come from training years alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config as base
from ..data import _load_raw
from . import config as tcfg

#: Identifier columns kept for disaggregated evaluation (same as Phase 2).
META_COLUMNS = [
    "year", "postal_code_3", "gementee", "gementee_code", "age_group", "gender",
]

#: The target column in the panel.
RATE_COL = "rate_total"


# ---------------------------------------------------------------------------
# Panel + lag
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame, value_cols: dict[str, str], max_depth: int
) -> pd.DataFrame:
    """Append exact year-validated lag columns to an entity-year frame.

    For each ``lag_name -> value_col`` in *value_cols* and each k in
    ``1..max_depth``, ``<lag_name>_lag<k>`` at year t is the entity's
    *value_col* at year t-k — matched by a self-merge on
    ``(entity, year - k)`` — and NaN when that exact year is unobserved.
    Requires the ``(entity, year)`` grain to be unique.
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

    One row per entity-year. Columns: the meta/structural claims columns, the
    raw per-category ``costs_*_real`` columns (kept for the forecast's
    constant-composition allocation), ``rate_total``, and ``rate_total_lag1``
    (NaN where the entity lacks an observation the year before).

    Cached; treat the returned frame as read-only (``.copy()`` before
    mutating).
    """
    df = _load_raw()
    iy = df[base.WEIGHT_COLUMN].to_numpy(dtype=float)
    df[RATE_COL] = df[base.COST_COLUMNS].sum(axis=1).to_numpy(dtype=float) / iy

    df = df.sort_values(tcfg.ENTITY_KEYS + ["year"], kind="mergesort").reset_index(drop=True)
    return add_lag_features(df, {RATE_COL: RATE_COL}, max_depth=1)


def entity_index(df: pd.DataFrame) -> pd.MultiIndex:
    """MultiIndex of the entity key columns for row alignment."""
    return pd.MultiIndex.from_frame(df[tcfg.ENTITY_KEYS])


@lru_cache(maxsize=1)
def eval_entities() -> pd.MultiIndex:
    """Entities observed in every year of ``tcfg.EVAL_HISTORY_YEARS``.

    Membership guarantees a complete lag for every scored row and an unbroken
    recursive chain from the earliest seed (2020) through the last scored year
    (2023), so every protocol and treatment scores identical rows.
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


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def train_medians(panel: pd.DataFrame, train_end: int) -> tuple[float, float]:
    """Lat/long imputation medians fit on years <= *train_end* only."""
    train = panel[panel["year"] <= train_end]
    return float(train["latitude"].median()), float(train["longitude"].median())


def prepare_X(rows: pd.DataFrame, lat_med: float, lon_med: float) -> pd.DataFrame:
    """Clean feature frame: structural features + the t-1 lag.

    Mirrors Phase 2's ``_prepare_features`` (UNKNOWN categoricals, median
    lat/long) and appends the lag column as float. Lag values must already be
    complete — recursive/forecast callers overwrite them before calling.
    """
    X = rows[base.FEATURES + [tcfg.LAG_COL]].copy()
    for col in base.CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna(base.UNKNOWN_CATEGORY).astype(str)
    X["latitude"] = X["latitude"].fillna(lat_med)
    X["longitude"] = X["longitude"].fillna(lon_med)
    X["number_of_insured"] = X["number_of_insured"].astype(float)
    X["insured_years"] = X["insured_years"].astype(float)
    X["year"] = X["year"].astype(int)
    X[tcfg.LAG_COL] = X[tcfg.LAG_COL].astype(float)
    return X


# ---------------------------------------------------------------------------
# Training set container + loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalDataset:
    """Training split for one COVID treatment (evaluation is recursive-only).

    ``X_train`` columns are ``base.FEATURES + [tcfg.LAG_COL]``; ``y_train``
    is the total rate; ``w_train`` is ``insured_years``. ``lat_med`` /
    ``lon_med`` are the train-only imputation medians, exposed so recursive
    and forecast feature assembly stays consistent with training.
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

    Train rows: target years in ``tcfg.train_target_years(treatment,
    train_end)`` whose t-1 lag is present (the first panel year has no
    history; gapped entities are dropped per-row). The excluded COVID years
    remain in the panel as lag history — they are simply never the thing
    being predicted during training.
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
        meta_train=rows[META_COLUMNS].copy().reset_index(drop=True),
        lat_med=lat_med,
        lon_med=lon_med,
    )
    _assert_temporal_ok(ds)
    return ds


def _assert_temporal_ok(ds: TemporalDataset) -> None:
    """Fail loudly if the training split or feature set is malformed."""
    # 1. Only the validated lag may carry cost-derived signal — never a raw
    #    (current-year) cost column.
    leaked = sorted(set(base.COST_COLUMNS) & set(ds.X_train.columns))
    assert not leaked, f"Raw cost columns leaked into features: {leaked}"
    expected = set(base.FEATURES) | {tcfg.LAG_COL}
    assert set(ds.X_train.columns) == expected, "Unexpected feature columns"

    # 2. Training target years must match the declared treatment exactly:
    #    under no_covid the COVID years may not appear at all.
    declared = set(tcfg.train_target_years(ds.treatment, ds.train_end))
    train_years = set(ds.meta_train["year"].unique())
    assert train_years == declared, (
        f"Train years {sorted(train_years)} != declared "
        f"{sorted(declared)} for treatment {ds.treatment!r}"
    )

    # 3. Positive weights, no residual nulls.
    assert (ds.w_train > 0).all(), "Non-positive train weight"
    assert not ds.X_train.isnull().any().any(), "Null in X_train"
    assert not ds.y_train.isnull().any(), "Null in y_train"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

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
