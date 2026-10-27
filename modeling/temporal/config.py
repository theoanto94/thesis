"""Configuration for the temporal forecasting track.

One target, one lag, four models, and one question: does dropping the COVID
years from training improve forecasts? Extends the Phase 2 config (``base``).
"""

from __future__ import annotations

from .. import config as base

# One panel cell per combination, observed yearly.
ENTITY_KEYS: list[str] = ["postal_code_3", "age_group", "gender"]

TARGET: str = base.TARGET_TOTAL
LAG_COL: str = "rate_total_lag1"

# persistence carries the last observed rate forward — the honest floor. The
# rest reuse their Phase 2 tuned params; the question here is the COVID
# treatment, not model capacity.
TEMPORAL_MODELS: list[str] = ["persistence", "ridge", "lightgbm", "xgboost"]

# persistence has nothing to fit.
FITTED_MODELS: list[str] = ["ridge", "lightgbm", "xgboost"]

# base keeps every training year; no_covid drops the pandemic-distorted ones.
TREATMENTS: tuple[str, ...] = ("base", "no_covid")

# 2020 is the shock year (deferred elective care, -4.42% national real rate);
# 2021 is still below the 2019 level.
COVID_TARGET_YEARS: tuple[int, ...] = (2020, 2021)

TRAIN_START_TARGET_YEAR: int = 2012   # the 2011 -> 2012 transition

# Mirrors the Phase 2 split (train 2011-2020, test 2021-2023).
TRAIN_END: int = 2020


def train_target_years(treatment: str, train_end: int = TRAIN_END) -> tuple[int, ...]:
    """Target years a *treatment* trains on, up to *train_end* inclusive.

    A training row is a one-year transition, rate at ``y - 1`` onto rate at
    ``y``. no_covid drops those target years; lag values are untouched, since
    last year's actual rate is always known at prediction time.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment!r}")
    years = range(TRAIN_START_TARGET_YEAR, train_end + 1)
    if treatment == "no_covid":
        return tuple(y for y in years if y not in COVID_TARGET_YEARS)
    return tuple(years)


# Both protocols are recursive: predictions feed the next year's lag.

# A — the structural mirror. Base treatment only, seeded from actual 2020
# rates, scoring the same hold-out the Phase 2 pipeline uses.
MIRROR_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023)
MIRROR_SEED_YEAR: int = 2020

# B — the head-to-head that selects the winner. Same years, same seed, so
# only the training set differs. 2021 is excluded from scoring.
H2H_EVAL_YEARS: tuple[int, ...] = (2022, 2023)
H2H_SEED_YEAR: int = 2021

# An entity must be fully observed across this window to be scored. One fixed
# window for the whole suite, so every protocol scores identical rows and no
# recursive chain is broken.
EVAL_HISTORY_YEARS: tuple[int, ...] = tuple(
    range(MIRROR_SEED_YEAR, max(MIRROR_EVAL_YEARS) + 1)
)

FORECAST_YEARS: range = range(2024, 2031)
FORECAST_BASE_YEAR: int = 2023

# Exposure path of each 2023 cell (see population.py). The deployed
# projection grows every (age band x gender) by its own CBS kernprognose
# factor, so the age mix ages as CBS projects.
EXPOSURE_SCENARIO: str = "cbs_age_sex"

# Realized 2011-2023 CAGR of the national real rate, reported beside the
# projected CAGR as a plausibility check — never a gating rule. Part of it is
# coverage, not growth: on the categories present in every year it is +1.23%
# (modeling.coverage_audit, artifacts/coverage/growth.json).
REALIZED_CAGR: float = 0.0210
REALIZED_CAGR_FIXED_BASKET: float = 0.0123

# Pinned rather than taken from the leaderboard. The accuracy leader
# (LightGBM/no_covid, RMSE €453) saturates under recursion — trees cannot
# predict above the training cost levels, giving a +0.89%/yr projected CAGR
# under the CBS age x sex exposure (+0.34%/yr at a constant age mix) against a
# realized +2.10%. Ridge/base is 3.4% worse on RMSE but extrapolates
# a plausible growth path. Both configs are reported in the experiments.
DEPLOYED_MODEL: str = "ridge"
DEPLOYED_TREATMENT: str = "base"

# Created by forecast.py, read by app.py.
FORECAST_TABLE: str = "claims_forecast"
EXTENDED_VIEW: str = "claims_extended"

TEMPORAL_DIR = base.ARTIFACTS / "temporal"
T_METRICS_DIR = TEMPORAL_DIR / "metrics"
T_MODELS_DIR = TEMPORAL_DIR / "models"
T_FORECAST_DIR = TEMPORAL_DIR / "forecast"
# Row-level recursive predictions, one file per (protocol, treatment, model);
# read by evaluate_extra.py for bootstrap intervals and per-year bias.
T_PREDICTIONS_DIR = TEMPORAL_DIR / "predictions"

_ALL_DIRS = (TEMPORAL_DIR, T_METRICS_DIR, T_MODELS_DIR, T_FORECAST_DIR, T_PREDICTIONS_DIR)

EXPERIMENTS_PATH = T_METRICS_DIR / "experiments.parquet"
WINNER_PATH = T_METRICS_DIR / "winner.json"
FORECAST_PATH = T_FORECAST_DIR / "forecast.parquet"
FORECAST_SUMMARY_PATH = T_FORECAST_DIR / "forecast_summary.json"


def ensure_dirs() -> None:
    """Create the temporal artifact directory tree if it does not exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def prediction_path(protocol: str, treatment: str, model_key: str):
    """Row-level recursive predictions of one evaluated config."""
    return T_PREDICTIONS_DIR / f"{protocol}__{treatment}__{model_key}.parquet"


def final_model_path(model_key: str, treatment: str):
    """Path of the refit-through-2023 estimator used for the projection."""
    return T_MODELS_DIR / f"final_{model_key}__{treatment}.joblib"
