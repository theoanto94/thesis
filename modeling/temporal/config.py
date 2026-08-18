"""Configuration for the temporal forecasting track.

One target (the total real cost rate), one lag (t-1), four models, and one
experimental question: does removing the COVID years from training improve
forecasts? Extends the Phase 2 config (imported as ``base``) with the
treatment definitions, the two evaluation protocols, the forecast horizon,
and the DuckDB objects that carry the projections into the EDA dashboard.
Importing this module performs no I/O.
"""

from __future__ import annotations

from .. import config as base

# ---------------------------------------------------------------------------
# Panel & target
# ---------------------------------------------------------------------------

#: The panel entity — one cell per combination, observed yearly.
ENTITY_KEYS: list[str] = ["postal_code_3", "age_group", "gender"]

#: The single modelled target: total real cost per insured-year.
TARGET: str = base.TARGET_TOTAL

#: The one lag feature — the entity's total rate exactly one year earlier.
LAG_COL: str = "rate_total_lag1"

# ---------------------------------------------------------------------------
# Models & treatments
# ---------------------------------------------------------------------------

#: Models in the comparison. ``persistence`` carries the last observed rate
#: forward unchanged (the honest floor); the rest reuse their Phase 2 tuned
#: hyperparameters from ``artifacts/params/<model>.json`` (no re-tuning: the
#: question is the COVID treatment, not model capacity).
TEMPORAL_MODELS: list[str] = ["persistence", "ridge", "lightgbm", "xgboost"]

#: Models that are actually fitted (persistence has no parameters).
FITTED_MODELS: list[str] = ["ridge", "lightgbm", "xgboost"]

#: The two COVID treatments compared head-to-head. ``base`` keeps every
#: training year; ``no_covid`` drops the pandemic-distorted target years.
TREATMENTS: tuple[str, ...] = ("base", "no_covid")

#: Target years removed from training under the ``no_covid`` treatment.
#: 2020 is the shock year (deferred elective care, -4.42% national real rate);
#: 2021 is the depressed year that follows, still below the 2019 level.
COVID_TARGET_YEARS: tuple[int, ...] = (2020, 2021)

#: First possible training target year (the 2011 -> 2012 transition).
TRAIN_START_TARGET_YEAR: int = 2012

#: Last training target year for the experiments — mirrors the Phase 2
#: structural split (train 2011-2020, test 2021-2023).
TRAIN_END: int = 2020


def train_target_years(treatment: str, train_end: int = TRAIN_END) -> tuple[int, ...]:
    """Target years a *treatment* trains on, up to *train_end* inclusive.

    A training row is a one-year transition: it maps the entity's rate at
    ``y - 1`` onto the rate at ``y``. ``no_covid`` removes the COVID target
    years from that set; the remaining years' *lag values* are untouched —
    last year's actual rate is always known at prediction time.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment!r}")
    years = range(TRAIN_START_TARGET_YEAR, train_end + 1)
    if treatment == "no_covid":
        return tuple(y for y in years if y not in COVID_TARGET_YEARS)
    return tuple(years)


# ---------------------------------------------------------------------------
# Evaluation protocols (both recursive: predictions feed the next year's lag)
# ---------------------------------------------------------------------------

#: Protocol A — the structural mirror. Trains through ``TRAIN_END`` (base
#: treatment only), seeds the recursive chain from the actual 2020 rates, and
#: scores 2021 -> 2022 -> 2023: the same hold-out the Phase 2 pipeline uses.
MIRROR_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023)
MIRROR_SEED_YEAR: int = 2020

#: Protocol B — the head-to-head that selects the winner. Both treatments
#: predict the same years (2022, 2023) from the same seed (actual 2021), so
#: only the training set differs; 2021 itself is excluded from scoring per
#: the COVID-removal scenario.
H2H_EVAL_YEARS: tuple[int, ...] = (2022, 2023)
H2H_SEED_YEAR: int = 2021

#: Window an entity must be fully observed in to enter any evaluation set:
#: from the earliest seed year through the last scored year. One fixed window
#: for the whole suite, so every protocol/treatment scores identical rows and
#: every recursive chain is unbroken.
EVAL_HISTORY_YEARS: tuple[int, ...] = tuple(
    range(MIRROR_SEED_YEAR, max(MIRROR_EVAL_YEARS) + 1)
)

# ---------------------------------------------------------------------------
# Forecast horizon & population scenario
# ---------------------------------------------------------------------------

FORECAST_YEARS: range = range(2024, 2031)       # 2024-2030 inclusive
FORECAST_BASE_YEAR: int = 2023

#: National annual population growth applied uniformly to every cell's 2023
#: exposure (CBS national trend, ~0.6%/yr). A deliberate scenario choice:
#: projections isolate cost-rate dynamics from local demographic shifts.
POPULATION_GROWTH_RATE: float = 0.006

#: The realized 2011-2023 CAGR of the national real cost rate. Reported next
#: to the projected CAGR as a plausibility diagnostic — never a selection or
#: gating rule.
REALIZED_CAGR: float = 0.0210

#: The configuration the 2024-2030 projection actually uses. Pinned rather
#: than derived from the leaderboard: the accuracy leader (LightGBM/no_covid,
#: RMSE €453) saturates under recursion — its projected CAGR is +0.34%/yr
#: against a realized +2.10%/yr — because trees cannot predict above the cost
#: levels seen in training. Ridge/base (RMSE €469, +3.4% worse) extrapolates
#: its linear trend and keeps a plausible growth path. A small, stated
#: accuracy compromise for a projection that behaves like the history it
#: extends; both configs and the trade-off are reported in the experiments.
DEPLOYED_MODEL: str = "ridge"
DEPLOYED_TREATMENT: str = "base"

# ---------------------------------------------------------------------------
# DuckDB objects (created by forecast.py; read by app.py)
# ---------------------------------------------------------------------------

FORECAST_TABLE: str = "claims_forecast"
EXTENDED_VIEW: str = "claims_extended"

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

TEMPORAL_DIR = base.ARTIFACTS / "temporal"
T_METRICS_DIR = TEMPORAL_DIR / "metrics"
T_MODELS_DIR = TEMPORAL_DIR / "models"
T_FORECAST_DIR = TEMPORAL_DIR / "forecast"

_ALL_DIRS = (TEMPORAL_DIR, T_METRICS_DIR, T_MODELS_DIR, T_FORECAST_DIR)

EXPERIMENTS_PATH = T_METRICS_DIR / "experiments.parquet"
WINNER_PATH = T_METRICS_DIR / "winner.json"
FORECAST_PATH = T_FORECAST_DIR / "forecast.parquet"
FORECAST_SUMMARY_PATH = T_FORECAST_DIR / "forecast_summary.json"


def ensure_dirs() -> None:
    """Create the temporal artifact directory tree if it does not exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def final_model_path(model_key: str, treatment: str):
    """Path of the refit-through-2023 estimator used for the projection."""
    return T_MODELS_DIR / f"final_{model_key}__{treatment}.joblib"
