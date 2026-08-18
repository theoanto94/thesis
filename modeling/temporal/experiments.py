"""The COVID-treatment experiment: fit, recursive evaluation, winner selection.

One experimental question — does removing the COVID target years (2020, 2021)
from training improve forecasts? — answered on a fixed grid:

* Models: persistence (last observed rate carried forward), Ridge, LightGBM,
  XGBoost. Fitted models reuse their Phase 2 tuned hyperparameters
  (``artifacts/params/<model>.json``); persistence has none.
* Treatments: ``base`` (train target years 2012-2020) vs ``no_covid``
  (2020 and 2021 dropped). 7 fits total: 3 models x 2 treatments + none.
* Protocols (both recursive — each year's prediction feeds the next year's
  lag, exactly as the 2024-2030 projection operates):

  - **structural_mirror** (headline, base treatment + persistence): seed from
    actual 2020, predict 2021 -> 2022 -> 2023. Mirrors the Phase 2 split.
  - **head_to_head** (winner selection): seed from actual 2021, predict
    2022 -> 2023, identical for every candidate — only the training set
    differs, which isolates the treatment effect. 2021 itself is excluded
    from scoring per the COVID-removal scenario.

Every protocol scores the same ~29k entities (observed 2020-2023 without
gaps), insured_years-weighted. Winner = lowest weighted RMSE on the
head-to-head protocol; written to ``winner.json`` with the full leaderboard.

Usage::

    uv run python -m modeling.temporal.experiments
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)
from xgboost import XGBRegressor

from .. import config as base
from ..factory import apply_best_params
from ..metrics import METRIC_NAMES, weighted_metrics
from ..tune import load_best_params
from . import config as tcfg
from .data import (
    META_COLUMNS,
    RATE_COL,
    TemporalDataset,
    build_panel,
    entity_index,
    eval_entities,
    load_train,
    prepare_X,
    rates_for_year,
    train_medians,
)

#: model key -> encoder family (all fitted models are sklearn Pipelines).
_FAMILY = {"ridge": "linear", "lightgbm": "tree", "xgboost": "tree"}

#: Protocol names as they appear in the metrics table.
PROTOCOL_MIRROR = "structural_mirror"
PROTOCOL_H2H = "head_to_head"


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def make_preprocessor(family: str) -> ColumnTransformer:
    """Phase 2 preprocessor with the lag column added to the numerics.

    Standalone rather than reusing ``encoders.make_preprocessor``, which
    hard-codes ``config.NUMERIC_FEATURES`` (remainder="drop") and would
    silently drop the lag feature. ``"linear"``: target-encode+scale
    postcode3, one-hot age/gender, scale all numerics (structural + lag).
    ``"tree"``: ordinal-encode categoricals, pass numerics through.
    """
    numeric = base.NUMERIC_FEATURES + [tcfg.LAG_COL]
    if family == "linear":
        geo_pipe = Pipeline([
            ("target", TargetEncoder(smooth="auto", random_state=base.RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("geo_target", geo_pipe, base.HIGH_CARD_FEATURES),
                (
                    "cat_ohe",
                    OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False),
                    base.LOW_CARD_FEATURES,
                ),
                ("num_scale", StandardScaler(), numeric),
            ],
            remainder="drop",
        )
    if family == "tree":
        return ColumnTransformer(
            transformers=[
                (
                    "cat_ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1),
                    base.CATEGORICAL_FEATURES,
                ),
                ("num_pass", "passthrough", numeric),
            ],
            remainder="drop",
        )
    raise ValueError(f"Unknown encoder family {family!r}")


def _raw_estimator(model_key: str, params: dict):
    """Bare estimator with the given hyperparameters (Phase 2 constructors)."""
    if model_key == "ridge":
        return Ridge(random_state=base.RANDOM_STATE, **params)
    if model_key == "lightgbm":
        return LGBMRegressor(random_state=base.RANDOM_STATE, n_jobs=-1,
                             verbosity=-1, **params)
    if model_key == "xgboost":
        return XGBRegressor(random_state=base.RANDOM_STATE, n_jobs=-1,
                            tree_method="hist", objective="reg:squarederror",
                            **params)
    raise ValueError(f"Unknown temporal model {model_key!r}")


def build(model_key: str) -> Pipeline:
    """Unfitted Pipeline (lag-aware preprocessor + Phase 2 tuned estimator)."""
    params = apply_best_params(model_key, load_best_params(model_key))
    return Pipeline([
        ("pre", make_preprocessor(_FAMILY[model_key])),
        ("model", _raw_estimator(model_key, params)),
    ])


class _Persistence:
    """Carries the incoming lag forward unchanged: prediction = last rate.

    Run through the same recursive loop as the fitted models, its own output
    feeds its own lag, so at step k the prediction is the seed-year actual —
    the classic random-walk floor.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[tcfg.LAG_COL].to_numpy(dtype=float)


def fit(model_key: str, ds: TemporalDataset) -> tuple[Pipeline, float]:
    """Fit one (model, treatment) config; returns (pipeline, seconds)."""
    est = build(model_key)
    t0 = time.perf_counter()
    est.fit(ds.X_train, ds.y_train, model__sample_weight=ds.w_train)
    return est, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Recursive evaluation
# ---------------------------------------------------------------------------

def recursive_predict(
    estimator,
    eval_years: tuple[int, ...],
    seed_year: int,
    lat_med: float,
    lon_med: float,
) -> pd.DataFrame:
    """Roll *estimator* through *eval_years*, feeding predictions forward.

    The chain is seeded with the actual *seed_year* rates of the common
    evaluation entities; from the second scored year on, the lag input is the
    estimator's own previous prediction (clipped at zero), exactly as the
    2024-2030 projection operates. Returns one long frame with the meta
    columns, ``y_true``, ``y_pred``, ``weight``, and ``recursion_step``.
    """
    panel = build_panel()
    entities = eval_entities()
    history = rates_for_year(panel, seed_year).reindex(entities)
    assert not history.isnull().any(), f"Missing seed rates for {seed_year}"

    blocks: list[pd.DataFrame] = []
    for step, year in enumerate(sorted(eval_years), start=1):
        rows = panel[panel["year"] == year]
        rows = rows.loc[entity_index(rows).isin(entities)].copy()
        assert len(rows) == len(entities), f"Broken chain at {year}"
        rows[tcfg.LAG_COL] = history.reindex(entity_index(rows)).to_numpy()

        X = prepare_X(rows, lat_med, lon_med)
        y_pred = np.clip(estimator.predict(X), 0.0, None)
        history = pd.Series(y_pred, index=entity_index(rows))

        block = rows[META_COLUMNS].copy()
        block["y_true"] = rows[RATE_COL].to_numpy(dtype=float)
        block["y_pred"] = y_pred
        block["weight"] = rows[base.WEIGHT_COLUMN].to_numpy(dtype=float)
        block["recursion_step"] = step
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def score(pred: pd.DataFrame) -> list[dict]:
    """Weighted metrics overall (``year="all"``) and per scored year."""
    records = [{
        "year": "all", "recursion_step": pd.NA,
        "n_test": len(pred),
        **weighted_metrics(pred["y_true"], pred["y_pred"], pred["weight"]),
    }]
    for year, sub in pred.groupby("year", sort=True):
        records.append({
            "year": str(year), "recursion_step": int(sub["recursion_step"].iloc[0]),
            "n_test": len(sub),
            **weighted_metrics(sub["y_true"], sub["y_pred"], sub["weight"]),
        })
    return records


# ---------------------------------------------------------------------------
# The experiment matrix
# ---------------------------------------------------------------------------

def _assert_protocol_ok(train_years: set[int], eval_years: tuple[int, ...]) -> None:
    """Scored years must never have been training targets."""
    overlap = train_years & set(eval_years)
    assert not overlap, f"Train/eval years overlap: {sorted(overlap)}"


def run_matrix() -> pd.DataFrame:
    """Fit and evaluate every (model, treatment) under its protocols."""
    records: list[dict] = []
    default_medians = train_medians(build_panel(), tcfg.TRAIN_END)

    def add(protocol, treatment, model_key, estimator, ds, seconds):
        seed, years = {
            PROTOCOL_MIRROR: (tcfg.MIRROR_SEED_YEAR, tcfg.MIRROR_EVAL_YEARS),
            PROTOCOL_H2H: (tcfg.H2H_SEED_YEAR, tcfg.H2H_EVAL_YEARS),
        }[protocol]
        if ds is not None:
            _assert_protocol_ok(set(ds.meta_train["year"]), years)
        lat_med, lon_med = (
            (ds.lat_med, ds.lon_med) if ds is not None else default_medians
        )
        pred = recursive_predict(estimator, years, seed, lat_med, lon_med)
        for rec in score(pred):
            records.append({
                "protocol": protocol, "treatment": treatment,
                "model": model_key,
                "n_train": len(ds.X_train) if ds is not None else 0,
                "train_seconds": seconds, **rec,
            })

    # Persistence: no training set, both protocols.
    persistence = _Persistence()
    add(PROTOCOL_MIRROR, "none", "persistence", persistence, None, 0.0)
    add(PROTOCOL_H2H, "none", "persistence", persistence, None, 0.0)

    for treatment in tcfg.TREATMENTS:
        ds = load_train(treatment)
        for model_key in tcfg.FITTED_MODELS:
            print(f"Fitting {model_key} / {treatment} "
                  f"({len(ds.X_train):,} rows)...")
            est, seconds = fit(model_key, ds)
            if treatment == "base":
                add(PROTOCOL_MIRROR, treatment, model_key, est, ds, seconds)
            add(PROTOCOL_H2H, treatment, model_key, est, ds, seconds)

    cols = ["protocol", "treatment", "model", "year", "recursion_step",
            *METRIC_NAMES, "n_train", "n_test", "train_seconds"]
    return pd.DataFrame(records)[cols]


def select_winner(metrics: pd.DataFrame) -> dict:
    """Rank on head-to-head weighted RMSE; the argmin is the winner.

    Persistence sits in the leaderboard as the honest floor — if it wins,
    that is the result, and the forecast degenerates to a carry-forward.
    """
    board = (
        metrics[(metrics["protocol"] == PROTOCOL_H2H) & (metrics["year"] == "all")]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    leaderboard = [
        {"model": r["model"], "treatment": r["treatment"],
         **{m: round(float(r[m]), 4) for m in METRIC_NAMES}}
        for _, r in board.iterrows()
    ]
    top = leaderboard[0]
    return {
        "model": top["model"],
        "treatment": top["treatment"],
        "rmse": top["rmse"],
        "selection": ("recursive weighted RMSE, common 2022-2023 protocol "
                      "seeded from actual 2021"),
        "leaderboard": leaderboard,
    }


def _print_table(metrics: pd.DataFrame, protocol: str, title: str) -> None:
    view = metrics[(metrics["protocol"] == protocol) & (metrics["year"] == "all")]
    view = view.sort_values("rmse")[
        ["model", "treatment", *METRIC_NAMES, "n_train", "train_seconds"]
    ]
    print(f"\n{title}")
    print(view.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def main() -> None:
    """Run the experiment matrix; write ``experiments.parquet`` + ``winner.json``."""
    tcfg.ensure_dirs()
    metrics = run_matrix()
    metrics.to_parquet(tcfg.EXPERIMENTS_PATH, index=False)

    winner = select_winner(metrics)
    tcfg.WINNER_PATH.write_text(json.dumps(winner, indent=2))

    _print_table(metrics, PROTOCOL_MIRROR,
                 "Structural mirror (train 2011-2020, recursive 2021-2023):")
    _print_table(metrics, PROTOCOL_H2H,
                 "Head-to-head (common recursive 2022-2023, seeded from 2021):")
    print(f"\nWinner: {winner['model']} / {winner['treatment']} "
          f"(weighted RMSE €{winner['rmse']:,.1f})")
    print(f"Wrote {tcfg.EXPERIMENTS_PATH} and {tcfg.WINNER_PATH}")


if __name__ == "__main__":
    main()
