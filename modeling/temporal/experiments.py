"""The COVID-treatment experiment: fit, recursive evaluation, winner selection.

Seven configs — persistence, plus three fitted models under each of the two
treatments — evaluated under two recursive protocols, where each year's
prediction feeds the next year's lag just as the 2024-2030 projection does:

  structural_mirror  seed from actual 2020, predict 2021 -> 2022 -> 2023,
                     mirroring the Phase 2 split. Base treatment only.
  head_to_head       seed from actual 2021, predict 2022 -> 2023, identical
                     for every candidate, so only the training set differs.

Both score the same entities, insured-years-weighted. The head-to-head RMSE
picks the winner, written to winner.json with the full leaderboard.

    uv run python -m modeling.temporal.experiments
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .. import config as base, factory
from ..factory import apply_best_params
from ..metrics import METRIC_NAMES, weighted_metrics
from ..tune import load_best_params
from . import config as tcfg
from .data import (
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

PROTOCOL_MIRROR = "structural_mirror"
PROTOCOL_H2H = "head_to_head"

# Passed to factory.build so the shared preprocessor keeps the lag instead of
# dropping it as an undeclared column.
NUMERIC_FEATURES = base.NUMERIC_FEATURES + [tcfg.LAG_COL]


def build(model_key: str) -> Pipeline:
    """Unfitted Pipeline (lag-aware preprocessor + Phase 2 tuned estimator).

    Goes through factory.build, so the lag column is the only difference
    between the two tracks.
    """
    params = apply_best_params(model_key, load_best_params(model_key))
    return factory.build(model_key, params, numeric_features=NUMERIC_FEATURES)


class _Persistence:
    """Carries the incoming lag forward unchanged: prediction = last rate.

    In the recursive loop its own output feeds its own lag, so at every step
    the prediction is the seed-year actual — the random-walk floor.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[tcfg.LAG_COL].to_numpy(dtype=float)


def fit(model_key: str, ds: TemporalDataset) -> tuple[Pipeline, float]:
    """Fit one (model, treatment) config; returns (pipeline, seconds)."""
    est = build(model_key)
    t0 = time.perf_counter()
    factory.fit(est, ds.X_train, ds.y_train, ds.w_train)
    return est, time.perf_counter() - t0


def recursive_predict(
    estimator,
    eval_years: tuple[int, ...],
    seed_year: int,
    lat_med: float,
    lon_med: float,
) -> pd.DataFrame:
    """Roll *estimator* through *eval_years*, feeding predictions forward.

    Seeded with the actual *seed_year* rates; from the second scored year on
    the lag is the estimator's own previous prediction, clipped at zero.
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

        block = rows[base.META_COLUMNS].copy()
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
        pred.to_parquet(tcfg.prediction_path(protocol, treatment, model_key), index=False)
        for rec in score(pred):
            records.append({
                "protocol": protocol, "treatment": treatment,
                "model": model_key,
                "n_train": len(ds.X_train) if ds is not None else 0,
                "train_seconds": seconds, **rec,
            })

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

    Persistence sits in the leaderboard as the honest floor. If it wins, that
    is the result and the forecast degenerates to a carry-forward.
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
