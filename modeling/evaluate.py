"""Held-out evaluation on 2021-2023 — touched exactly once.

Loads the persisted models, predicts the test set for every (model, target),
and writes the artifacts the results dashboard consumes:

* ``artifacts/predictions/<model>.parquet`` — row-level test predictions across
  all 24 targets (compact dtypes); the dashboard recomputes segmented metrics
  from these on the fly.
* ``artifacts/metrics/overall.parquet`` — weighted R2/RMSE/MAE/MAPE per
  (model, target).
* ``artifacts/metrics/by_age_group.parquet`` / ``by_gementee.parquet`` —
  pre-rolled single-dimension disaggregations for convenience.
* ``artifacts/learning_curves/<model>.parquet`` — train vs CV error by training
  set size (total target), for the learning-curve tab.

This module is the ONLY place the test split is read. Predictions are asserted
finite and the test years are re-checked before anything is written.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config, factory
from .data import Dataset, load_dataset
from .metrics import METRIC_NAMES, disaggregate, weighted_metrics
from .tier4_stacking import STACK_KEY, predict_stacked
from .train import fit_one, model_path

ALL_MODELS = factory.BASE_MODEL_KEYS + [STACK_KEY]

_ID_COLS = ["year", "postal_code_3", "gementee", "gementee_code", "age_group", "gender"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict(model_key: str, target: str, ds: Dataset) -> np.ndarray:
    """Predict the test set for one (model, target)."""
    if model_key == STACK_KEY:
        return predict_stacked(target, ds.X_test)
    est = joblib.load(model_path(model_key, target))
    return est.predict(ds.X_test)


def evaluate_model(model_key: str, ds: Dataset) -> pd.DataFrame:
    """Predict every target, persist row-level preds, return overall metrics.

    Returns a tidy frame with one row per target and the weighted metrics.
    """
    config.ensure_dirs()
    frames: list[pd.DataFrame] = []
    overall_rows: list[dict] = []

    meta = ds.meta_test[_ID_COLS].reset_index(drop=True)
    w = ds.w_test.to_numpy()

    for target in config.TARGETS:
        y_true = ds.y_test[target].to_numpy()
        y_pred = _predict(model_key, target, ds)
        assert np.isfinite(y_pred).all(), f"Non-finite prediction: {model_key}/{target}"

        block = meta.copy()
        block["target"] = target
        block["y_true"] = y_true.astype("float32")
        block["y_pred"] = y_pred.astype("float32")
        block["weight"] = w.astype("float32")
        frames.append(block)

        m = weighted_metrics(y_true, y_pred, w)
        m.update({"model": model_key, "target": target})
        overall_rows.append(m)

    preds = pd.concat(frames, ignore_index=True)
    for c in ["postal_code_3", "gementee", "gementee_code", "age_group", "gender", "target"]:
        preds[c] = preds[c].astype("category")
    preds.to_parquet(config.PREDICTIONS_DIR / f"{model_key}.parquet", index=False)

    cols = ["model", "target", *METRIC_NAMES]
    return pd.DataFrame(overall_rows)[cols]


# ---------------------------------------------------------------------------
# Disaggregation roll-ups
# ---------------------------------------------------------------------------

def _segment_metrics(model_key: str, ds: Dataset, by: str) -> pd.DataFrame:
    """Weighted metrics per (target x segment level) for one model."""
    preds = pd.read_parquet(config.PREDICTIONS_DIR / f"{model_key}.parquet")
    out: list[pd.DataFrame] = []
    for target, sub in preds.groupby("target", observed=True):
        d = disaggregate(sub, sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                         sub["weight"].to_numpy(), by)
        d.insert(0, "target", target)
        d.insert(0, "model", model_key)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Learning curves (total target only)
# ---------------------------------------------------------------------------

def learning_curve(model_key: str, ds: Dataset,
                   sizes: tuple[int, ...] | None = None, n_splits: int = 3) -> pd.DataFrame:
    """Weighted train vs CV RMSE as a function of training-set size.

    Computed on the total target with the model's tuned params. The stacked
    ensemble is skipped (it is a meta-model, not a base estimator).

    Training-set sizes are the log-spaced ``config.LEARNING_CURVE_SIZES`` (dense
    in the steep small-data regime), with the full training size always appended
    and any requested size >= it dropped.

    Each size is drawn as a *random* subsample (fixed seed), NOT the first ``n``
    rows: the DuckDB row order is not random (it correlates with year), so
    ``iloc[:n]`` would train each point on a different, non-representative year
    mix and produce a spurious non-monotonic curve.
    """
    from .tune import load_best_params

    params = factory.apply_best_params(model_key, load_best_params(model_key))
    X, y, w = ds.X_train, ds.y_train[config.TARGET_TOTAL], ds.w_train.to_numpy()
    n_total = len(X)
    requested = sizes or config.LEARNING_CURVE_SIZES
    plan = sorted({n for n in requested if n < n_total} | {n_total})
    order = np.random.RandomState(config.RANDOM_STATE).permutation(n_total)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    rows: list[dict] = []

    for n in plan:
        idx = order[:n]
        Xf, yf, wf = X.iloc[idx], y.iloc[idx], w[idx]
        tr_scores, cv_scores = [], []
        for tr_idx, va_idx in kf.split(Xf):
            est = factory.build(model_key, params)
            factory.fit(est, Xf.iloc[tr_idx], yf.iloc[tr_idx], wf[tr_idx])
            tr_pred = factory.predict(est, Xf.iloc[tr_idx])
            va_pred = factory.predict(est, Xf.iloc[va_idx])
            tr_scores.append(weighted_metrics(yf.iloc[tr_idx].to_numpy(), tr_pred, wf[tr_idx])["rmse"])
            cv_scores.append(weighted_metrics(yf.iloc[va_idx].to_numpy(), va_pred, wf[va_idx])["rmse"])
        rows.append({
            "model": model_key,
            "train_size": n,
            "train_rmse": float(np.mean(tr_scores)),
            "cv_rmse": float(np.mean(cv_scores)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(models: list[str] | None = None, learning_curves: bool = True) -> None:
    """Evaluate the requested models (default: all) and write all artifacts."""
    ds = load_dataset()

    # Re-assert the hold-out boundary before touching the test set.
    assert set(ds.meta_test["year"].unique()) <= set(config.TEST_YEARS), "Test-year leak"

    keys = models or ALL_MODELS
    overall, by_age, by_gem, curves = [], [], [], []

    for key in keys:
        print(f"Evaluating {key}...")
        overall.append(evaluate_model(key, ds))
        by_age.append(_segment_metrics(key, ds, "age_group"))
        by_gem.append(_segment_metrics(key, ds, "gementee"))
        if learning_curves and key != STACK_KEY:
            curves.append(lc := learning_curve(key, ds))
            lc.to_parquet(config.LEARNING_CURVES_DIR / f"{key}.parquet", index=False)

    config.ensure_dirs()
    pd.concat(overall, ignore_index=True).to_parquet(
        config.METRICS_DIR / "overall.parquet", index=False)
    pd.concat(by_age, ignore_index=True).to_parquet(
        config.METRICS_DIR / "by_age_group.parquet", index=False)
    pd.concat(by_gem, ignore_index=True).to_parquet(
        config.METRICS_DIR / "by_gementee.parquet", index=False)

    ov = pd.concat(overall, ignore_index=True)
    tot = ov[ov["target"] == config.TARGET_TOTAL].sort_values("rmse")
    print("\nTotal-target leaderboard (test 2021-2023):")
    print(tot[["model", "r2", "rmse", "mae", "mape"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    arg_models = sys.argv[1:] or None
    main(models=arg_models)
