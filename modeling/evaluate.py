"""Held-out evaluation on 2021-2023 — touched exactly once.

The only place the test split is read. Predicts every (model, target) from the
persisted models and writes what the results dashboard consumes: row-level
predictions, overall metrics, the by-age-group and by-gementee roll-ups, and
the learning curves. Predictions are asserted finite and the test years
re-checked before anything is written.
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
from .train import model_path

ALL_MODELS = factory.BASE_MODEL_KEYS + [STACK_KEY]


def _predict(model_key: str, target: str, ds: Dataset) -> np.ndarray:
    """Predict the test set for one (model, target)."""
    if model_key == STACK_KEY:
        return predict_stacked(target, ds.X_test)
    est = joblib.load(model_path(model_key, target))
    return est.predict(ds.X_test)


def evaluate_model(model_key: str, ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every target, persist row-level preds, return metrics + preds.

    The prediction frame is returned as well as written, so _segment_metrics
    does not have to read the parquet back.
    """
    config.ensure_dirs()
    frames: list[pd.DataFrame] = []
    overall_rows: list[dict] = []

    meta = ds.meta_test[config.META_COLUMNS].reset_index(drop=True)
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
    return pd.DataFrame(overall_rows)[cols], preds


def _segment_metrics(model_key: str, preds: pd.DataFrame, by: str) -> pd.DataFrame:
    """Weighted metrics per (target x segment level) for one model."""
    out: list[pd.DataFrame] = []
    for target, sub in preds.groupby("target", observed=True):
        d = disaggregate(sub, sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                         sub["weight"].to_numpy(), by)
        d.insert(0, "target", target)
        d.insert(0, "model", model_key)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def learning_curve(model_key: str, ds: Dataset,
                   sizes: tuple[int, ...] | None = None,
                   n_splits: int | None = None) -> pd.DataFrame:
    """Weighted train vs CV RMSE as a function of training-set size.

    Total target only, with the model's tuned params; the stack is skipped
    since it is a meta-model. Sizes come from config, plus the full training
    size.

    Each size is a random subsample rather than the first n rows: DuckDB row
    order correlates with year, so iloc[:n] would train every point on a
    different year mix and bend the curve for no real reason.
    """
    from .tune import load_best_params

    params = factory.apply_best_params(model_key, load_best_params(model_key))
    X, y, w = ds.X_train, ds.y_train[config.TARGET_TOTAL], ds.w_train.to_numpy()
    n_total = len(X)
    requested = sizes or config.LEARNING_CURVE_SIZES
    plan = sorted({n for n in requested if n < n_total} | {n_total})
    order = np.random.RandomState(config.RANDOM_STATE).permutation(n_total)
    n_splits = config.LEARNING_CURVE_SPLITS if n_splits is None else n_splits
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


def _upsert(path, new: pd.DataFrame) -> pd.DataFrame:
    """Write *new* to *path*, replacing only the rows of the models it contains.

    Evaluating a subset of models must not drop the others from the shared
    metric files the dashboard and the thesis tables read.
    """
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    order = {k: i for i, k in enumerate(config.MODEL_KEYS)}
    new = new.sort_values("model", key=lambda s: s.map(order).fillna(len(order)),
                          kind="stable").reset_index(drop=True)
    new.to_parquet(path, index=False)
    return new


def main(models: list[str] | None = None, learning_curves: bool = True) -> None:
    """Evaluate the requested models (default: all) and write all artifacts."""
    ds = load_dataset()

    assert set(ds.meta_test["year"].unique()) <= set(config.TEST_YEARS), "Test-year leak"

    keys = models or ALL_MODELS
    overall, by_age, by_gem = [], [], []

    for key in keys:
        print(f"Evaluating {key}...")
        metrics, preds = evaluate_model(key, ds)
        overall.append(metrics)
        by_age.append(_segment_metrics(key, preds, "age_group"))
        by_gem.append(_segment_metrics(key, preds, "gementee"))
        if learning_curves and key != STACK_KEY:
            learning_curve(key, ds).to_parquet(
                config.LEARNING_CURVES_DIR / f"{key}.parquet", index=False)

    config.ensure_dirs()
    ov = _upsert(config.METRICS_DIR / "overall.parquet", pd.concat(overall, ignore_index=True))
    _upsert(config.METRICS_DIR / "by_age_group.parquet", pd.concat(by_age, ignore_index=True))
    _upsert(config.METRICS_DIR / "by_gementee.parquet", pd.concat(by_gem, ignore_index=True))

    tot = ov[ov["target"] == config.TARGET_TOTAL].sort_values("rmse")
    print("\nTotal-target leaderboard (test 2021-2023):")
    print(tot[["model", "r2", "rmse", "mae", "mape"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    no_lc = "--no-learning-curves" in args
    arg_models = [a for a in args if not a.startswith("--")] or None
    main(models=arg_models, learning_curves=not no_lc)
