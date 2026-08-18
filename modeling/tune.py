"""Hyperparameter tuning — weighted K-Fold grid search on the total target.

Per the Phase 2 spec, tuning runs once on the primary ``"total"`` target via
K-Fold CV (K=5) within the 2011-2020 training set only; the resulting best
params are persisted and reused to refit every per-category target.

Selection metric is **population-weighted RMSE** (minimised). We implement the
search manually rather than via ``GridSearchCV`` so that (a) the CV score itself
is weighted by ``insured_years`` and (b) the same code path handles both the
Pipeline models and the bare CatBoost estimator.
"""

from __future__ import annotations

import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, ParameterGrid

from . import config, factory
from .data import Dataset, load_dataset
from .metrics import weighted_metrics


def _cv_rmse(model_key: str, params: dict, X, y, w, folds) -> float:
    """Mean weighted-RMSE of one param combo across the CV folds."""
    scores = []
    for tr_idx, va_idx in folds:
        est = factory.build(model_key, params)
        factory.fit(est, X.iloc[tr_idx], y.iloc[tr_idx], w[tr_idx])
        pred = factory.predict(est, X.iloc[va_idx])
        m = weighted_metrics(y.iloc[va_idx].to_numpy(), pred, w[va_idx])
        scores.append(m["rmse"])
    return float(np.mean(scores))


def tune_model(model_key: str, ds: Dataset, n_jobs: int = 1) -> dict:
    """Grid-search one model on the total target; return the best-param record.

    The record contains the bare best params (no pipeline prefix), the best CV
    RMSE, the grid size, and the wall-clock seconds taken.

    Default ``n_jobs=1`` runs the grid serially in-process: the tree models are
    already internally multithreaded (and saturate all cores per fit), so outer
    parallelism only oversubscribes the CPU and pays a large cost pickling the
    ~292k-row frame to each worker. ``n_jobs>1`` is offered for the
    single-threaded models (linear/lasso/gbm) if desired.
    """
    grid = list(ParameterGrid(config.PARAM_GRIDS[model_key]))
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(kf.split(ds.X_train))
    y = ds.y_train[config.TARGET_TOTAL]
    w = ds.w_train.to_numpy()

    t0 = time.time()
    rmses = Parallel(n_jobs=n_jobs)(
        delayed(_cv_rmse)(model_key, p, ds.X_train, y, w, folds) for p in grid
    )
    best_i = int(np.argmin(rmses))
    elapsed = time.time() - t0

    return {
        "model": model_key,
        "best_params": grid[best_i],
        "cv_rmse": float(rmses[best_i]),
        "n_candidates": len(grid),
        "tune_seconds": round(elapsed, 1),
    }


def save_params(record: dict) -> None:
    """Persist a tuning record to ``artifacts/params/<model>.json``."""
    config.ensure_dirs()
    path = config.PARAMS_DIR / f"{record['model']}.json"
    path.write_text(json.dumps(record, indent=2))


def load_best_params(model_key: str) -> dict:
    """Load persisted best params for *model_key* (empty for untuned models)."""
    path = config.PARAMS_DIR / f"{model_key}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("best_params", {})


#: Models that actually have a grid to search (everything except plain linear).
TUNABLE = [k for k in factory.BASE_MODEL_KEYS if k in config.PARAM_GRIDS]


def main(models: list[str] | None = None, n_jobs: int = 4) -> None:
    """Tune the requested models (default: all tunable) and persist params."""
    ds = load_dataset()
    targets = models or TUNABLE
    # 'linear' has no params but we persist an empty record for completeness.
    if (models is None) and ("linear" in factory.BASE_MODEL_KEYS):
        save_params({"model": "linear", "best_params": {}, "cv_rmse": None,
                     "n_candidates": 0, "tune_seconds": 0.0})
    for key in targets:
        print(f"Tuning {key} ({len(list(ParameterGrid(config.PARAM_GRIDS[key])))} combos)...")
        rec = tune_model(key, ds, n_jobs=n_jobs)
        save_params(rec)
        print(f"  best CV RMSE=€{rec['cv_rmse']:,.1f}  params={rec['best_params']}  "
              f"({rec['tune_seconds']}s)")


if __name__ == "__main__":
    import sys
    arg_models = sys.argv[1:] or None
    main(models=arg_models)
