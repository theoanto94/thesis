"""Tier 4 — stacked meta-ensemble.

Per target: K-Fold the training set, collect each base learner's out-of-fold
predictions into a meta-feature matrix, and fit a weighted Ridge on it. Because
no row is predicted by a model that trained on it, the meta-learner sees no
leakage.

At test time the base learners are the full-train artifacts from train.py, and
their 2021-2023 predictions feed the fitted meta-learner.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from . import config, factory
from .data import Dataset
from .tune import load_best_params

STACK_KEY = "stacked"


def _meta_path(target: str) -> Path:
    return config.MODELS_DIR / f"{STACK_KEY}_meta__{target}.joblib"


def build_oof(target: str, ds: Dataset) -> np.ndarray:
    """Return the OOF meta-feature matrix ``[n_train, n_base]`` for *target*.

    Column order follows config.STACK_BASE_LEARNERS, which predict_stacked
    reads too, so the meta-learner's inputs cannot drift between fit and
    predict.
    """
    base_keys = config.STACK_BASE_LEARNERS
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(kf.split(ds.X_train))
    y = ds.y_train[target]
    w = ds.w_train.to_numpy()
    n = len(ds.X_train)
    oof = np.zeros((n, len(base_keys)), dtype=float)

    for j, key in enumerate(base_keys):
        params = factory.apply_best_params(key, load_best_params(key))
        for tr_idx, va_idx in folds:
            est = factory.fit_for_target(key, params, ds.X_train.iloc[tr_idx],
                                         y.iloc[tr_idx], w[tr_idx])
            oof[va_idx, j] = factory.predict(est, ds.X_train.iloc[va_idx])
    return oof


def train_stacked(target: str, ds: Dataset) -> Ridge:
    """Generate OOF preds, fit the Ridge meta-learner, and persist it."""
    config.ensure_dirs()
    oof = build_oof(target, ds)
    meta = Ridge(random_state=config.RANDOM_STATE)
    meta.fit(oof, ds.y_train[target], sample_weight=ds.w_train.to_numpy())
    joblib.dump(meta, _meta_path(target))
    (config.PARAMS_DIR / f"{STACK_KEY}.json").write_text(
        json.dumps({"model": STACK_KEY,
                    "base_learners": list(config.STACK_BASE_LEARNERS),
                    "meta_learner": "ridge"}, indent=2)
    )
    return meta


def predict_stacked(target: str, X) -> np.ndarray:
    """Predict with the stack: full-train base models -> Ridge meta.

    Needs train.py and train_stacked to have run for *target* first.
    """
    from .train import model_path  # local import avoids a cycle

    meta_features = np.column_stack([
        joblib.load(model_path(key, target)).predict(X)
        for key in config.STACK_BASE_LEARNERS
    ])
    meta: Ridge = joblib.load(_meta_path(target))
    return meta.predict(meta_features)


def main(targets: list[str] | None = None) -> None:
    """Train the stack for the requested targets (default: all)."""
    from .data import load_dataset

    ds = load_dataset()
    keys = targets or list(config.TARGETS.keys())
    for target in keys:
        meta = train_stacked(target, ds)
        weights = dict(zip(config.STACK_BASE_LEARNERS, np.round(meta.coef_, 3)))
        print(f"stacked[{target}] meta weights={weights}")


if __name__ == "__main__":
    import sys
    arg_targets = sys.argv[1:] or None
    main(targets=arg_targets)
