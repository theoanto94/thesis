"""Tier 4 — stacked meta-ensemble.

Builds a stacked ensemble from the tier 2-3 base learners using **out-of-fold
(OOF)** predictions, so the meta-learner never sees a base prediction made by a
model that was trained on that row.

Pipeline (per target):

1. K-Fold the training set. For each base learner and each fold, fit on K-1
   folds (tuned params) and predict the held-out fold. Concatenating the
   held-out predictions yields one leakage-free OOF prediction per training row.
2. Stack the base learners' OOF columns into the meta-feature matrix.
3. Fit a Ridge meta-learner (population-weighted) on that matrix.

For test prediction the base learners are refit on the *full* training set
(exactly the artifacts persisted by ``train.py``), predict 2021-2023, and those
predictions feed the trained meta-learner.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from . import config, factory
from .data import Dataset
from .tune import load_best_params

STACK_KEY = "stacked"


def _base_keys() -> list[str]:
    """Base learners feeding the stack (tiers 2-3)."""
    return list(config.STACK_BASE_LEARNERS)


def _meta_path(target: str) -> "config.Path":
    return config.MODELS_DIR / f"{STACK_KEY}_meta__{target}.joblib"


def build_oof(target: str, ds: Dataset, base_keys: list[str]) -> np.ndarray:
    """Return the OOF meta-feature matrix ``[n_train, n_base]`` for *target*."""
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    y = ds.y_train[target]
    w = ds.w_train.to_numpy()
    n = len(ds.X_train)
    oof = np.zeros((n, len(base_keys)), dtype=float)

    for j, key in enumerate(base_keys):
        params = factory.apply_best_params(key, load_best_params(key))
        for tr_idx, va_idx in kf.split(ds.X_train):
            # fit_for_target guards against folds whose target is constant.
            est = factory.fit_for_target(key, params, ds.X_train.iloc[tr_idx],
                                         y.iloc[tr_idx], w[tr_idx])
            oof[va_idx, j] = factory.predict(est, ds.X_train.iloc[va_idx])
    return oof


def train_stacked(target: str, ds: Dataset, base_keys: list[str] | None = None) -> Ridge:
    """Generate OOF preds, fit the Ridge meta-learner, and persist it."""
    config.ensure_dirs()
    base_keys = base_keys or _base_keys()
    oof = build_oof(target, ds, base_keys)
    meta = Ridge(random_state=config.RANDOM_STATE)
    meta.fit(oof, ds.y_train[target], sample_weight=ds.w_train.to_numpy())
    joblib.dump(meta, _meta_path(target))
    # Record the base-learner order once (shared across targets).
    (config.PARAMS_DIR / f"{STACK_KEY}.json").write_text(
        json.dumps({"model": STACK_KEY, "base_learners": base_keys,
                    "meta_learner": "ridge"}, indent=2)
    )
    return meta


def predict_stacked(target: str, X, base_keys: list[str] | None = None) -> np.ndarray:
    """Predict with the stack: full-train base models -> Ridge meta.

    Requires that ``train.py`` has already persisted the full-train base models
    and that :func:`train_stacked` has persisted the meta-learner for *target*.
    """
    from .train import model_path  # local import avoids a cycle

    base_keys = base_keys or _base_keys()
    meta_features = np.column_stack([
        joblib.load(model_path(key, target)).predict(X) for key in base_keys
    ])
    meta: Ridge = joblib.load(_meta_path(target))
    return meta.predict(meta_features)


def main(targets: list[str] | None = None) -> None:
    """Train the stack for the requested targets (default: all)."""
    from .data import load_dataset

    ds = load_dataset()
    keys = targets or list(config.TARGETS.keys())
    base_keys = _base_keys()
    for target in keys:
        meta = train_stacked(target, ds, base_keys)
        weights = dict(zip(base_keys, np.round(meta.coef_, 3)))
        print(f"stacked[{target}] meta weights={weights}")


if __name__ == "__main__":
    import sys
    arg_targets = sys.argv[1:] or None
    main(targets=arg_targets)
