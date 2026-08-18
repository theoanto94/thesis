"""Model dispatch + uniform fit/predict/tuning helpers.

Every base model (tiers 1-3) is built through :func:`build`, which returns a
native scikit-learn-compatible estimator — either a ``Pipeline`` (linear/tree
families) or a bare ``CatBoostRegressor``. The two forms differ only in how
``sample_weight`` is routed and how hyperparameter names are namespaced; the
helpers here paper over that so ``train.py``/``tune.py``/``tier4_stacking.py``
can treat all models identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from . import config, tier1_linear, tier2_trees, tier3_boosting

#: model key -> the tier module that builds it.
_BUILDERS = {
    **{k: tier1_linear for k in tier1_linear.MODEL_KEYS},
    **{k: tier2_trees for k in tier2_trees.MODEL_KEYS},
    **{k: tier3_boosting for k in tier3_boosting.MODEL_KEYS},
}

#: base (non-stacked) model keys in tier order.
BASE_MODEL_KEYS = list(_BUILDERS.keys())


def build(model_key: str, params: dict | None = None):
    """Construct an unfitted estimator for *model_key* with *params*."""
    if model_key not in _BUILDERS:
        raise ValueError(f"Unknown base model {model_key!r}")
    return _BUILDERS[model_key].build(model_key, params)


def is_pipeline(estimator) -> bool:
    """True for the linear/tree Pipeline form (vs. a bare CatBoost)."""
    return isinstance(estimator, Pipeline)


def fit(estimator, X, y, sample_weight):
    """Fit *estimator* with population weights, routing per estimator form."""
    if is_pipeline(estimator):
        estimator.fit(X, y, model__sample_weight=sample_weight)
    else:  # CatBoost: cat_features already set in the constructor
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


def predict(estimator, X):
    """Predict with *estimator* (uniform call site)."""
    return estimator.predict(X)


def fit_for_target(model_key: str, params: dict | None, X, y, sample_weight):
    """Fit a model on one target, with a constant-predictor fallback.

    Some cost categories are zero (or otherwise constant) across the entire
    training period — e.g. ``integrated_birth_care``, which did not exist before
    the test years. There is no signal to learn, and CatBoost refuses to train
    on a constant target ("All train targets are equal"). For any zero-variance
    target we return a fitted :class:`~sklearn.dummy.DummyRegressor` serving that
    constant, so every model/target combination behaves uniformly downstream.
    """
    y_arr = np.asarray(y, dtype=float)
    if np.unique(y_arr).size <= 1:
        const = float(y_arr[0]) if y_arr.size else 0.0
        dummy = DummyRegressor(strategy="constant", constant=const)
        dummy.fit(X, y_arr, sample_weight=sample_weight)
        return dummy
    est = build(model_key, params)
    return fit(est, X, y, sample_weight)


def grid_for(model_key: str, estimator) -> dict[str, list]:
    """Return the GridSearchCV param grid, namespaced for the estimator form.

    Pipelines need the ``model__`` prefix; CatBoost takes bare param names.
    """
    grid = config.PARAM_GRIDS[model_key]
    if is_pipeline(estimator):
        return {f"model__{k}": v for k, v in grid.items()}
    return dict(grid)


def fit_params_for(estimator, sample_weight) -> dict:
    """Return the ``**fit_params`` to pass to ``GridSearchCV.fit``."""
    if is_pipeline(estimator):
        return {"model__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def apply_best_params(model_key: str, best_params: dict) -> dict:
    """Strip any ``model__`` prefix from tuned params back to bare kwargs.

    The persisted/best params from GridSearchCV carry the pipeline prefix;
    :func:`build` expects bare estimator kwargs.
    """
    return {k.replace("model__", "", 1): v for k, v in best_params.items()}
