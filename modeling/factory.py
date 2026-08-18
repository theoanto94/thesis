"""Model dispatch plus uniform fit/predict/tuning helpers.

Base models come back either as a Pipeline (linear/tree families) or as a bare
CatBoostRegressor. The two differ in how sample_weight is routed and how
hyperparameter names are namespaced; the helpers here hide that so callers can
treat every model the same.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from . import tier1_linear, tier2_trees, tier3_boosting

# model key -> the tier module that builds it
_BUILDERS = {
    **{k: tier1_linear for k in tier1_linear.MODEL_KEYS},
    **{k: tier2_trees for k in tier2_trees.MODEL_KEYS},
    **{k: tier3_boosting for k in tier3_boosting.MODEL_KEYS},
}

BASE_MODEL_KEYS = list(_BUILDERS.keys())


def build(model_key: str, params: dict | None = None,
          numeric_features: list[str] | None = None):
    """Construct an unfitted estimator for *model_key* with *params*.

    *numeric_features* overrides the preprocessor's numeric columns; the
    temporal track uses it to add its t-1 lag.
    """
    if model_key not in _BUILDERS:
        raise ValueError(f"Unknown base model {model_key!r}")
    return _BUILDERS[model_key].build(model_key, params, numeric_features)


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
    """Predict with *estimator*."""
    return estimator.predict(X)


def fit_for_target(model_key: str, params: dict | None, X, y, sample_weight):
    """Fit a model on one target, with a constant-predictor fallback.

    A few cost categories are constant across the whole training period —
    integrated_birth_care did not exist before the test years — and CatBoost
    refuses to train on those at all. Falling back to a DummyRegressor keeps
    every model/target combination uniform downstream.
    """
    y_arr = np.asarray(y, dtype=float)
    if np.unique(y_arr).size <= 1:
        const = float(y_arr[0]) if y_arr.size else 0.0
        dummy = DummyRegressor(strategy="constant", constant=const)
        dummy.fit(X, y_arr, sample_weight=sample_weight)
        return dummy
    est = build(model_key, params)
    return fit(est, X, y, sample_weight)


def apply_best_params(model_key: str, best_params: dict) -> dict:
    """Strip the ``model__`` pipeline prefix GridSearchCV adds to tuned params."""
    return {k.replace("model__", "", 1): v for k, v in best_params.items()}
