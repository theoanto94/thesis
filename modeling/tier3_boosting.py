"""Tier 3 — advanced gradient boosting.

LightGBM comes back as a Pipeline on the "tree" family; CatBoost as a bare
regressor taking the raw frame through its native cat_features. factory.py
knows how to feed each form. Both are explainable via TreeSHAP in Phase 3.
"""

from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from . import config
from .encoders import make_preprocessor

MODEL_KEYS = ["lightgbm", "catboost"]


def raw_estimator(model_key: str, params: dict):
    """Construct the bare tier-3 estimator with the given hyperparameters."""
    if model_key == "lightgbm":
        return LGBMRegressor(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    if model_key == "catboost":
        return CatBoostRegressor(
            random_seed=config.RANDOM_STATE,
            cat_features=config.CATEGORICAL_FEATURES,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    raise ValueError(f"Unknown tier-3 model {model_key!r}")


def build(model_key: str, params: dict | None = None,
          numeric_features: list[str] | None = None):
    """Return an unfitted estimator for a tier-3 model.

    LightGBM -> Pipeline(tree preprocessor + LGBMRegressor).
    CatBoost -> CatBoostRegressor with ``cat_features`` set (no preprocessor).
    """
    params = dict(params or {})
    est = raw_estimator(model_key, params)

    if model_key == "catboost":
        # With no preprocessor there is nothing to route an extended column
        # set through — an extra numeric would be ignored, not modelled.
        if numeric_features is not None:
            raise ValueError(
                "catboost consumes raw features and cannot take a "
                "numeric_features override"
            )
        return est

    return Pipeline([
        ("pre", make_preprocessor(config.MODEL_ENCODER[model_key], numeric_features)),
        ("model", est),
    ])
