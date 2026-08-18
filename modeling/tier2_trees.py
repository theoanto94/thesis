"""Tier 2 — tree-based ensembles.

Random Forest, Gradient Boosting, and XGBoost. All consume the ``"tree"``
encoder family (ordinal-encoded categoricals + passthrough numerics) and are
wrapped in a Pipeline whose final step is named ``"model"`` so ``sample_weight``
routes via ``model__sample_weight``. Fully explainable via exact TreeSHAP in
Phase 3.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import config
from .encoders import make_preprocessor

MODEL_KEYS = ["random_forest", "gbm", "xgboost"]


def _raw_estimator(model_key: str, params: dict):
    """Construct the bare tree estimator with the given hyperparameters."""
    if model_key == "random_forest":
        return RandomForestRegressor(
            random_state=config.RANDOM_STATE, n_jobs=-1, **params
        )
    if model_key == "gbm":
        return GradientBoostingRegressor(random_state=config.RANDOM_STATE, **params)
    if model_key == "xgboost":
        return XGBRegressor(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            **params,
        )
    raise ValueError(f"Unknown tier-2 model {model_key!r}")


def build(model_key: str, params: dict | None = None) -> Pipeline:
    """Return an unfitted Pipeline (tree preprocessor + estimator)."""
    params = dict(params or {})
    return Pipeline([
        ("pre", make_preprocessor("tree")),
        ("model", _raw_estimator(model_key, params)),
    ])
