"""Tier 1 — traditional linear baselines.

Linear Regression, Ridge and Lasso: the performance floor, interpretable
through their coefficients, so out of scope for the Phase 3 SHAP analysis.
"""

from __future__ import annotations

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from . import config
from .encoders import make_preprocessor

MODEL_KEYS = ["linear", "ridge", "lasso"]


def raw_estimator(model_key: str, params: dict):
    """Construct the bare linear estimator with the given hyperparameters."""
    if model_key == "linear":
        return LinearRegression()
    if model_key == "ridge":
        return Ridge(random_state=config.RANDOM_STATE, **params)
    if model_key == "lasso":
        return Lasso(random_state=config.RANDOM_STATE, max_iter=10_000, **params)
    raise ValueError(f"Unknown tier-1 model {model_key!r}")


def build(model_key: str, params: dict | None = None,
          numeric_features: list[str] | None = None) -> Pipeline:
    """Return an unfitted Pipeline (linear preprocessor + estimator).

    The final step is named "model" so callers can route
    ``model__sample_weight`` through fit.
    """
    params = dict(params or {})
    return Pipeline([
        ("pre", make_preprocessor(config.MODEL_ENCODER[model_key], numeric_features)),
        ("model", raw_estimator(model_key, params)),
    ])
