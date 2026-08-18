"""Tier 1 — traditional linear baselines.

Linear Regression, Ridge (L2), and Lasso (L1). These establish the performance
floor and are intrinsically interpretable via their coefficients, so they are
scoped out of the Phase 3 SHAP analysis.

Each estimator is wrapped in a Pipeline with the ``"linear"`` preprocessor
(target-encoded geography + one-hot age/gender + scaled numerics). ``sample_weight``
is forwarded to the final estimator under the pipeline step name ``model``.
"""

from __future__ import annotations

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from . import config
from .encoders import make_preprocessor

MODEL_KEYS = ["linear", "ridge", "lasso"]


def _raw_estimator(model_key: str, params: dict):
    """Construct the bare linear estimator with the given hyperparameters."""
    if model_key == "linear":
        return LinearRegression()
    if model_key == "ridge":
        return Ridge(random_state=config.RANDOM_STATE, **params)
    if model_key == "lasso":
        return Lasso(random_state=config.RANDOM_STATE, max_iter=10_000, **params)
    raise ValueError(f"Unknown tier-1 model {model_key!r}")


def build(model_key: str, params: dict | None = None) -> Pipeline:
    """Return an unfitted Pipeline (linear preprocessor + estimator).

    The final step is named ``"model"`` so callers can pass
    ``model__sample_weight`` to :meth:`fit`.
    """
    params = dict(params or {})
    return Pipeline([
        ("pre", make_preprocessor("linear")),
        ("model", _raw_estimator(model_key, params)),
    ])
