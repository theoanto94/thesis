"""Tier 3 — advanced gradient boosting.

LightGBM (ordinal-encoded via the ``"tree"`` family) and CatBoost (raw
categoricals via its native ``cat_features``, so *no* preprocessor). Both are
fully explainable via exact TreeSHAP in Phase 3.

LightGBM is returned as a Pipeline (``model__`` sample-weight routing).
CatBoost is returned as a bare ``CatBoostRegressor`` with ``cat_features`` fixed
in the constructor, so its ``fit``/``predict`` accept the raw feature frame
directly. ``factory.py`` knows how to feed each form.
"""

from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from . import config
from .encoders import make_preprocessor

MODEL_KEYS = ["lightgbm", "catboost"]


def build(model_key: str, params: dict | None = None):
    """Return an unfitted estimator for a tier-3 model.

    LightGBM -> Pipeline(tree preprocessor + LGBMRegressor).
    CatBoost -> CatBoostRegressor with ``cat_features`` set (no preprocessor).
    """
    params = dict(params or {})

    if model_key == "lightgbm":
        est = LGBMRegressor(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        return Pipeline([("pre", make_preprocessor("tree")), ("model", est)])

    if model_key == "catboost":
        return CatBoostRegressor(
            random_seed=config.RANDOM_STATE,
            cat_features=config.CATEGORICAL_FEATURES,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    raise ValueError(f"Unknown tier-3 model {model_key!r}")
