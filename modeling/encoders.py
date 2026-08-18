"""Per-model-family preprocessing.

Three encoder families, selected per model via ``config.MODEL_ENCODER``:

* ``"linear"`` — smoothed **target encoding** for the high-cardinality
  geographic categoricals (postcode3, gemeente), one-hot for the low-card
  ordered/binary categoricals (age band, gender), and standard scaling for the
  numeric features. Suited to Linear/Ridge/Lasso.
* ``"tree"`` — **ordinal encoding** of all categoricals, numerics passed
  through. Suited to Random Forest / GBM / XGBoost / LightGBM, which split on
  encoded integers without needing one-hot expansion.
* ``"catboost"`` — no transformer; CatBoost consumes raw categoricals via its
  native ``cat_features`` (handled in ``tier3_boosting.py``).

Every transformer is fit on training data only. When wrapped in a scikit-learn
``Pipeline`` and driven by ``GridSearchCV``/``cross_val_predict``, the encoders
are refit on each fold's training portion, so target encoding never leaks the
held-out fold's target.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from . import config


def make_preprocessor(family: str) -> ColumnTransformer:
    """Build the ``ColumnTransformer`` for a given encoder *family*.

    Parameters
    ----------
    family:
        One of ``"linear"`` or ``"tree"``. (``"catboost"`` uses no preprocessor
        and must not be passed here.)
    """
    if family == "linear":
        # Target-encode geography THEN scale it, so the encoded euro values sit
        # on the same standardized scale as the numeric features — important for
        # Ridge/Lasso penalties. Drop one dummy level to avoid the
        # dummy-variable trap (rank-deficient design / unstable OLS coefs).
        geo_pipe = Pipeline([
            ("target", TargetEncoder(smooth="auto", random_state=config.RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("geo_target", geo_pipe, config.HIGH_CARD_FEATURES),
                (
                    "cat_ohe",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    config.LOW_CARD_FEATURES,
                ),
                ("num_scale", StandardScaler(), config.NUMERIC_FEATURES),
            ],
            remainder="drop",
        )

    if family == "tree":
        return ColumnTransformer(
            transformers=[
                (
                    "cat_ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    config.CATEGORICAL_FEATURES,
                ),
                ("num_pass", "passthrough", config.NUMERIC_FEATURES),
            ],
            remainder="drop",
        )

    raise ValueError(
        f"Unknown encoder family {family!r}; expected 'linear' or 'tree' "
        "('catboost' is handled natively without a preprocessor)."
    )
