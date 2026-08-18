"""Per-model-family preprocessing, selected via ``config.MODEL_ENCODER``.

* "linear" — smoothed target encoding for the high-cardinality geography,
  one-hot for age band and gender, standard scaling for the numerics.
* "tree" — ordinal encoding of all categoricals, numerics passed through.
* "catboost" — no transformer at all; see tier3_boosting.py.

Everything lives inside a Pipeline, so the encoders are refit per CV fold and
target encoding never sees the held-out fold's target.
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


def make_preprocessor(
    family: str, numeric_features: list[str] | None = None
) -> ColumnTransformer:
    """Build the ``ColumnTransformer`` for the "linear" or "tree" family.

    Since the transformers drop the remainder, a caller with an extended
    feature frame (the temporal track and its t-1 lag) has to pass its own
    *numeric_features*, or the extra columns disappear silently.
    """
    numeric_features = (
        list(config.NUMERIC_FEATURES) if numeric_features is None
        else list(numeric_features)
    )
    if family == "linear":
        # Scale after target-encoding, so the encoded euro values land on the
        # same scale as the numerics — the Ridge/Lasso penalties care. Dropping
        # one dummy level avoids a rank-deficient design.
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
                ("num_scale", StandardScaler(), numeric_features),
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
                ("num_pass", "passthrough", numeric_features),
            ],
            remainder="drop",
        )

    raise ValueError(
        f"Unknown encoder family {family!r}; expected 'linear' or 'tree' "
        "('catboost' is handled natively without a preprocessor)."
    )
