"""Per-model-family preprocessing, selected via ``config.MODEL_ENCODER``.

* "linear" — smoothed target encoding for the high-cardinality geography,
  one-hot for age band and gender, standard scaling for the numerics.
* "linear_interact" — as "linear", but age band and gender enter as one-hots
  of their 38 joint cells, so the age x gender reversal is representable.
* "tree" — ordinal encoding of all categoricals, numerics passed through.
* "catboost" — no transformer at all; see tier3_boosting.py.

Everything lives inside a Pipeline, so the encoders are refit per CV fold and
target encoding never sees the held-out fold's target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from . import config


class AgeGenderCross(BaseEstimator, TransformerMixin):
    """One-hot of the joint age x gender cell.

    Input columns: ``age_group`` and ``gender``. Output: 37 dummies (first
    cell dropped, to keep the design full-rank next to the intercept). Cells
    unseen in training get all-zero dummies.
    """

    @staticmethod
    def _cells(X: pd.DataFrame) -> np.ndarray:
        """``age|gender`` labels, one per row."""
        return (X["age_group"].astype(str) + "|" + X["gender"].astype(str)).to_numpy()

    def fit(self, X: pd.DataFrame, y=None):
        """Record the age x gender cells seen in training."""
        self.cells_ = np.array(sorted(set(self._cells(X))))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Drop-first dummies of the joint cell."""
        onehot = (self._cells(X)[:, None] == self.cells_[None, :]).astype(float)
        return onehot[:, 1:]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names ``cell=<age|gender>``."""
        return np.array([f"cell={c}" for c in self.cells_[1:]], dtype=object)


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

    if family == "linear_interact":
        geo_pipe = Pipeline([
            ("target", TargetEncoder(smooth="auto", random_state=config.RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("geo_target", geo_pipe, config.HIGH_CARD_FEATURES),
                ("age_gender", AgeGenderCross(), config.LOW_CARD_FEATURES),
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
        f"Unknown encoder family {family!r}; expected 'linear', "
        "'linear_interact' or 'tree' "
        "('catboost' is handled natively without a preprocessor)."
    )
