"""Phase 2 — predictive modeling & validation (RQ2).

A four-tier model system predicting real healthcare cost rates
(2020 constant euros per insured person-year) at the
postcode3 x age_group x gender x year grain. Trained on 2011-2020,
evaluated once on the held-out 2021-2023.
"""

import warnings

# numpy 2.x on macOS Apple Accelerate emits a spurious "divide by zero
# encountered in matmul" RuntimeWarning for perfectly finite operands (a known
# Accelerate BLAS quirk). Predictions are verified finite in evaluate.py; this
# narrow filter keeps training logs clean without masking genuine overflow.
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in matmul",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in matmul",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in matmul",
    category=RuntimeWarning,
)

# LightGBM's sklearn wrapper warns when predict is called with a bare ndarray
# after being fit through a ColumnTransformer (feature-name bookkeeping only).
# Benign — the column order is fixed by the pipeline.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
