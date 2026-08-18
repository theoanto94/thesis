"""Phase 2 — predictive modeling & validation (RQ2).

A four-tier model system predicting real cost rates at the
postcode3 x age_group x gender x year grain, trained on 2011-2020 and
evaluated once on the held-out 2021-2023.
"""

import warnings

# numpy 2.x on Apple Accelerate raises matmul warnings for perfectly finite
# operands. evaluate.py verifies predictions are finite anyway, so filter the
# three narrowly rather than living with the noise.
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

# LightGBM warns about feature names when predicting from a bare ndarray after
# a ColumnTransformer fit. The pipeline fixes the column order, so it's noise.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
