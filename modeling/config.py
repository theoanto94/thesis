"""Central configuration for Phase 2 modeling. Importing this performs no I/O."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42
N_SPLITS: int = 5

# Fewer folds than N_SPLITS: the curve refits every model at every size.
LEARNING_CURVE_SPLITS: int = 3

# Log-spaced — the curve is most informative in the small-data regime.
LEARNING_CURVE_SIZES: tuple[int, ...] = (
    2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 150_000,
)

ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = ROOT / "identifier.db"

ARTIFACTS: Path = ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS / "models"
PARAMS_DIR: Path = ARTIFACTS / "params"
METRICS_DIR: Path = ARTIFACTS / "metrics"
PREDICTIONS_DIR: Path = ARTIFACTS / "predictions"
LEARNING_CURVES_DIR: Path = ARTIFACTS / "learning_curves"
TIMINGS_DIR: Path = ARTIFACTS / "timings"
SHAP_DIR: Path = ARTIFACTS / "shap"

_ALL_DIRS = (
    ARTIFACTS, MODELS_DIR, PARAMS_DIR, METRICS_DIR,
    PREDICTIONS_DIR, LEARNING_CURVES_DIR, TIMINGS_DIR, SHAP_DIR,
)


def ensure_dirs() -> None:
    """Create the artifact directory tree if it does not yet exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# Temporal hold-out, never a random split.
TRAIN_YEARS: range = range(2011, 2021)
TEST_YEARS: range = range(2021, 2024)

# Mirrors COST_COLUMNS in app.py. Uses the mental-health roll-up rather than
# its sub-components, to avoid double counting.
COST_CATEGORIES: dict[str, str] = {
    "Medical specialist care":          "costs_medical_specialist_care_real",
    "Pharmacy":                         "costs_pharmacy_real",
    "GP — registration fee":            "costs_gp_registration_fee_real",
    "GP — consultation":                "costs_gp_consultation_real",
    "GP — multidisciplinary care":      "costs_gp_multidisciplinary_care_real",
    "GP — other":                       "costs_gp_other_real",
    "Medical aids":                     "costs_medical_aids_real",
    "Dental care":                      "costs_dental_care_real",
    "Physiotherapy":                    "costs_physiotherapy_real",
    "Paramedical (other)":              "costs_paramedical_other_real",
    "Patient transport — seated":       "costs_patient_transport_seated_real",
    "Patient transport — ambulance":    "costs_patient_transport_ambulance_real",
    "Maternity care":                   "costs_maternity_care_real",
    "Obstetric care":                   "costs_obstetric_care_real",
    "Cross-border care":                "costs_cross_border_care_real",
    "Primary care support":             "costs_primary_care_support_real",
    "Geriatric rehabilitation":         "costs_geriatric_rehabilitation_real",
    "Primary care residence":           "costs_primary_care_residence_real",
    "Nursing care":                     "costs_nursing_care_real",
    "Medical care (specific patients)": "costs_medical_care_specific_patients_real",
    "Integrated birth care":            "costs_integrated_birth_care_real",
    "Mental health (total)":            "costs_mental_health_total_real",
    "Other":                            "costs_other_real",
}

COST_COLUMNS: list[str] = list(COST_CATEGORIES.values())

TARGET_TOTAL: str = "total"


def _target_key(cost_col: str) -> str:
    """``costs_pharmacy_real`` -> ``pharmacy``."""
    return cost_col[len("costs_"):-len("_real")]


# target key -> (display label, cost columns summed into the numerator)
TARGETS: dict[str, tuple[str, list[str]]] = {
    TARGET_TOTAL: ("Total (all categories)", COST_COLUMNS),
    **{
        _target_key(col): (label, [col])
        for label, col in COST_CATEGORIES.items()
    },
}

TARGET_LABELS: dict[str, str] = {k: v[0] for k, v in TARGETS.items()}

# Structural features only — no cost columns, no lags, so no target leakage.
CATEGORICAL_FEATURES: list[str] = [
    "age_group",       # 19 ordered 5-year bands
    "gender",          # M / V
    "postal_code_3",   # ~800 levels
    # gementee_code dropped: redundant with postal_code_3 (~€1 RMSE either
    # way); kept in META_COLUMNS for dashboard filtering.
]

NUMERIC_FEATURES: list[str] = [
    "year",
    "insured_years",       # population exposure, also the sample weight
    "number_of_insured",
    "latitude",
    "longitude",
]

FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Carried alongside every prediction/SHAP artifact for disaggregated
# evaluation and dashboard filtering. Not model inputs.
META_COLUMNS: list[str] = [
    "year", "postal_code_3", "gementee", "gementee_code", "age_group", "gender",
]

# Target-encoded for linear and tree models alike; ordinal codes turned out
# to be nearly useless to the trees.
HIGH_CARD_FEATURES: list[str] = ["postal_code_3"]
LOW_CARD_FEATURES: list[str] = ["age_group", "gender"]

WEIGHT_COLUMN: str = "insured_years"

UNKNOWN_CATEGORY: str = "UNKNOWN"

# Ordered by tier; drives training and dashboard ordering.
MODEL_KEYS: list[str] = [
    "linear", "ridge", "lasso",          # tier 1
    "random_forest", "gbm", "xgboost",   # tier 2
    "lightgbm", "catboost",              # tier 3
    "stacked",                           # tier 4
]

MODEL_LABELS: dict[str, str] = {
    "linear": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "gbm": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "stacked": "Stacked Ensemble",
}

# model key -> encoder family (see encoders.py)
MODEL_ENCODER: dict[str, str] = {
    "linear": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "random_forest": "tree",
    "gbm": "tree",
    "xgboost": "tree",
    "lightgbm": "tree",
    "catboost": "catboost",
}

# Kept compact: tuning runs once on the total target and the best params are
# reused to refit per-category. Plain 'linear' has nothing to tune.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ridge":         {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "lasso":         {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [16, None],
        "min_samples_leaf": [5, 20],
        "max_features": ["sqrt", 0.5],
    },
    "gbm": {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "xgboost": {
        "n_estimators": [400, 800],
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lightgbm": {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [400, 800],
        "feature_fraction": [0.8, 1.0],
    },
    "catboost": {
        "depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "iterations": [400, 800],
        "l2_leaf_reg": [3, 5],
    },
}

# Chosen by OOF meta-CV on the total target (METHODOLOGY_PHASE2.md §7.4).
# Random Forest adds the only non-boosting perspective; GBM and Ridge added
# nothing beyond noise.
STACK_BASE_LEARNERS: list[str] = ["xgboost", "lightgbm", "catboost", "random_forest"]

# MAPE is undefined near zero actuals; only score rows at/above this floor.
MAPE_FLOOR_EUR: float = 1.0

# SHAP is pre-computed offline by shap_compute.py and only read by the
# dashboard — never recomputed in a callback.
#
# Tier 1 is interpretable through its coefficients; the tier-4 stack is out of
# scope, since its meta-model sees base predictions rather than the original
# features, so there is no unified attribution to give.
SHAP_MODELS: list[str] = ["random_forest", "xgboost", "lightgbm", "catboost"]

# The headline RQ2 target; the feature set is shared across all 24 anyway.
SHAP_TARGET: str = TARGET_TOTAL

# The same held-out rows feed every model, so chart 6.3 compares like with
# like. mean(|SHAP|) is stable at this size.
SHAP_SAMPLE_SIZE: int = 20_000
