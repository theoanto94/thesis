"""Central configuration for Phase 2 modeling.

Single source of truth for the random seed, filesystem paths, the target
definitions (total + per-category cost rates), the structural feature set, and
the per-model hyperparameter grids. Importing this module performs no I/O.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global seed used for every model, CV split, and shuffle in Phase 2.
RANDOM_STATE: int = 42

#: Number of cross-validation folds used for tuning (within 2011-2020 only).
N_SPLITS: int = 5

#: Training-set sizes (row counts) at which the learning curve is sampled.
#: Log-spaced rather than evenly-spaced: a learning curve is most informative
#: where it is steepest — the small-data regime — so points are dense below
#: ~80k and sparse on the already-flat tail. The full training size is always
#: appended (and any size >= it is dropped) by ``evaluate.learning_curve``.
LEARNING_CURVE_SIZES: tuple[int, ...] = (
    2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 150_000,
)

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root)
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = ROOT / "identifier.db"

ARTIFACTS: Path = ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS / "models"
PARAMS_DIR: Path = ARTIFACTS / "params"
METRICS_DIR: Path = ARTIFACTS / "metrics"
PREDICTIONS_DIR: Path = ARTIFACTS / "predictions"
LEARNING_CURVES_DIR: Path = ARTIFACTS / "learning_curves"
TIMINGS_DIR: Path = ARTIFACTS / "timings"
SHAP_DIR: Path = ARTIFACTS / "shap"   # Phase 3 — pre-computed SHAP matrices

#: All artifact subdirectories, created on demand by ``ensure_dirs``.
_ALL_DIRS = (
    ARTIFACTS, MODELS_DIR, PARAMS_DIR, METRICS_DIR,
    PREDICTIONS_DIR, LEARNING_CURVES_DIR, TIMINGS_DIR, SHAP_DIR,
)


def ensure_dirs() -> None:
    """Create the artifact directory tree if it does not yet exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data split (temporal hold-out — forecasting setup, never a random split)
# ---------------------------------------------------------------------------

TRAIN_YEARS: range = range(2011, 2021)   # 2011-2020 inclusive
TEST_YEARS: range = range(2021, 2024)    # 2021-2023 inclusive

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# The 23 canonical real cost categories — identical to ``COST_COLUMNS`` in
# app.py (the EDA dashboard), so the per-category split here matches the total
# used throughout Phase 1. The mental-health roll-up column is used (not its
# sub-components) to avoid double counting.

#: display label -> deflated cost column in ``claims``
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

#: The 23 cost columns whose sum defines the total cost numerator.
COST_COLUMNS: list[str] = list(COST_CATEGORIES.values())

#: Stable machine key for each target. ``"total"`` is the primary RQ2 target;
#: the remaining 23 are the per-category targets. Derived from the cost column
#: by stripping the ``costs_`` prefix and ``_real`` suffix.
TARGET_TOTAL: str = "total"


def _target_key(cost_col: str) -> str:
    """``costs_pharmacy_real`` -> ``pharmacy``."""
    return cost_col[len("costs_"):-len("_real")]


#: target key -> (display label, [cost columns summed into the numerator])
TARGETS: dict[str, tuple[str, list[str]]] = {
    TARGET_TOTAL: ("Total (all categories)", COST_COLUMNS),
    **{
        _target_key(col): (label, [col])
        for label, col in COST_CATEGORIES.items()
    },
}

#: target key -> display label (convenience view used by the dashboard).
TARGET_LABELS: dict[str, str] = {k: v[0] for k, v in TARGETS.items()}

# ---------------------------------------------------------------------------
# Features (structural only — NO cost columns, NO lags => no target leakage)
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: list[str] = [
    "age_group",       # 19 ordered 5-year bands
    "gender",          # M / V
    "postal_code_3",   # ~800 levels (high cardinality, geographic)
    # gementee_code dropped: redundant with postal_code_3 (ablation showed
    # ~€1 RMSE either way). It remains available as dashboard metadata
    # (see modeling/data.py _META_COLUMNS) for region filtering/segmentation.
]

NUMERIC_FEATURES: list[str] = [
    "year",                # temporal index
    "insured_years",       # population exposure (also the sample weight)
    "number_of_insured",   # headcount
    "latitude",            # geographic coordinate
    "longitude",
]

FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#: High-cardinality categorical(s) — target-encoded for both linear and tree
#: models (ablation showed ordinal codes are nearly useless to trees).
HIGH_CARD_FEATURES: list[str] = ["postal_code_3"]
LOW_CARD_FEATURES: list[str] = ["age_group", "gender"]

#: Column carrying the population weight (sample_weight + metric weight).
WEIGHT_COLUMN: str = "insured_years"

#: Sentinel used when a categorical value is missing (e.g. null gementee_code).
UNKNOWN_CATEGORY: str = "UNKNOWN"

# ---------------------------------------------------------------------------
# Model registry & hyperparameter grids
# ---------------------------------------------------------------------------
# Grids are intentionally compact: tuning runs once on the total target via
# GridSearchCV (K=5), and the best params are reused to refit per-category.
# Param keys are bare estimator kwargs; tune.py maps them onto each estimator.

#: ordered model keys grouped by tier; drives training/dashboard ordering.
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

#: model key -> the encoder family it consumes (see encoders.py).
#: 'linear'  -> target-encode high-card + one-hot low-card + scale numerics
#: 'tree'    -> ordinal-encode all categoricals, pass numerics through
#: 'catboost'-> raw categoricals (native handling), pass numerics through
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

#: Tunable hyperparameter grids (tier 1-3). Tier-1 'linear' has no grid.
#: Grids cover the spec's named hyperparameters with 2 values each (plus a few
#: 3-value sweeps) to keep the single total-target grid search tractable.
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

#: Base learners whose OOF predictions feed the tier-4 stack. Selected
#: empirically by OOF meta-CV on the total target (see METHODOLOGY_PHASE2.md
#: §7.4): adding Random Forest to the three boosters improved CV RMSE and gave
#: the best held-out test RMSE. RandomForest contributes the only non-boosting
#: (bagging) perspective; CatBoost is near-redundant but retained as it is
#: harmless. GBM/Ridge added nothing beyond noise and are excluded.
STACK_BASE_LEARNERS: list[str] = ["xgboost", "lightgbm", "catboost", "random_forest"]

#: MAPE is undefined near zero actuals; only score rows at/above this floor (€).
MAPE_FLOOR_EUR: float = 1.0

# ---------------------------------------------------------------------------
# Phase 3 — SHAP explainability
# ---------------------------------------------------------------------------
# SHAP is pre-computed offline (modeling/shap_compute.py) and read by the
# dashboard; it is NEVER recomputed in a callback.

#: Tier 2-3 base learners explained via exact TreeSHAP. Tier-1 models are
#: interpretable through their coefficients (not SHAP), and the tier-4 stack is
#: out of scope — its meta-model operates on base predictions, not the original
#: features, so a unified SHAP attribution to the structural features does not
#: exist. This list mirrors the final stack composition (see STACK_BASE_LEARNERS).
SHAP_MODELS: list[str] = ["random_forest", "xgboost", "lightgbm", "catboost"]

#: SHAP explains the primary RQ2 target only — the total real cost rate. (The
#: structural feature set is shared across all 24 targets; the total is the
#: headline target whose drivers RQ2 asks about.)
SHAP_TARGET: str = TARGET_TOTAL

#: A shared, reproducible sample of held-out (2021-2023) instances on which
#: SHAP is computed. The SAME rows feed every model so the cross-model
#: comparison (chart 6.3) is like-for-like. Aggregate mean(|SHAP|) is stable at
#: this sample size; the per-row insured_years weight is persisted alongside so
#: importances stay population-weighted, consistent with the metric weighting.
SHAP_SAMPLE_SIZE: int = 20_000
