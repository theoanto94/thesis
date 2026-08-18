"""Phase 3 — offline SHAP computation (exact TreeSHAP).

Computes per-instance SHAP values for the Tier 2-3 base learners
(``config.SHAP_MODELS``) on a shared, reproducible sample of the held-out
2021-2023 instances, explaining the *total* real cost-rate target. One parquet
is written per model to ``artifacts/shap/``:

    shap_<feature>   per-instance SHAP contribution, in the model's input space
    val_<feature>    the feature value the model consumed (beeswarm colouring)
    base_value       the explainer's expected value (constant per model)
    <meta...>        year, postal_code_3, gementee, gementee_code, age_group,
                     gender — for the dashboard's region/age/year filtering
    weight           insured_years, so importances stay population-weighted

The dashboard loads these matrices and NEVER recomputes SHAP in a callback.

Notes
-----
* The SAME sampled rows are used for every model, so the cross-model feature
  importance comparison is like-for-like.
* Tree-family models (RF / XGBoost / LightGBM) are scikit-learn Pipelines; SHAP
  runs on the post-encoding matrix (ordinal codes + raw numerics), and the
  encoded columns are mapped back to bare feature names. CatBoost consumes raw
  categoricals via a ``Pool``; its categorical ``val_`` columns are factorised
  to integer codes purely for colour-coding.
* Random Forest's total-target model was pruned after Phase 2 stacking; it is
  retrained here from the persisted tuned params (and re-persisted) before its
  SHAP is computed.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import shap

from . import config, data, factory

#: Canonical feature order for the output columns (matches config.FEATURES).
_FEATURES = config.FEATURES
_META = data._META_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tuned_params(model_key: str) -> dict:
    """Load the persisted tuned hyperparameters for *model_key* (bare kwargs)."""
    path = config.PARAMS_DIR / f"{model_key}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("best_params", {})


def _fitted_total_model(model_key: str, ds: data.Dataset):
    """Return the fitted total-target model, retraining RF if it was pruned."""
    path = config.MODELS_DIR / f"{model_key}__{config.SHAP_TARGET}.joblib"
    if path.exists():
        return joblib.load(path)
    # Model file missing (e.g. random_forest, pruned after stacking): refit
    # from tuned params on the full training set and re-persist for reuse.
    print(f"  [{model_key}] no persisted model — retraining total target …")
    est = factory.fit_for_target(
        model_key,
        _tuned_params(model_key),
        ds.X_train,
        ds.y_train[config.SHAP_TARGET],
        ds.w_train,
    )
    joblib.dump(est, path)
    return est


def _sample_indices(n_rows: int) -> np.ndarray:
    """Reproducible row indices for the shared SHAP sample (sorted)."""
    size = min(config.SHAP_SAMPLE_SIZE, n_rows)
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(n_rows, size=size, replace=False)
    idx.sort()
    return idx


def _bare_name(encoded_name: str) -> str:
    """``cat_ordinal__age_group`` / ``num_pass__year`` -> bare feature name."""
    return encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name


def _shap_pipeline(estimator, X_sample: pd.DataFrame):
    """SHAP for a tree-family Pipeline -> (shap[FEATURES], val[FEATURES], base).

    Runs TreeSHAP on the post-encoding matrix, then reorders both the SHAP
    values and the encoded feature values into ``config.FEATURES`` order.
    """
    pre = estimator.named_steps["pre"]
    model = estimator.named_steps["model"]

    X_enc = np.asarray(pre.transform(X_sample), dtype=float)
    enc_names = [_bare_name(n) for n in pre.get_feature_names_out()]
    col = {name: j for j, name in enumerate(enc_names)}

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X_enc), dtype=float)
    base = float(np.ravel(explainer.expected_value)[0])

    order = [col[f] for f in _FEATURES]
    return sv[:, order], X_enc[:, order], base


def _shap_catboost(estimator, X_sample: pd.DataFrame):
    """SHAP for a bare CatBoost model -> (shap[FEATURES], val[FEATURES], base)."""
    from catboost import Pool

    pool = Pool(X_sample[_FEATURES], cat_features=config.CATEGORICAL_FEATURES)
    explainer = shap.TreeExplainer(estimator)
    sv = np.asarray(explainer.shap_values(pool), dtype=float)  # FEATURES order
    base = float(np.ravel(explainer.expected_value)[0])

    # Model-input value for colouring: raw numerics, factorised categoricals.
    vals = np.empty((len(X_sample), len(_FEATURES)), dtype=float)
    for j, f in enumerate(_FEATURES):
        if f in config.CATEGORICAL_FEATURES:
            vals[:, j] = pd.factorize(X_sample[f])[0]
        else:
            vals[:, j] = X_sample[f].to_numpy(dtype=float)
    return sv, vals, base


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def compute_for_model(model_key: str, ds: data.Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Compute and persist the SHAP matrix for one model; return it."""
    X_sample = ds.X_test.iloc[idx].reset_index(drop=True)
    meta_sample = ds.meta_test.iloc[idx].reset_index(drop=True)
    w_sample = ds.w_test.iloc[idx].reset_index(drop=True)

    estimator = _fitted_total_model(model_key, ds)

    t0 = time.perf_counter()
    if factory.is_pipeline(estimator):
        sv, vals, base = _shap_pipeline(estimator, X_sample)
    else:  # CatBoost
        sv, vals, base = _shap_catboost(estimator, X_sample)
    elapsed = time.perf_counter() - t0

    out = pd.DataFrame(index=range(len(idx)))
    for j, f in enumerate(_FEATURES):
        out[f"shap_{f}"] = sv[:, j].astype("float32")
        out[f"val_{f}"] = vals[:, j].astype("float32")
    out["base_value"] = np.float32(base)
    for m in _META:
        out[m] = meta_sample[m].astype(str).values
    out["weight"] = w_sample.to_numpy(dtype="float32")

    dest = config.SHAP_DIR / f"{model_key}.parquet"
    out.to_parquet(dest, index=False)

    mean_abs = (np.abs(sv) * w_sample.to_numpy()[:, None]).sum(0) / w_sample.sum()
    top = sorted(zip(_FEATURES, mean_abs), key=lambda t: -t[1])[:4]
    top_str = ", ".join(f"{f}={v:.1f}" for f, v in top)
    print(f"  [{model_key}] {len(idx):,} rows · {elapsed:5.1f}s · "
          f"base €{base:,.0f} · top mean|SHAP|: {top_str}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute SHAP matrices for every in-scope model on the shared sample."""
    config.ensure_dirs()
    print(f"Loading dataset … (SHAP target = {config.SHAP_TARGET!r})")
    ds = data.load_dataset()
    idx = _sample_indices(len(ds.X_test))
    print(f"Shared SHAP sample: {len(idx):,} of {len(ds.X_test):,} held-out rows "
          f"(seed {config.RANDOM_STATE})")
    print(f"Models: {config.SHAP_MODELS}\n")

    for model_key in config.SHAP_MODELS:
        compute_for_model(model_key, ds, idx)

    print("\nSHAP computation complete →", config.SHAP_DIR)


if __name__ == "__main__":
    main()
