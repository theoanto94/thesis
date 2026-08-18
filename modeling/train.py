"""Training orchestration — fit every base model on all 24 targets.

Each base model is refit on the full 2011-2020 training set for every target,
using the params tuned once on the total. Timing records feed the dashboard's
computational-cost table. The tier-4 stack is built in tier4_stacking.py.

    uv run python -m modeling.train            # all base models, all targets
    uv run python -m modeling.train xgboost    # a single model, all targets
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import joblib

from . import config, factory
from .data import Dataset, load_dataset
from .tune import load_best_params


def model_path(model_key: str, target: str) -> Path:
    """Filesystem path for a fitted ``(model, target)`` artifact."""
    return config.MODELS_DIR / f"{model_key}__{target}.joblib"


def fit_one(model_key: str, target: str, ds: Dataset):
    """Fit *model_key* on one *target* using its tuned params."""
    params = factory.apply_best_params(model_key, load_best_params(model_key))
    return factory.fit_for_target(model_key, params, ds.X_train,
                                  ds.y_train[target], ds.w_train.to_numpy())


def train_model(model_key: str, ds: Dataset) -> dict:
    """Fit *model_key* across every target, persist artifacts, return timings."""
    config.ensure_dirs()
    targets = list(config.TARGETS.keys())

    tracemalloc.start()
    t0 = time.time()
    size_bytes = 0
    for target in targets:
        est = fit_one(model_key, target, ds)
        path = model_path(model_key, target)
        joblib.dump(est, path)
        size_bytes += path.stat().st_size
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    record = {
        "model": model_key,
        "n_targets": len(targets),
        "train_seconds_total": round(elapsed, 1),
        "train_seconds_per_target": round(elapsed / len(targets), 2),
        # tracemalloc only sees Python-side allocations, so the C-backed tree
        # libraries under-report; on-disk size is the honest footprint proxy.
        "peak_python_memory_mb": round(peak / 1e6, 1),
        "model_size_mb": round(size_bytes / 1e6, 2),
    }
    (config.TIMINGS_DIR / f"{model_key}.json").write_text(json.dumps(record, indent=2))
    return record


def main(models: list[str] | None = None) -> None:
    """Train the requested base models (default: all) on every target."""
    ds = load_dataset()
    keys = models or factory.BASE_MODEL_KEYS
    for key in keys:
        print(f"Training {key} on {len(config.TARGETS)} targets...")
        rec = train_model(key, ds)
        print(f"  done in {rec['train_seconds_total']}s "
              f"({rec['train_seconds_per_target']}s/target, "
              f"{rec['model_size_mb']} MB on disk)")


if __name__ == "__main__":
    import sys
    arg_models = sys.argv[1:] or None
    main(models=arg_models)
