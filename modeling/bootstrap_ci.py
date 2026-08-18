"""Bootstrap confidence intervals for the held-out test metrics (Chapter 6).

Percentile bootstrap over the 87,895 test cells (postcode3 x age_group x
gender x year, total target, 2021-2023). Rows are resampled with replacement
using ONE shared index draw per replicate, so per-model intervals and paired
model-vs-stacked deltas come from identical resamples (paired bootstrap).

Metrics are computed by ``modeling.metrics.weighted_metrics`` (the canonical
insured-years-weighted RMSE / MAE / R2 and floored MAPE).

Usage:
    uv run python -m modeling.bootstrap_ci        # writes artifacts/metrics/bootstrap_ci.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from modeling.config import METRICS_DIR, PREDICTIONS_DIR, RANDOM_STATE
from modeling.metrics import weighted_metrics

N_BOOT = 1000
ALPHA = 0.05
BASELINE = "stacked"  # paired deltas are computed against this model
KEYS = ["year", "postal_code_3", "age_group", "gender"]


def _load_total_predictions() -> dict[str, pd.DataFrame]:
    """Load each model's total-target predictions, aligned on the cell keys."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=KEYS + ["target", "y_true", "y_pred", "weight"])
        df = df[df["target"] == "total"].sort_values(KEYS).reset_index(drop=True)
        frames[path.stem] = df
    ref = next(iter(frames.values()))
    for name, df in frames.items():
        if len(df) != len(ref) or not df[KEYS].equals(ref[KEYS]):
            raise AssertionError(f"prediction rows of {name!r} do not align with the reference")
    return frames


def main() -> None:
    frames = _load_total_predictions()
    models = list(frames)
    n = len(frames[models[0]])
    y = {m: frames[m]["y_true"].to_numpy() for m in models}
    p = {m: frames[m]["y_pred"].to_numpy() for m in models}
    w = frames[models[0]]["weight"].to_numpy()

    rng = np.random.default_rng(RANDOM_STATE)
    stats = {m: {k: np.empty(N_BOOT) for k in ("rmse", "mae", "r2", "mape")} for m in models}
    delta_rmse = {m: np.empty(N_BOOT) for m in models if m != BASELINE}

    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)  # one shared draw -> paired across models
        per_model = {m: weighted_metrics(y[m][idx], p[m][idx], w[idx]) for m in models}
        for m in models:
            for k in ("rmse", "mae", "r2", "mape"):
                stats[m][k][b] = per_model[m][k]
        for m in delta_rmse:
            delta_rmse[m][b] = per_model[m]["rmse"] - per_model[BASELINE]["rmse"]

    lo, hi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    out = {
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "n_cells": int(n),
        "baseline": BASELINE,
        "random_state": RANDOM_STATE,
        "models": {},
    }
    for m in models:
        point = weighted_metrics(y[m], p[m], w)
        entry = {
            k: {"point": point[k],
                "ci_lo": float(np.percentile(stats[m][k], lo)),
                "ci_hi": float(np.percentile(stats[m][k], hi))}
            for k in ("rmse", "mae", "r2", "mape")
        }
        if m != BASELINE:
            entry["delta_rmse_vs_baseline"] = {
                "point": point["rmse"] - weighted_metrics(y[BASELINE], p[BASELINE], w)["rmse"],
                "ci_lo": float(np.percentile(delta_rmse[m], lo)),
                "ci_hi": float(np.percentile(delta_rmse[m], hi)),
            }
        out["models"][m] = entry

    dest = METRICS_DIR / "bootstrap_ci.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    for m in sorted(models, key=lambda m: out["models"][m]["rmse"]["point"]):
        e = out["models"][m]
        line = (f"{m:>14}  RMSE {e['rmse']['point']:7.1f} "
                f"[{e['rmse']['ci_lo']:7.1f}, {e['rmse']['ci_hi']:7.1f}]")
        if "delta_rmse_vs_baseline" in e:
            d = e["delta_rmse_vs_baseline"]
            line += f"  dRMSE vs {BASELINE} {d['point']:+6.1f} [{d['ci_lo']:+6.1f}, {d['ci_hi']:+6.1f}]"
        print(line)


if __name__ == "__main__":
    main()
