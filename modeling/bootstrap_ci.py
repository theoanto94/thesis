"""Bootstrap confidence intervals for the held-out test metrics (Chapter 6).

Percentile bootstrap over the test cells, total target. One shared index draw
per replicate, so per-model intervals and the model-vs-baseline deltas come
from identical resamples.

The resampling core, ``bootstrap_metrics``, is shared with the per-year tables
and the temporal track (``modeling.temporal.evaluate_extra``).

    uv run python -m modeling.bootstrap_ci   # -> artifacts/metrics/bootstrap_ci.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .metrics import METRIC_NAMES, weighted_metrics

N_BOOT = 1000
ALPHA = 0.05
BASELINE = "stacked"  # paired deltas are computed against this model

# Cell grain of one test row; the join key the model frames align on.
KEYS = ["year", "postal_code_3", "age_group", "gender"]

# mape_coverage is a diagnostic, not something to bootstrap.
CI_METRICS = tuple(m for m in METRIC_NAMES if m != "mape_coverage")


def _load_total_predictions() -> dict[str, pd.DataFrame]:
    """Load each model's total-target predictions, aligned on the cell keys."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(config.PREDICTIONS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=KEYS + ["target", "y_true", "y_pred", "weight"])
        df = (df[df["target"] == config.TARGET_TOTAL]
              .sort_values(KEYS).reset_index(drop=True))
        frames[path.stem] = df
    ref = next(iter(frames.values()))
    for name, df in frames.items():
        if len(df) != len(ref) or not df[KEYS].equals(ref[KEYS]):
            raise AssertionError(f"prediction rows of {name!r} do not align with the reference")
    return frames


def bootstrap_metrics(
    y: np.ndarray,
    w: np.ndarray,
    preds: dict[str, np.ndarray],
    baseline: str | None = None,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int = config.RANDOM_STATE,
) -> dict[str, dict]:
    """Paired percentile bootstrap of the weighted metrics for several models.

    All arrays are row-aligned: *y* the actuals, *w* the insured_years weights,
    *preds* one prediction array per model. Each replicate draws one index
    vector (cells with replacement) shared by every model, so the intervals of
    the RMSE difference against *baseline* are paired. Returns, per model,
    ``{metric: {point, ci_lo, ci_hi}}`` plus ``delta_rmse_vs_baseline`` for
    every non-baseline model when a baseline is given.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    p = {m: np.asarray(v, dtype=float) for m, v in preds.items()}
    models = list(p)
    n = len(y)

    rng = np.random.default_rng(seed)
    stats = {m: {k: np.empty(n_boot) for k in CI_METRICS} for m in models}
    delta_rmse = {m: np.empty(n_boot) for m in models if baseline and m != baseline}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # one shared draw -> paired across models
        yt, ww = y[idx], w[idx]
        per_model = {m: weighted_metrics(yt, p[m][idx], ww) for m in models}
        for m in models:
            for k in CI_METRICS:
                stats[m][k][b] = per_model[m][k]
        for m in delta_rmse:
            delta_rmse[m][b] = per_model[m]["rmse"] - per_model[baseline]["rmse"]

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    baseline_rmse = weighted_metrics(y, p[baseline], w)["rmse"] if baseline else None
    out: dict[str, dict] = {}
    for m in models:
        point = weighted_metrics(y, p[m], w)
        entry = {
            k: {"point": point[k],
                "ci_lo": float(np.percentile(stats[m][k], lo)),
                "ci_hi": float(np.percentile(stats[m][k], hi))}
            for k in CI_METRICS
        }
        if m in delta_rmse:
            entry["delta_rmse_vs_baseline"] = {
                "point": point["rmse"] - baseline_rmse,
                "ci_lo": float(np.percentile(delta_rmse[m], lo)),
                "ci_hi": float(np.percentile(delta_rmse[m], hi)),
            }
        out[m] = entry
    return out


def _print_table(results: dict[str, dict], baseline: str | None) -> None:
    """Print RMSE with its interval (and the paired delta) per model."""
    for m in sorted(results, key=lambda m: results[m]["rmse"]["point"]):
        e = results[m]
        line = (f"{m:>16}  RMSE {e['rmse']['point']:7.1f} "
                f"[{e['rmse']['ci_lo']:7.1f}, {e['rmse']['ci_hi']:7.1f}]")
        if "delta_rmse_vs_baseline" in e:
            d = e["delta_rmse_vs_baseline"]
            line += f"  dRMSE vs {baseline} {d['point']:+6.1f} [{d['ci_lo']:+6.1f}, {d['ci_hi']:+6.1f}]"
        print(line)


def main() -> None:
    """Pooled and per-test-year intervals for every model with saved predictions."""
    frames = _load_total_predictions()
    models = list(frames)
    ref = frames[models[0]]
    # Frames are row-aligned on KEYS, so actuals and weights are shared.
    y = ref["y_true"].to_numpy()
    w = ref["weight"].to_numpy()
    p = {m: frames[m]["y_pred"].to_numpy() for m in models}

    out = {
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "n_cells": int(len(y)),
        "baseline": BASELINE,
        "random_state": config.RANDOM_STATE,
        "models": bootstrap_metrics(y, w, p, baseline=BASELINE),
        "by_year": {},
    }
    print("pooled 2021-2023")
    _print_table(out["models"], BASELINE)

    years = ref["year"].to_numpy()
    for year in sorted(np.unique(years)):
        mask = years == year
        out["by_year"][str(int(year))] = {
            "n_cells": int(mask.sum()),
            "models": bootstrap_metrics(y[mask], w[mask], {m: v[mask] for m, v in p.items()},
                                        baseline=BASELINE),
        }
        print(f"\n{int(year)}")
        _print_table(out["by_year"][str(int(year))]["models"], BASELINE)

    dest = config.METRICS_DIR / "bootstrap_ci.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
