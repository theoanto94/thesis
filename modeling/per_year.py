"""Per-test-year metrics and signed bias for every model (post-hoc).

The held-out period 2021-2023 is a forecast horizon, so pooled metrics hide
whether error grows with distance from the training window. This reads the
row-level predictions written by ``evaluate`` (no retraining, no test-set
refitting) and reports, per model x test year on the total target:

* the weighted R2 / RMSE / MAE / MAPE (``metrics.disaggregate`` by year);
* the mean signed residual (actual - predicted, insured-years weighted):
  positive = under-prediction;
* the weighted actual and predicted national mean rate.

A second table splits the signed residual by decile of the *predicted* rate
within each year (a calibration view: binning on the actual would show a
spurious positive tail from regression to the mean), which separates bias in
the bulk from under-prediction of the high-cost cells.

    uv run python -m modeling.per_year   # -> artifacts/metrics/by_year*.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .metrics import disaggregate

N_TAIL_BINS = 10


def _load_predictions(target: str = config.TARGET_TOTAL) -> dict[str, pd.DataFrame]:
    """Load every model's row-level test predictions for *target*."""
    frames = {}
    for path in sorted(config.PREDICTIONS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["year", "target", "y_true", "y_pred", "weight"])
        frames[path.stem] = df[df["target"] == target].reset_index(drop=True)
    return frames


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Insured-years-weighted mean."""
    return float(np.average(x, weights=w))


def per_year_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weighted metrics, signed bias and mean levels per model x year (+ pooled)."""
    rows = []
    for model, df in frames.items():
        yt, yp, w = (df[c].to_numpy(dtype=float) for c in ("y_true", "y_pred", "weight"))
        per_year = disaggregate(df, yt, yp, w, by="year")
        pooled = disaggregate(df.assign(year="all"), yt, yp, w, by="year")
        for _, r in pd.concat([per_year, pooled]).iterrows():
            sel = np.ones(len(df), bool) if r["year"] == "all" else (df["year"] == r["year"]).to_numpy()
            rows.append({
                "model": model,
                "year": str(r["year"]),
                **{k: r[k] for k in ("r2", "rmse", "mae", "mape", "n", "exposure")},
                "mean_signed_residual": _weighted_mean(yt[sel] - yp[sel], w[sel]),
                "actual_mean": _weighted_mean(yt[sel], w[sel]),
                "predicted_mean": _weighted_mean(yp[sel], w[sel]),
            })
    out = pd.DataFrame(rows)
    out["bias_pct"] = 100 * out["mean_signed_residual"] / out["actual_mean"]
    return out


def tail_bias_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean signed residual by within-year decile of the *predicted* rate.

    Deciles are exposure-weighted so each bin holds ~10% of insured-years.
    """
    rows = []
    for model, df in frames.items():
        for year, sub in df.groupby("year"):
            sub = sub.sort_values("y_pred")
            cumw = sub["weight"].cumsum() / sub["weight"].sum()
            bins = np.minimum((cumw.to_numpy() * N_TAIL_BINS).astype(int), N_TAIL_BINS - 1)
            for b, part in sub.groupby(bins):
                w = part["weight"].to_numpy(dtype=float)
                rows.append({
                    "model": model, "year": int(year), "decile": int(b) + 1,
                    "predicted_mean": _weighted_mean(part["y_pred"].to_numpy(dtype=float), w),
                    "actual_mean": _weighted_mean(part["y_true"].to_numpy(dtype=float), w),
                    "mean_signed_residual": _weighted_mean(
                        (part["y_true"] - part["y_pred"]).to_numpy(dtype=float), w),
                })
    return pd.DataFrame(rows)


def main() -> None:
    """Write the per-year and tail-bias tables and print the headline view."""
    frames = _load_predictions()
    by_year = per_year_table(frames)
    tail = tail_bias_table(frames)
    by_year.to_parquet(config.METRICS_DIR / "by_year.parquet", index=False)
    tail.to_parquet(config.METRICS_DIR / "by_year_tail_bias.parquet", index=False)

    pd.set_option("display.width", 200)
    order = {k: i for i, k in enumerate(config.MODEL_KEYS)}
    view = by_year.sort_values(["model", "year"], key=lambda s: s.map(order) if s.name == "model" else s)
    print(view[["model", "year", "r2", "rmse", "mae", "mape", "mean_signed_residual",
                "actual_mean", "predicted_mean", "bias_pct"]].round(3).to_string(index=False))
    stk = tail[tail["model"] == "stacked"].pivot(index="decile", columns="year", values="mean_signed_residual")
    print("\nStacked: mean signed residual by predicted-rate decile\n", stk.round(1).to_string())
    print(f"\nwrote {config.METRICS_DIR / 'by_year.parquet'}")


if __name__ == "__main__":
    main()
