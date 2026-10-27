"""Uncertainty, horizon and comparability diagnostics for the temporal track.

Reads the row-level recursive predictions persisted by ``experiments`` (no
refitting) and writes to ``artifacts/temporal/metrics/``:

1. ``bootstrap_ci.json`` — paired cell-bootstrap 95% intervals for every row of
   the structural-mirror (A) and head-to-head (B) tables, with the RMSE
   difference against the deployed config (Ridge/base) and, for B, against
   the accuracy leader.
2. ``by_year.parquet`` — per scored year: the four metrics plus the mean
   signed residual (actual - predicted; positive = under-prediction).
3. ``phase2_same_rows.parquet`` / ``phase2_same_rows_ci.json`` — Phase 2
   (lag-free) models re-scored on exactly the rows of protocol A, next to the
   lagged temporal models, pooled and per year, with paired intervals.
4. ``ridge_coefficients.json`` — the deployed Ridge written out: intercept,
   coefficients on the standardised design and in raw units, the lag
   coefficient rho and the implied long-run annual increment.

    uv run python -m modeling.temporal.evaluate_extra
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .. import config as base
from ..bootstrap_ci import bootstrap_metrics
from ..metrics import weighted_metrics
from . import config as tcfg
from .experiments import PROTOCOL_H2H, PROTOCOL_MIRROR

KEYS = ["year", "postal_code_3", "age_group", "gender"]
DEPLOYED = f"{tcfg.DEPLOYED_MODEL}/{tcfg.DEPLOYED_TREATMENT}"

# Phase 2 models placed next to the temporal ones on the protocol-A rows.
PHASE2_COMPARE = ["stacked", "lightgbm", "xgboost", "ridge", "ridge_ag"]


def load_protocol(protocol: str) -> dict[str, pd.DataFrame]:
    """Row-aligned predictions of every config evaluated under *protocol*."""
    frames = {}
    for path in sorted(tcfg.T_PREDICTIONS_DIR.glob(f"{protocol}__*.parquet")):
        _, treatment, model = path.stem.split("__")
        df = pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)
        frames[f"{model}/{treatment}"] = df
    ref = next(iter(frames.values()))
    for name, df in frames.items():
        assert df[KEYS].equals(ref[KEYS]), f"{protocol}: rows of {name} do not align"
    return frames


def _arrays(frames: dict[str, pd.DataFrame]):
    """Shared actuals/weights and the per-config prediction arrays."""
    ref = next(iter(frames.values()))
    y = ref["y_true"].to_numpy(dtype=float)
    w = ref["weight"].to_numpy(dtype=float)
    return y, w, {k: f["y_pred"].to_numpy(dtype=float) for k, f in frames.items()}


def protocol_ci(frames: dict[str, pd.DataFrame], baselines: list[str]) -> dict:
    """Paired intervals, one bootstrap per baseline (identical seeds -> same draws)."""
    y, w, p = _arrays(frames)
    out = bootstrap_metrics(y, w, p, baseline=baselines[0])
    for extra in baselines[1:]:
        other = bootstrap_metrics(y, w, p, baseline=extra)
        for m in out:
            if "delta_rmse_vs_baseline" in other[m]:
                out[m][f"delta_rmse_vs_{extra}"] = other[m]["delta_rmse_vs_baseline"]
    for m in out:
        if "delta_rmse_vs_baseline" in out[m]:
            out[m][f"delta_rmse_vs_{baselines[0]}"] = out[m].pop("delta_rmse_vs_baseline")
    return {"n_rows": int(len(y)), "baselines": baselines, "models": out}


def per_year(frames: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    """Metrics and mean signed residual per config x scored year (+ pooled)."""
    rows = []
    for name, df in frames.items():
        for year, sub in [("all", df), *df.groupby("year")]:
            yt, yp, w = (sub[c].to_numpy(dtype=float) for c in ("y_true", "y_pred", "weight"))
            rows.append({
                "protocol": protocol, "config": name, "year": str(year),
                **weighted_metrics(yt, yp, w),
                "mean_signed_residual": float(np.average(yt - yp, weights=w)),
                "actual_mean": float(np.average(yt, weights=w)),
                "predicted_mean": float(np.average(yp, weights=w)),
                "n": int(len(sub)),
            })
    return pd.DataFrame(rows)


def phase2_same_rows(mirror: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Phase 2 total-target predictions restricted to the protocol-A rows."""
    ref = next(iter(mirror.values()))[KEYS + ["y_true", "weight"]]
    frames = {f"temporal:{k}": v for k, v in mirror.items()}
    for model in PHASE2_COMPARE:
        path = base.PREDICTIONS_DIR / f"{model}.parquet"
        if not path.exists():
            continue
        p2 = pd.read_parquet(path, columns=KEYS + ["target", "y_true", "y_pred"])
        p2 = p2[p2["target"] == base.TARGET_TOTAL].drop(columns="target")
        for c in KEYS[1:]:
            p2[c] = p2[c].astype(str)
        merged = ref.merge(p2, on=KEYS, how="left", suffixes=("", "_p2"), validate="one_to_one")
        assert merged["y_pred"].notna().all(), f"{model}: protocol-A rows missing in Phase 2"
        assert np.allclose(merged["y_true"], merged["y_true_p2"], rtol=1e-4, atol=0.05), \
            f"{model}: actuals differ between tracks"
        frames[f"phase2:{model}"] = merged.assign(y_pred=merged["y_pred"].astype(float))
    table = per_year(frames, "same_rows")
    y, w, p = _arrays(frames)
    ci = bootstrap_metrics(y, w, p, baseline="phase2:stacked")
    return table, {"n_rows": int(len(y)), "baseline": "phase2:stacked", "models": ci}


def ridge_coefficients(model_key: str = tcfg.DEPLOYED_MODEL,
                       treatment: str = tcfg.DEPLOYED_TREATMENT) -> dict:
    """Deployed Ridge in standardised and raw units, plus rho and the long-run slope."""
    pipe = joblib.load(tcfg.final_model_path(model_key, treatment))
    pre, ridge = pipe.named_steps["pre"], pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    coef = dict(zip(names, ridge.coef_.tolist()))

    raw = {}
    num = pre.named_transformers_["num_scale"]
    for name, scale in zip(num.feature_names_in_, num.scale_):
        raw[name] = coef[name] / scale
    geo_scale = pre.named_transformers_["geo_target"].named_steps["scale"].scale_[0]
    geo_name = [n for n in names if n.startswith("postal_code_3")][0]
    raw["postal_code_3_target_encoding"] = coef[geo_name] / geo_scale
    for name in names:
        if name.startswith(("age_group_", "gender_")):
            raw[name] = coef[name]  # one-hots are unscaled: already euros

    rho = raw[tcfg.LAG_COL]
    year_slope = raw["year"]
    return {
        "model": model_key, "treatment": treatment,
        "alpha": float(ridge.alpha),
        "intercept_standardised_design": float(ridge.intercept_),
        "coef_standardised": coef,
        "coef_raw_units": raw,
        "rho_lag": rho,
        "year_coef_eur_per_year": year_slope,
        "long_run_annual_increment_eur": year_slope / (1.0 - rho),
        "note": ("rate_t = b0 + rho*rate_{t-1} + theta*year + age/gender/postcode "
                 "terms + exposure/headcount/coordinates; long-run increment "
                 "theta/(1-rho) is the steady-state yearly rise of a cell's rate"),
    }


def main() -> None:
    """Compute and write all four diagnostics; print the headline numbers."""
    mirror, h2h = load_protocol(PROTOCOL_MIRROR), load_protocol(PROTOCOL_H2H)
    leader = json.loads(tcfg.WINNER_PATH.read_text())
    leader_key = f"{leader['model']}/{leader['treatment']}"

    ci = {
        PROTOCOL_MIRROR: protocol_ci(mirror, [DEPLOYED]),
        PROTOCOL_H2H: protocol_ci(h2h, [DEPLOYED, leader_key]),
    }
    (tcfg.T_METRICS_DIR / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2))

    by_year = pd.concat([per_year(mirror, PROTOCOL_MIRROR), per_year(h2h, PROTOCOL_H2H)],
                        ignore_index=True)
    by_year.to_parquet(tcfg.T_METRICS_DIR / "by_year.parquet", index=False)

    same_rows, same_ci = phase2_same_rows(mirror)
    same_rows.to_parquet(tcfg.T_METRICS_DIR / "phase2_same_rows.parquet", index=False)
    (tcfg.T_METRICS_DIR / "phase2_same_rows_ci.json").write_text(json.dumps(same_ci, indent=2))

    coefs = ridge_coefficients()
    (tcfg.T_METRICS_DIR / "ridge_coefficients.json").write_text(json.dumps(coefs, indent=2))

    pd.set_option("display.width", 200)
    for proto, block in ci.items():
        print(f"\n{proto} (n={block['n_rows']:,})")
        for m, e in sorted(block["models"].items(), key=lambda kv: kv[1]["rmse"]["point"]):
            line = f"  {m:22s} RMSE {e['rmse']['point']:6.1f} [{e['rmse']['ci_lo']:6.1f}, {e['rmse']['ci_hi']:6.1f}]"
            for k, d in e.items():
                if k.startswith("delta_rmse_vs_"):
                    line += f"  {k[14:]}: {d['point']:+5.1f} [{d['ci_lo']:+5.1f}, {d['ci_hi']:+5.1f}]"
            print(line)
    cols = ["protocol", "config", "year", "rmse", "r2", "mae", "mape", "mean_signed_residual"]
    print("\n", by_year[cols].round(2).to_string(index=False))
    print("\nSame rows as protocol A:\n",
          same_rows.pivot(index="config", columns="year", values="rmse").round(1).to_string())
    print("  bias:\n", same_rows.pivot(index="config", columns="year",
                                      values="mean_signed_residual").round(1).to_string())
    for m, e in same_ci["models"].items():
        if "delta_rmse_vs_baseline" in e:
            d = e["delta_rmse_vs_baseline"]
            print(f"  {m:28s} dRMSE vs phase2:stacked {d['point']:+6.1f} [{d['ci_lo']:+6.1f}, {d['ci_hi']:+6.1f}]")
    print(f"\nRidge: rho={coefs['rho_lag']:.3f}, year={coefs['year_coef_eur_per_year']:.2f} €/yr, "
          f"long-run +{coefs['long_run_annual_increment_eur']:.1f} €/yr")
    print(f"wrote {tcfg.T_METRICS_DIR}")


if __name__ == "__main__":
    main()
