"""Recursive 2024-2030 projection and the DuckDB objects the dashboard reads.

Refits the deployed config (pinned in config.py, with the reasoning there) on
every transition through 2023, then rolls forward year by year from the 2023
actuals — each year's predicted total rate becomes the next year's lag.
Exposure follows the national growth scenario; rates are clipped at zero.

Only the total rate is modelled. Per-category costs on projected rows are
allocated by each cell's 2023 shares, so composition is held constant and the
dashboard's "Total" reproduces the modelled total exactly.

Writes claims_forecast (one row per 2023 cell x forecast year, mirroring the
claims column set with NULLs where the forecast produces nothing) and the
claims_extended view over both, flagged by is_projected.

    uv run python -m modeling.temporal.forecast                 # deployed config
    uv run python -m modeling.temporal.forecast --model lightgbm --treatment no_covid
"""

from __future__ import annotations

import argparse
import json

import duckdb
import joblib
import numpy as np
import pandas as pd

from .. import config as base
from . import config as tcfg
from .data import (
    build_panel,
    entity_index,
    load_train,
    prepare_X,
    rates_for_year,
    train_medians,
)
from .experiments import _Persistence, fit

# Copied unchanged from each cell's 2023 row.
_CARRIED = ["gender", "postal_code_3", "age_group",
            "gementee", "gementee_code", "latitude", "longitude"]


def load_winner() -> tuple[str, str] | None:
    """The experiment accuracy leader from winner.json, if experiments ran."""
    if not tcfg.WINNER_PATH.exists():
        return None
    rec = json.loads(tcfg.WINNER_PATH.read_text())
    return rec["model"], rec["treatment"]


def refit(model_key: str, treatment: str):
    """Refit on all transitions through 2023 -> (estimator, lat_med, lon_med).

    Persistence needs no fit, so nothing is persisted for it.
    """
    if model_key == "persistence":
        lat_med, lon_med = train_medians(build_panel(), tcfg.FORECAST_BASE_YEAR)
        return _Persistence(), lat_med, lon_med

    ds = load_train(treatment, train_end=tcfg.FORECAST_BASE_YEAR)
    print(f"Refitting {model_key} / {treatment} on {len(ds.X_train):,} rows "
          f"(years {sorted(int(y) for y in ds.meta_train['year'].unique())})...")
    est, seconds = fit(model_key, ds)
    tcfg.ensure_dirs()
    joblib.dump(est, tcfg.final_model_path(model_key, treatment))
    print(f"  fitted in {seconds:,.1f}s -> {tcfg.final_model_path(model_key, treatment)}")
    return est, ds.lat_med, ds.lon_med


def build_forecast(estimator, lat_med: float, lon_med: float) -> pd.DataFrame:
    """Roll the refit winner forward through 2024-2030.

    Every 2023 row is a base cell — with a single t-1 lag the seed is 2023
    itself, so nothing drops out for incomplete history.
    """
    panel = build_panel()
    base_rows = panel[panel["year"] == tcfg.FORECAST_BASE_YEAR].copy()
    ent = entity_index(base_rows)
    history = rates_for_year(panel, tcfg.FORECAST_BASE_YEAR).reindex(ent)
    assert not history.isnull().any(), "Missing 2023 seed rates"
    print(f"Forecast base: {len(base_rows):,} 2023 cells")

    iy_2023 = base_rows[base.WEIGHT_COLUMN].to_numpy(dtype=float)
    n_ins_2023 = base_rows["number_of_insured"].to_numpy(dtype=float)

    out: list[pd.DataFrame] = []
    for h in tcfg.FORECAST_YEARS:
        factor = (1.0 + tcfg.POPULATION_GROWTH_RATE) ** (h - tcfg.FORECAST_BASE_YEAR)
        work = base_rows.copy()
        work["year"] = h
        work["insured_years"] = iy_2023 * factor
        work["number_of_insured"] = np.round(n_ins_2023 * factor)
        work[tcfg.LAG_COL] = history.reindex(ent).to_numpy()

        X_h = prepare_X(work, lat_med, lon_med)
        rates = np.clip(estimator.predict(X_h), 0.0, None)
        history = pd.Series(rates, index=ent)

        year_df = work[_CARRIED].copy()
        year_df["year"] = h
        year_df["insured_years"] = work["insured_years"].to_numpy()
        year_df["number_of_insured"] = work["number_of_insured"].to_numpy()
        year_df["rate_total"] = rates
        out.append(year_df)

        national = float(np.average(rates, weights=work["insured_years"]))
        print(f"  {h}: national total rate €{national:,.0f} per insured-year")

    return pd.concat(out, ignore_index=True)


def allocate_categories(fc: pd.DataFrame) -> pd.DataFrame:
    """Split each projected total into the 23 categories by 2023 shares.

    Cells whose 2023 total is zero or missing take the national shares. Shares
    sum to one, so sum(costs_<c>_real) == rate_total * insured_years per row.
    """
    panel = build_panel()
    rows_2023 = panel[panel["year"] == tcfg.FORECAST_BASE_YEAR]
    costs_2023 = rows_2023[base.COST_COLUMNS].to_numpy(dtype=float)
    totals = costs_2023.sum(axis=1)

    national = costs_2023.sum(axis=0) / totals.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = costs_2023 / totals[:, None]
    degenerate = ~np.isfinite(shares).all(axis=1) | (totals <= 0)
    shares[degenerate] = national
    n_fallback = int(degenerate.sum())
    if n_fallback:
        print(f"  category shares: {n_fallback:,} cells fell back to national 2023 shares")
    share_df = pd.DataFrame(shares, columns=base.COST_COLUMNS,
                            index=entity_index(rows_2023))

    cell_shares = share_df.reindex(entity_index(fc)).to_numpy()
    assert np.isfinite(cell_shares).all(), "Unmatched forecast cell in 2023 shares"
    total_costs = (fc["rate_total"] * fc["insured_years"]).to_numpy()

    fc = fc.copy()
    fc[base.COST_COLUMNS] = np.round(cell_shares * total_costs[:, None], 2)
    return fc


def projected_cagr(fc: pd.DataFrame) -> float:
    """Compound annual growth from the actual 2023 rate to the projected 2030."""
    panel = build_panel()
    rows_2023 = panel[panel["year"] == tcfg.FORECAST_BASE_YEAR]
    rate_2023 = float(np.average(rows_2023["rate_total"],
                                 weights=rows_2023[base.WEIGHT_COLUMN]))
    last_year = max(tcfg.FORECAST_YEARS)
    last = fc[fc["year"] == last_year]
    rate_last = float(np.average(last["rate_total"], weights=last["insured_years"]))
    return (rate_last / rate_2023) ** (1.0 / (last_year - tcfg.FORECAST_BASE_YEAR)) - 1.0


def summarize(fc: pd.DataFrame, model_key: str, treatment: str) -> dict:
    """National projected rate by year plus the CAGR diagnostic."""
    by_year = {
        int(y): round(float(np.average(g["rate_total"], weights=g["insured_years"])), 2)
        for y, g in fc.groupby("year")
    }
    cagr = projected_cagr(fc)
    print(f"\nProjected 2023-2030 CAGR: {cagr:+.2%} "
          f"(realized 2011-2023: {tcfg.REALIZED_CAGR:+.2%}) — reported "
          "diagnostic, not a selection rule")
    return {
        "model": model_key,
        "treatment": treatment,
        "national_rate_by_year": by_year,
        "projected_cagr_2023_2030": round(float(cagr), 5),
        "realized_cagr_2011_2023": tcfg.REALIZED_CAGR,
    }


def write_db(fc: pd.DataFrame) -> None:
    """Create ``claims_forecast`` and the ``claims_extended`` view.

    Opens the database for writing, so the dashboards must be stopped first.
    """
    try:
        con = duckdb.connect(str(base.DB_PATH))
    except duckdb.IOException as exc:
        raise SystemExit(
            f"Could not open {base.DB_PATH} for writing ({exc}).\n"
            "Stop any running dashboards (app.py / app_models.py) and retry."
        ) from exc
    try:
        schema = con.execute("DESCRIBE claims").fetchall()
        select_cols = ", ".join(
            f'CAST("{name}" AS {ctype}) AS "{name}"' if name in fc.columns
            else f'CAST(NULL AS {ctype}) AS "{name}"'
            for name, ctype, *_ in schema
        )
        con.register("fc_df", fc)
        con.execute(
            f"CREATE OR REPLACE TABLE {tcfg.FORECAST_TABLE} AS "
            f"SELECT {select_cols} FROM fc_df"
        )
        con.execute(
            f"CREATE OR REPLACE VIEW {tcfg.EXTENDED_VIEW} AS "
            f"SELECT *, FALSE AS is_projected FROM claims "
            f"UNION ALL BY NAME "
            f"SELECT *, TRUE AS is_projected FROM {tcfg.FORECAST_TABLE}"
        )

        n_rows, n_cells, n_years = con.execute(
            f"SELECT count(*), count(DISTINCT (postal_code_3, age_group, gender)), "
            f"count(DISTINCT year) FROM {tcfg.FORECAST_TABLE}"
        ).fetchone()
        assert n_rows == n_cells * n_years == len(fc), "Forecast grain not unique"
        yr_min, yr_max = con.execute(
            f"SELECT min(year), max(year) FROM {tcfg.FORECAST_TABLE}").fetchone()
        assert (yr_min, yr_max) == (min(tcfg.FORECAST_YEARS), max(tcfg.FORECAST_YEARS))
        neg = con.execute(
            "SELECT count(*) FROM " + tcfg.FORECAST_TABLE + " WHERE "
            + " OR ".join(f"{c} < 0" for c in base.COST_COLUMNS)
        ).fetchone()[0]
        assert neg == 0, f"{neg} forecast rows carry negative costs"
        n_claims = con.execute("SELECT count(*) FROM claims").fetchone()[0]
        n_view = con.execute(f"SELECT count(*) FROM {tcfg.EXTENDED_VIEW}").fetchone()[0]
        assert n_view == n_claims + n_rows, "View row count mismatch"
        print(f"Wrote {tcfg.FORECAST_TABLE}: {n_rows:,} rows "
              f"({n_cells:,} cells x {n_years} years) + view {tcfg.EXTENDED_VIEW} "
              f"({n_view:,} rows).")
    finally:
        con.close()


def main(model_key: str | None = None, treatment: str | None = None) -> None:
    """Refit the deployed config, project 2024-2030, allocate, write the DB."""
    if model_key is None:
        model_key, treatment = tcfg.DEPLOYED_MODEL, tcfg.DEPLOYED_TREATMENT
    print(f"Projection config: {model_key} / {treatment} (deployed)")
    leader = load_winner()
    if leader is not None and leader != (model_key, treatment):
        print(f"  note: experiment accuracy leader is {leader[0]} / {leader[1]} — "
              "deployment overrides it on trend plausibility (see config.py)")

    estimator, lat_med, lon_med = refit(model_key, treatment)
    fc = build_forecast(estimator, lat_med, lon_med)
    fc = allocate_categories(fc)

    tcfg.ensure_dirs()
    fc.to_parquet(tcfg.FORECAST_PATH, index=False)
    summary = summarize(fc, model_key, treatment)
    tcfg.FORECAST_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {tcfg.FORECAST_PATH} and {tcfg.FORECAST_SUMMARY_PATH}")

    write_db(fc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tcfg.TEMPORAL_MODELS, default=None)
    parser.add_argument("--treatment", choices=list(tcfg.TREATMENTS) + ["none"],
                        default=None)
    args = parser.parse_args()
    if (args.model is None) != (args.treatment is None):
        parser.error("--model and --treatment must be given together")
    main(args.model, args.treatment)
