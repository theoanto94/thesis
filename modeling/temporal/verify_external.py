"""External check of the deployed forecast against a later Vektis release.

OUT OF THESIS SCOPE. Not imported by run_all, and contributes nothing to model
selection, the methodology, or any figure. The panel ends in 2023 and every
decision in this track is justified from 2011-2023 alone — a hold-out year
consulted while choosing a configuration stops being a hold-out. This runs
after the configuration is fixed and changes nothing on its own.

The throwaway database goes through the ordinary preprocessing pipeline, so the
actuals are directly comparable to claims_forecast. A persistence baseline is
scored alongside the model, since a forecast that cannot beat "nothing changes"
is not adding anything.

    uv run python -m modeling.temporal.verify_external
    uv run python -m modeling.temporal.verify_external --year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import preprocessing

from .. import config as base
from ..metrics import weighted_metrics
from . import config as tcfg

# Outside data/, so the thesis build cannot pick these up by accident.
VERIFICATION_DIR = Path(__file__).resolve().parents[2] / "data_for_verification_temporal"

# Throwaway; safe to delete, rebuilt on demand.
VERIFICATION_DB = Path(__file__).resolve().parents[2] / "verification_external.db"


def build_verification_db(rebuild: bool = False) -> Path:
    """Build the extended-panel database, reusing it if already present."""
    if VERIFICATION_DB.exists() and not rebuild:
        print(f"Reusing {VERIFICATION_DB.name} (pass --rebuild to refresh)")
        return VERIFICATION_DB
    globs = [preprocessing.CLAIMS_GLOB, str(VERIFICATION_DIR / "*_postcode3_en.csv")]
    print(f"Building {VERIFICATION_DB.name} from {globs}...")
    preprocessing.build(str(VERIFICATION_DB), claims_glob=globs)
    return VERIFICATION_DB


def _rate_expr(prefix: str = "") -> str:
    """SQL summing the 23 category costs — the dashboard's total."""
    return " + ".join(f"COALESCE({prefix}{c}, 0)" for c in base.COST_COLUMNS)


def load_comparison(year: int) -> pd.DataFrame:
    """Actual vs forecast vs persistence rates for every matched cell."""
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{VERIFICATION_DB}' AS ext (READ_ONLY)")
    con.execute(f"ATTACH '{base.DB_PATH}' AS thesis (READ_ONLY)")
    keys = "gender, postal_code_3, age_group"
    expr = _rate_expr()
    df = con.execute(f"""
        WITH actual AS (
            SELECT {keys}, insured_years AS iy_actual,
                   ({expr}) / NULLIF(insured_years, 0) AS rate_actual
            FROM ext.claims WHERE year = {year} AND insured_years > 0
        ), forecast AS (
            SELECT {keys}, insured_years AS iy_forecast,
                   ({expr}) / NULLIF(insured_years, 0) AS rate_forecast
            FROM thesis.claims_forecast WHERE year = {year}
        ), persistence AS (
            SELECT {keys}, ({expr}) / NULLIF(insured_years, 0) AS rate_persistence
            FROM thesis.claims
            WHERE year = {tcfg.FORECAST_BASE_YEAR} AND insured_years > 0
        )
        SELECT a.*, f.iy_forecast, f.rate_forecast,
               p.rate_persistence
        FROM actual a
        JOIN forecast f USING (gender, postal_code_3, age_group)
        LEFT JOIN persistence p USING (gender, postal_code_3, age_group)
    """).df()
    con.close()
    return df


def _score(actual: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    """Canonical weighted metrics plus the level diagnostics this check needs.

    RMSE/MAE/R2 use the same definition as every table in the thesis, which is
    the whole point; bias and the two levels are added on top.
    """
    m = weighted_metrics(actual, pred, w)
    return {
        "rmse": m["rmse"],
        "mae": m["mae"],
        "r2": m["r2"],
        "bias": float(np.average(pred - actual, weights=w)),
        "actual": float(np.average(actual, weights=w)),
        "predicted": float(np.average(pred, weights=w)),
    }


def _score_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group scores with the percentage level error appended."""
    out = df.groupby(by).apply(
        lambda g: pd.Series(_score(g["rate_actual"].to_numpy(),
                                   g["rate_forecast"].to_numpy(),
                                   g["iy_actual"].to_numpy())),
        include_groups=False,
    )
    out["bias_%"] = (out["predicted"] / out["actual"] - 1) * 100
    return out


def report(year: int) -> None:
    """Print the comparison. Nothing downstream reads any of it."""
    df = load_comparison(year)
    if df.empty:
        raise SystemExit(f"No matched cells for {year} — is the forecast written?")
    w = df["iy_actual"].to_numpy()
    actual = df["rate_actual"].to_numpy()

    print(f"\n=== External check: {year} (OUT OF THESIS SCOPE) ===")
    print(f"matched cells: {len(df):,}")
    print(f"insured-years: actual {df['iy_actual'].sum():,.0f} | "
          f"projected {df['iy_forecast'].sum():,.0f} "
          f"({df['iy_forecast'].sum() / df['iy_actual'].sum() - 1:+.3%})")

    rows = {"deployed model": _score(actual, df["rate_forecast"].to_numpy(), w)}
    have_persistence = df["rate_persistence"].notna().all()
    if have_persistence:
        rows[f"persistence ({tcfg.FORECAST_BASE_YEAR} carried forward)"] = _score(
            actual, df["rate_persistence"].to_numpy(), w)
    table = pd.DataFrame(rows).T[["rmse", "mae", "r2", "bias", "actual", "predicted"]]
    print("\ninsured-years-weighted, EUR per insured year:")
    print(table.round(2).to_string())

    if have_persistence:
        m, p = table.loc["deployed model"], table.iloc[1]
        verdict = "beats" if m["rmse"] < p["rmse"] else "does NOT beat"
        print(f"\nThe model {verdict} persistence on RMSE "
              f"({m['rmse']:,.1f} vs {p['rmse']:,.1f}); "
              f"bias {m['bias']:+,.1f} vs {p['bias']:+,.1f}.")

    by_age = _score_by(df, "age_group")
    print("\nby age group:")
    print(by_age[["actual", "predicted", "rmse", "bias_%"]].round(1).to_string())

    by_gem = df.merge(
        duckdb.connect(str(base.DB_PATH), read_only=True)
        .execute("SELECT DISTINCT postal_code_3, gementee FROM claims").df(),
        on="postal_code_3", how="left",
    ).dropna(subset=["gementee"])
    gem = _score_by(by_gem, "gementee")
    worst = gem.reindex(gem["bias_%"].abs().sort_values(ascending=False).index)
    print("\n10 gemeenten with the largest level error:")
    print(worst.head(10)[["actual", "predicted", "bias_%"]].round(1).to_string())

    print("\nReminder: this year is not part of the thesis panel and did not "
          "inform any modelling decision.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the throwaway verification database")
    args = parser.parse_args()
    build_verification_db(rebuild=args.rebuild)
    report(args.year)


if __name__ == "__main__":
    main()
