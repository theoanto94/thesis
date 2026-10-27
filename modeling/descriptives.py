"""Per-year descriptive statistics of the analysis panel (thesis Table 3.4).

One row per year: number of cells, exposure (million insured-years), total
real cost (EUR billion) and the population-weighted national real cost rate
(total real cost / total insured-years, never / number_of_insured).

Read-only on ``identifier.db``; writes ``artifacts/descriptives/by_year.csv``.

    uv run python -m modeling.descriptives
"""
from __future__ import annotations

import duckdb
import pandas as pd

from . import config

DESCRIPTIVES_DIR = config.ARTIFACTS / "descriptives"


def by_year() -> pd.DataFrame:
    """Cells, exposure, total real cost and national real cost rate per year."""
    total = " + ".join(f"COALESCE({c}, 0)" for c in config.COST_COLUMNS)
    sql = f"""
        SELECT year,
               COUNT(*)                          AS cells,
               SUM({config.WEIGHT_COLUMN}) / 1e6 AS insured_years_m,
               SUM({total}) / 1e9                AS total_real_cost_bn,
               SUM({total}) / SUM({config.WEIGHT_COLUMN}) AS real_cost_rate
        FROM claims
        GROUP BY year
        ORDER BY year
    """
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def main() -> None:
    """Write and print the per-year table."""
    table = by_year()
    DESCRIPTIVES_DIR.mkdir(parents=True, exist_ok=True)
    dest = DESCRIPTIVES_DIR / "by_year.csv"
    table.round({"insured_years_m": 2, "total_real_cost_bn": 2, "real_cost_rate": 0}).to_csv(
        dest, index=False)
    print(table.round(2).to_string(index=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
