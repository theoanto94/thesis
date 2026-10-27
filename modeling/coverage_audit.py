"""Coverage audit of the cost categories, and fixed-basket cost growth.

The Zvw benefit basket changes over the panel: categories enter (geriatric
rehabilitation 2013, nursing and GP multidisciplinary care 2015, primary-care
residence 2018, specific patient groups 2020, integrated birth care 2021),
leave (primary-care support after 2018) or are restructured (mental health).
A category that has not entered yet is NULL and counts as zero in the total, so
every entry is a step in the total cost rate that is coverage, not growth.

This script documents those steps and separates them from like-for-like growth:

* per category: first/last year with non-zero national cost, national real
  cost per year, and year-on-year jumps above ``JUMP_THRESHOLD``;
* mental-health sub-columns straight from the raw CSVs (nominal euros), since
  the ``claims`` table only keeps the roll-up plus the post-2021 components;
* the national cost rate on all 23 categories vs a *fixed basket* of the
  categories present in every year, both raw and standardised to the 2023
  age x gender mix of insured-years.

Read-only on ``identifier.db``; writes ``artifacts/coverage/``.

    uv run python -m modeling.coverage_audit
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from . import config

COVERAGE_DIR = config.ARTIFACTS / "coverage"

# Year-on-year change in national real cost above which a year is flagged.
JUMP_THRESHOLD = 0.50

FIRST_YEAR, LAST_YEAR = config.TRAIN_YEARS.start, config.TEST_YEARS.stop - 1
STANDARD_YEAR = LAST_YEAR  # age x gender mix used for standardisation

# Raw mental-health columns (translated names), summed into
# costs_mental_health_total by preprocessing.py.
RAW_MENTAL_HEALTH_COLUMNS: list[str] = [
    "costs_secondary_mental_health",
    "costs_primary_psychological_care",
    "costs_specialist_mental_health",
    "costs_general_basic_mental_health",
    "costs_longterm_mental_health",
    "costs_mental_health_innovation",
    "costs_mental_health_consultations",
    "costs_inpatient_mental_health",
    "costs_other_mental_health_services",
]


def _cagr(first: float, last: float, years: int) -> float:
    """Compound annual growth rate between two levels *years* apart."""
    return (last / first) ** (1.0 / years) - 1.0


def load_cells() -> pd.DataFrame:
    """Cell-level real costs per category plus the keys and insured_years."""
    cols = ", ".join(["year", "age_group", "gender", config.WEIGHT_COLUMN, *config.COST_COLUMNS])
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        return con.execute(f"SELECT {cols} FROM claims").fetchdf()
    finally:
        con.close()


def category_coverage(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """National real cost per category and year, and a per-category summary.

    Returns ``(national, summary)``: *national* is year x category in euros
    (NULL treated as zero); *summary* has first/last non-zero year and the
    flagged jumps for each category.
    """
    national = cells.groupby("year")[config.COST_COLUMNS].sum(min_count=1).fillna(0.0)
    label_of = {col: label for label, col in config.COST_CATEGORIES.items()}
    rows = []
    for col in config.COST_COLUMNS:
        s = national[col]
        nz = s[s.abs() > 0]
        jumps = []
        for year in s.index[1:]:
            prev, cur = s.loc[year - 1], s.loc[year]
            if prev == 0 and cur != 0:
                jumps.append(f"{year}: enters")
            elif prev != 0 and cur == 0:
                jumps.append(f"{year}: drops to zero")
            elif prev != 0 and abs(cur / prev - 1) > JUMP_THRESHOLD:
                jumps.append(f"{year}: {cur / prev - 1:+.0%}")
        rows.append({
            "category": label_of[col],
            "column": col,
            "first_nonzero_year": int(nz.index.min()) if len(nz) else None,
            "last_nonzero_year": int(nz.index.max()) if len(nz) else None,
            "continuous": bool(len(nz) == len(s)),
            "cost_first_year_meur": float(nz.iloc[0] / 1e6) if len(nz) else 0.0,
            "cost_2023_meur": float(s.loc[LAST_YEAR] / 1e6),
            "flags": "; ".join(jumps),
        })
    return national, pd.DataFrame(rows)


def raw_mental_health() -> pd.DataFrame:
    """National nominal cost per raw mental-health column and year, from the CSVs."""
    glob = str(config.ROOT / "data" / "*_postcode3_en.csv")
    con = duckdb.connect()
    try:
        present = set(con.execute(
            f"SELECT * FROM read_csv('{glob}', delim=';', header=true, "
            f"union_by_name=true, null_padding=true, all_varchar=true) LIMIT 0"
        ).fetchdf().columns)
        cols = [c for c in RAW_MENTAL_HEALTH_COLUMNS if c in present]
        sums = ", ".join(
            f"SUM(TRY_CAST(NULLIF(NULLIF({c}, '(null)'), '') AS DOUBLE)) AS {c}" for c in cols
        )
        df = con.execute(
            f"SELECT CAST(substr(regexp_extract(regexp_extract(filename, '[^/]+$'), '\\d+'), 1, 4) AS INTEGER) AS year, "
            f"{sums} FROM read_csv('{glob}', delim=';', header=true, union_by_name=true, "
            f"null_padding=true, all_varchar=true, filename=true) "
            # Vektis files carry a national total row with empty keys; drop it.
            f"WHERE postal_code_3 IS NOT NULL AND postal_code_3 NOT IN ('', '(null)') "
            f"AND age_class IS NOT NULL AND age_class NOT IN ('', '(null)') "
            f"GROUP BY 1 ORDER BY 1"
        ).fetchdf().set_index("year")
    finally:
        con.close()
    return df.fillna(0.0)


def _national_rate(cells: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Crude national rate Σcost / Σinsured_years per year."""
    g = cells.assign(_c=cells[cols].fillna(0.0).sum(axis=1)).groupby("year")
    return g["_c"].sum() / g[config.WEIGHT_COLUMN].sum()


def _standardised_rate(cells: pd.DataFrame, cols: list[str]) -> pd.Series:
    """National rate with each year's age x gender rates on the STANDARD_YEAR mix."""
    c = cells.assign(_c=cells[cols].fillna(0.0).sum(axis=1))
    grp = c.groupby(["year", "age_group", "gender"])[["_c", config.WEIGHT_COLUMN]].sum()
    rate = grp["_c"] / grp[config.WEIGHT_COLUMN]
    mix = grp.loc[STANDARD_YEAR, config.WEIGHT_COLUMN]
    mix = mix / mix.sum()
    out = {}
    for year in sorted(c["year"].unique()):
        r = rate.loc[year].reindex(mix.index)
        out[int(year)] = float((r * mix).sum())
    return pd.Series(out)


def growth_decomposition(cells: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Headline vs fixed-basket national rate growth, crude and standardised."""
    baskets = {
        "all_23": config.COST_COLUMNS,
        "fixed_basket": summary.loc[summary["continuous"], "column"].tolist(),
    }
    baskets["fixed_basket_ex_mental_health"] = [
        c for c in baskets["fixed_basket"] if c != "costs_mental_health_total_real"
    ]
    n_years = LAST_YEAR - FIRST_YEAR
    out: dict = {"standard_year": STANDARD_YEAR, "baskets": {}}
    for name, cols in baskets.items():
        crude = _national_rate(cells, cols)
        std = _standardised_rate(cells, cols)
        out["baskets"][name] = {
            "n_categories": len(cols),
            "categories": cols,
            "crude_rate": {int(k): float(v) for k, v in crude.items()},
            "standardised_rate": {int(k): float(v) for k, v in std.items()},
            "crude_growth_total": float(crude[LAST_YEAR] / crude[FIRST_YEAR] - 1),
            "crude_cagr": _cagr(crude[FIRST_YEAR], crude[LAST_YEAR], n_years),
            "standardised_growth_total": float(std[LAST_YEAR] / std[FIRST_YEAR] - 1),
            "standardised_cagr": _cagr(std[FIRST_YEAR], std[LAST_YEAR], n_years),
        }
    # Pure composition effect: 2023 rates (all 23) applied to each year's mix.
    c = cells.assign(_c=cells[config.COST_COLUMNS].fillna(0.0).sum(axis=1))
    grp = c.groupby(["year", "age_group", "gender"])[["_c", config.WEIGHT_COLUMN]].sum()
    rate_std = (grp["_c"] / grp[config.WEIGHT_COLUMN]).loc[STANDARD_YEAR]
    comp = {}
    for year in sorted(c["year"].unique()):
        mix = grp.loc[year, config.WEIGHT_COLUMN]
        comp[int(year)] = float((rate_std.reindex(mix.index) * mix / mix.sum()).sum())
    out["composition_only_rate"] = comp
    out["composition_only_cagr"] = _cagr(comp[FIRST_YEAR], comp[LAST_YEAR], n_years)
    iy = cells.groupby("year")[config.WEIGHT_COLUMN].sum()
    out["insured_years"] = {int(k): float(v) for k, v in iy.items()}
    out["insured_years_cagr"] = _cagr(iy[FIRST_YEAR], iy[LAST_YEAR], n_years)
    return out


def main() -> None:
    """Run the audit and write CSV/JSON outputs to ``artifacts/coverage/``."""
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    national, summary = category_coverage(cells)
    mh = raw_mental_health()
    growth = growth_decomposition(cells, summary)

    (national / 1e6).round(3).to_csv(COVERAGE_DIR / "national_cost_by_category_meur.csv")
    summary.to_csv(COVERAGE_DIR / "category_coverage.csv", index=False)
    (mh / 1e6).round(3).to_csv(COVERAGE_DIR / "raw_mental_health_nominal_meur.csv")
    (COVERAGE_DIR / "growth.json").write_text(json.dumps(growth, indent=2))

    pd.set_option("display.width", 200)
    print(summary.drop(columns="column").to_string(index=False))
    print("\nRaw mental-health columns, nominal €M:")
    print((mh / 1e6).round(1).to_string())
    print("\nNational rate growth 2011-2023:")
    for name, b in growth["baskets"].items():
        print(f"  {name:32s} n={b['n_categories']:2d}  crude {b['crude_growth_total']:+.1%} "
              f"(CAGR {b['crude_cagr']:+.2%})  std {b['standardised_growth_total']:+.1%} "
              f"(CAGR {b['standardised_cagr']:+.2%})")
    print(f"  composition-only CAGR {growth['composition_only_cagr']:+.2%}; "
          f"insured-years CAGR {growth['insured_years_cagr']:+.3%}")
    print(f"wrote {COVERAGE_DIR}")


if __name__ == "__main__":
    main()
