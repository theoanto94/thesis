"""Age x sex exposure growth for the 2024-2030 projection, from CBS.

The projection holds each cell's cost-rate model fixed and needs a path for
its exposure (insured_years, number_of_insured). A single national growth
rate would freeze the 2023 age mix, yet 2024-2030 is when the large post-war
cohorts move into the 75+ bands. This module builds one growth factor per
(Vektis age band, gender, year) from the CBS core population forecast
(kernprognose 2024-2070), so each band grows or shrinks as CBS projects.

Sources (CBS StatLine OData, cached as one tidy CSV in ``data/external/``):

* 7461BEV  — observed population on 1 January by sex and 5-year age band;
  used for 1 Jan 2023 and 1 Jan 2024, the base of the forecast vintage.
* 86041NED — kernprognose 2024-2070, population on 1 January 2025-2070 by sex
  and age. Only this forecast vintage is used from 2025 on (no later
  observations), so the scenario is the one CBS published in December 2024.

The exposure of cost year h approximates the mid-year population,
``(P[1 Jan h] + P[1 Jan h+1]) / 2``, and the factor is relative to 2023:
``g(a, s, h) = Pmid(a, s, h) / Pmid(a, s, 2023)``.

    uv run python -m modeling.temporal.population   # download (if needed) + print
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config as base
from . import config as tcfg

EXTERNAL_DIR = base.ROOT / "data" / "external"
POPULATION_CSV = EXTERNAL_DIR / "cbs_population_age_sex.csv"

ODATA = "https://opendata.cbs.nl/ODataApi/odata/{table}/TypedDataSet"
OBSERVED_TABLE, FORECAST_TABLE = "7461BEV", "86041NED"
OBSERVED_YEARS = (2023, 2024)       # 1 January, observed
FORECAST_JAN1_YEARS = range(2025, max(tcfg.FORECAST_YEARS) + 2)  # up to 1 Jan 2031

_SEX = {"3000": "M", "4000": "V"}   # CBS Mannen / Vrouwen -> Vektis M / V

# CBS 5-year age codes -> Vektis bands; 90-94 and 95+ pool into 90+.
_AGE = {
    "70100": "00-04", "70200": "05-09", "70300": "10-14", "70400": "15-19",
    "70500": "20-24", "70600": "25-29", "70700": "30-34", "70800": "35-39",
    "70900": "40-44", "71000": "45-49", "71100": "50-54", "71200": "55-59",
    "71300": "60-64", "71400": "65-69", "71500": "70-74", "71600": "75-79",
    "71700": "80-84", "71800": "85-89", "71900": "90+", "22000": "90+",
}


def _fetch(table: str, years, value_col: str, extra_filter: str = "") -> pd.DataFrame:
    """Download population on 1 January for *years* from one CBS table."""
    # Dimension keys come back space-padded (e.g. '3000   '), so sex and age
    # are filtered after download rather than in the OData query.
    periods = " or ".join(f"Perioden eq '{y}JJ00'" for y in years)
    flt = f"({periods}){extra_filter}"
    url = ODATA.format(table=table) + "?" + urllib.parse.urlencode(
        {"$format": "json", "$filter": flt})
    with urllib.request.urlopen(url, timeout=120) as resp:
        rows = json.load(resp)["value"]
    df = pd.DataFrame(rows)
    df["age_code"] = df["Leeftijd"].str.strip()
    df["sex_code"] = df["Geslacht"].str.strip()
    df = df[df["age_code"].isin(_AGE) & df["sex_code"].isin(_SEX)]
    return pd.DataFrame({
        "source": table,
        "jan1_year": df["Perioden"].str[:4].astype(int),
        "gender": df["sex_code"].map(_SEX),
        "age_group": df["age_code"].map(_AGE),
        "population": df[value_col].astype(float),
    })


def download(force: bool = False) -> pd.DataFrame:
    """Fetch both CBS tables and cache the tidy result (idempotent)."""
    if POPULATION_CSV.exists() and not force:
        return pd.read_csv(POPULATION_CSV)
    obs = _fetch(OBSERVED_TABLE, OBSERVED_YEARS, "Bevolking_1",
                 " and BurgerlijkeStaat eq 'T001019'")
    fc = _fetch(FORECAST_TABLE, FORECAST_JAN1_YEARS, "BevolkingOp1Januari_1")
    tidy = (pd.concat([obs, fc], ignore_index=True)
            .groupby(["source", "jan1_year", "gender", "age_group"], as_index=False)["population"].sum())
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(POPULATION_CSV, index=False)
    print(f"wrote {POPULATION_CSV} ({len(tidy)} rows)")
    return tidy


@lru_cache(maxsize=1)
def midyear_population() -> pd.DataFrame:
    """Approximate mid-year population per (year, age_group, gender), 2023-2030."""
    jan1 = download().set_index(["jan1_year", "age_group", "gender"])["population"]
    rows = []
    for h in range(tcfg.FORECAST_BASE_YEAR, max(tcfg.FORECAST_YEARS) + 1):
        mid = (jan1.loc[h] + jan1.loc[h + 1]) / 2.0
        rows.append(mid.rename("population").reset_index().assign(year=h))
    out = pd.concat(rows, ignore_index=True)
    n_cells = out.groupby("year").size()
    assert (n_cells == 38).all(), f"expected 19 bands x 2 sexes per year, got {n_cells.to_dict()}"
    return out


def growth_factors() -> pd.DataFrame:
    """g(age_group, gender, year) = Pmid(h) / Pmid(2023) as ``factor_age_sex``.

    Each band grows by its own CBS factor, so the projected age mix ages.
    """
    mid = midyear_population()
    base_pop = mid[mid["year"] == tcfg.FORECAST_BASE_YEAR].set_index(["age_group", "gender"])["population"]
    out = mid.copy()
    out["factor_age_sex"] = out["population"].to_numpy() / base_pop.reindex(
        pd.MultiIndex.from_frame(out[["age_group", "gender"]])).to_numpy()
    return out


def exposure_factor(rows: pd.DataFrame, year: int, scenario: str) -> np.ndarray:
    """Per-row multiplier on 2023 exposure for forecast *year* under *scenario*.

    ``cbs_age_sex`` — each band's own CBS growth (ageing);
    ``uniform:<r>`` — a fixed annual rate r for every band (legacy check).
    """
    if scenario.startswith("uniform:"):
        rate = float(scenario.split(":", 1)[1])
        return np.full(len(rows), (1.0 + rate) ** (year - tcfg.FORECAST_BASE_YEAR))
    col = {"cbs_age_sex": "factor_age_sex"}[scenario]
    g = growth_factors()
    g = g[g["year"] == year].set_index(["age_group", "gender"])[col]
    out = g.reindex(pd.MultiIndex.from_frame(rows[["age_group", "gender"]].astype(str))).to_numpy()
    assert np.isfinite(out).all(), f"missing CBS factor for some cells in {year}"
    return out


def main() -> None:
    """Download if needed and print the factors and the 75+ share path."""
    g = growth_factors()
    pd.set_option("display.width", 200)
    piv = g.pivot_table(index="age_group", columns="year", values="factor_age_sex", aggfunc="mean")
    print("Mean (M/V) CBS growth factor vs 2023 by band:\n", piv.round(3).to_string())
    tot = g.groupby("year")["population"].sum()
    old = g[g["age_group"].isin(["75-79", "80-84", "85-89", "90+"])].groupby("year")["population"].sum()
    cagr = (tot.iloc[-1] / tot.iloc[0]) ** (1 / (len(tot) - 1)) - 1
    print("\nTotal mid-year population (M):", (tot / 1e6).round(3).to_dict())
    print(f"CAGR 2023-2030: {cagr:+.3%}; 75+ share:", (100 * old / tot).round(2).to_dict())


if __name__ == "__main__":
    main()
