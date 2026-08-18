"""Lag-construction correctness on a synthetic panel with a year gap.

Exercises the production lag builder (``modeling.temporal.data.add_lag_features``)
directly: a lag k must be the entity's value exactly k years earlier, and NULL
when that year is unobserved — never the previous *row* across a gap (the
classic ``groupby().shift()`` bug) and never another entity's value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.temporal.data import add_lag_features


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two entities, 2018-2022; entity B is missing 2020 (a year gap)."""
    rows = []
    for year in [2018, 2019, 2020, 2021, 2022]:
        rows.append(("100", "0-4", "M", year, float(year)))          # entity A
    for year in [2018, 2019, 2021, 2022]:                            # gap: 2020
        rows.append(("200", "0-4", "V", year, float(year) * 10))     # entity B
    return pd.DataFrame(
        rows, columns=["postal_code_3", "age_group", "gender", "year", "rate_x"]
    )


def _lagged(panel: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(panel, {"x": "rate_x"}, max_depth=3)


def _cell(df, pc3, year, col):
    return df.loc[(df["postal_code_3"] == pc3) & (df["year"] == year), col].iloc[0]


def test_contiguous_entity_gets_exact_lags(panel):
    df = _lagged(panel)
    assert _cell(df, "100", 2022, "x_lag1") == 2021.0
    assert _cell(df, "100", 2022, "x_lag2") == 2020.0
    assert _cell(df, "100", 2022, "x_lag3") == 2019.0


def test_leading_years_have_null_lags(panel):
    df = _lagged(panel)
    assert np.isnan(_cell(df, "100", 2018, "x_lag1"))
    assert np.isnan(_cell(df, "100", 2019, "x_lag2"))
    assert np.isnan(_cell(df, "100", 2020, "x_lag3"))


def test_year_gap_yields_null_not_previous_row(panel):
    """B(2021): 2020 is missing -> lag1 NULL; 2019/2018 exist -> lag2/lag3 valid."""
    df = _lagged(panel)
    assert np.isnan(_cell(df, "200", 2021, "x_lag1"))
    assert _cell(df, "200", 2021, "x_lag2") == 20190.0
    assert _cell(df, "200", 2021, "x_lag3") == 20180.0


def test_lags_skip_the_gap_but_recover_beyond_it(panel):
    """B(2022): lag1 -> 2021 valid, lag2 -> 2020 NULL, lag3 -> 2019 valid."""
    df = _lagged(panel)
    assert _cell(df, "200", 2022, "x_lag1") == 20210.0
    assert np.isnan(_cell(df, "200", 2022, "x_lag2"))
    assert _cell(df, "200", 2022, "x_lag3") == 20190.0


def test_lags_do_not_cross_entities(panel):
    """B's first year must not inherit lag values from entity A's rows."""
    df = _lagged(panel)
    for k in (1, 2, 3):
        assert np.isnan(_cell(df, "200", 2018, f"x_lag{k}"))


def test_row_count_and_grain_preserved(panel):
    df = _lagged(panel)
    assert len(df) == len(panel)
    assert not df.duplicated(["postal_code_3", "age_group", "gender", "year"]).any()
