"""Shared bootstrap core and the CBS exposure factors.

The population tests read the cached CBS extract in data/external/ and are
skipped when it is absent (it is created by modeling.temporal.population).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.bootstrap_ci import bootstrap_metrics
from modeling.metrics import weighted_metrics
from modeling.temporal import population


class TestBootstrap:
    def _data(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 1000.0, 500)
        w = rng.uniform(1, 50, 500)
        return y, w, {"a": y + rng.normal(0, 100, 500), "b": y + rng.normal(0, 300, 500)}

    def test_deterministic_and_point_matches(self):
        y, w, p = self._data()
        r1 = bootstrap_metrics(y, w, p, baseline="a", n_boot=50)
        r2 = bootstrap_metrics(y, w, p, baseline="a", n_boot=50)
        assert r1 == r2
        assert r1["b"]["rmse"]["point"] == pytest.approx(weighted_metrics(y, p["b"], w)["rmse"])
        d = r1["b"]["delta_rmse_vs_baseline"]
        assert d["ci_lo"] > 0                          # b is clearly worse than a

    def test_no_baseline_no_delta(self):
        y, w, p = self._data()
        r = bootstrap_metrics(y, w, p, n_boot=20)
        assert all("delta_rmse_vs_baseline" not in e for e in r.values())


@pytest.mark.skipif(not population.POPULATION_CSV.exists(), reason="CBS extract not downloaded")
class TestPopulation:
    def test_factors_reproduce_cbs_totals(self):
        g = population.growth_factors()
        base = g[g["year"] == 2023].set_index(["age_group", "gender"])["population"]
        for year, sub in g.groupby("year"):
            implied = (base.reindex(pd.MultiIndex.from_frame(sub[["age_group", "gender"]])).to_numpy()
                       * sub["factor_age_sex"].to_numpy()).sum()
            assert implied == pytest.approx(sub["population"].sum())

    def test_exposure_factor_scenarios(self):
        rows = pd.DataFrame({"age_group": ["80-84", "00-04"], "gender": ["V", "M"]})
        ageing = population.exposure_factor(rows, 2030, "cbs_age_sex")
        assert ageing[0] > ageing[1]                   # the 80-84 band outgrows toddlers
        assert population.exposure_factor(rows, 2025, "uniform:0.01")[0] == pytest.approx(1.01 ** 2)
