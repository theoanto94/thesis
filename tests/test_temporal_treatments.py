"""Treatment split and persistence-baseline correctness.

Exercises the pure config logic (``train_target_years``) and the persistence
model's identity property: run through the recursive loop, its prediction at
every step equals the seed value, because its own output feeds its own lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.temporal import config as tcfg
from modeling.temporal.experiments import _Persistence


class TestTrainTargetYears:
    def test_base_keeps_every_transition(self):
        assert tcfg.train_target_years("base") == tuple(range(2012, 2021))

    def test_no_covid_drops_2020_and_2021(self):
        years = tcfg.train_target_years("no_covid")
        assert 2020 not in years and 2021 not in years
        assert years == tuple(y for y in range(2012, 2021) if y not in (2020, 2021))

    def test_refit_through_2023_respects_treatment(self):
        base_years = tcfg.train_target_years("base", train_end=2023)
        no_covid_years = tcfg.train_target_years("no_covid", train_end=2023)
        assert base_years == tuple(range(2012, 2024))
        assert set(base_years) - set(no_covid_years) == {2020, 2021}
        # Post-COVID transitions survive the exclusion.
        assert {2022, 2023} <= set(no_covid_years)

    def test_unknown_treatment_rejected(self):
        with pytest.raises(ValueError):
            tcfg.train_target_years("everything")


class TestPersistence:
    def test_prediction_is_the_incoming_lag(self):
        X = pd.DataFrame({tcfg.LAG_COL: [1.0, 2.5, 0.0], "year": [2024] * 3})
        np.testing.assert_array_equal(
            _Persistence().predict(X), np.array([1.0, 2.5, 0.0])
        )

    def test_recursive_chain_carries_seed_forward(self):
        """Feeding predictions back as lags leaves the seed unchanged at every step."""
        seed = np.array([100.0, 250.0, 0.0])
        lag = seed
        for _ in range(7):  # the 2024-2030 horizon length
            pred = _Persistence().predict(pd.DataFrame({tcfg.LAG_COL: lag}))
            np.testing.assert_array_equal(pred, seed)
            lag = pred
