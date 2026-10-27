"""The age x gender encoder of Ridge-AG and the metric-file merging.

Synthetic frames only — no database access.
"""

from __future__ import annotations

import pandas as pd

from modeling import evaluate
from modeling.encoders import AgeGenderCross


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "age_group": ["00-04", "00-04", "00-04", "90+", "90+"],
        "gender": ["M", "M", "V", "M", "M"],
        "postal_code_3": ["101", "102", "101", "101", "101"],
        "year": [2011, 2012, 2011, 2019, 2020],
    })


class TestAgeGenderCross:
    def test_onehot_drop_first(self):
        X = _frame()
        out = AgeGenderCross().fit(X).transform(X)
        assert out.shape == (5, 2)                     # 3 cells, first dropped
        assert (out.sum(axis=1) <= 1).all()


class TestUpsert:
    def test_keeps_other_models(self, tmp_path):
        path = tmp_path / "overall.parquet"
        pd.DataFrame({"model": ["ridge", "stacked"], "rmse": [1.0, 2.0]}).to_parquet(path)
        evaluate._upsert(path, pd.DataFrame({"model": ["ridge", "ridge_ag"], "rmse": [9.0, 3.0]}))
        out = pd.read_parquet(path).set_index("model")["rmse"]
        assert out.to_dict() == {"ridge": 9.0, "ridge_ag": 3.0, "stacked": 2.0}
