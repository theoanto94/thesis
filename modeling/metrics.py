"""Population-weighted regression metrics and disaggregation.

Every metric is weighted by ``insured_years`` (the population exposure), per the
project rule that this column is the canonical weight. Metrics follow the set
used by Orji & Ukwandu (2024): R2, RMSE, MAE, MAPE.

MAPE is undefined as the actual approaches zero — common for niche cost
categories (e.g. maternity care for elderly men). We therefore score MAPE only
on rows whose actual is at least ``config.MAPE_FLOOR_EUR`` and report the
fraction of (weighted) exposure that floor covers, so the number is honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

#: Canonical metric column order used in tables and the dashboard.
METRIC_NAMES: list[str] = ["r2", "rmse", "mae", "mape", "mape_coverage"]


def weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    """Return weighted R2, RMSE, MAE, MAPE (%) and MAPE coverage (fraction).

    Parameters
    ----------
    y_true, y_pred, weight:
        1-D arrays of equal length. ``weight`` is ``insured_years``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(weight, dtype=float)

    if w.sum() <= 0 or len(y_true) == 0:
        return {k: float("nan") for k in METRIC_NAMES}

    resid = y_true - y_pred
    w_sum = w.sum()

    # R2 against the weighted-mean baseline.
    w_mean = np.average(y_true, weights=w)
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * (y_true - w_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rmse = float(np.sqrt(ss_res / w_sum))
    mae = float(np.sum(w * np.abs(resid)) / w_sum)

    # MAPE only where the actual is meaningfully non-zero.
    mask = np.abs(y_true) >= config.MAPE_FLOOR_EUR
    if mask.any():
        wm = w[mask]
        mape = float(
            np.sum(wm * np.abs(resid[mask] / y_true[mask])) / wm.sum() * 100.0
        )
        coverage = float(wm.sum() / w_sum)
    else:
        mape = float("nan")
        coverage = 0.0

    return {
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mape_coverage": coverage,
    }


def disaggregate(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
    by: str,
) -> pd.DataFrame:
    """Compute weighted metrics within each level of a grouping column.

    Parameters
    ----------
    meta:
        Identifier frame aligned row-for-row with the predictions; must contain
        the column named by *by* (e.g. ``"age_group"``, ``"gementee"``,
        ``"postal_code_3"``).
    by:
        The grouping column.

    Returns
    -------
    DataFrame indexed by group value with one column per metric plus ``n`` (row
    count) and ``exposure`` (summed weight).
    """
    frame = pd.DataFrame({
        "_g": meta[by].to_numpy(),
        "_yt": np.asarray(y_true, dtype=float),
        "_yp": np.asarray(y_pred, dtype=float),
        "_w": np.asarray(weight, dtype=float),
    })
    rows: list[dict] = []
    for g, sub in frame.groupby("_g", sort=True):
        m = weighted_metrics(sub["_yt"].to_numpy(), sub["_yp"].to_numpy(), sub["_w"].to_numpy())
        m[by] = g
        m["n"] = int(len(sub))
        m["exposure"] = float(sub["_w"].sum())
        rows.append(m)
    cols = [by, *METRIC_NAMES, "n", "exposure"]
    return pd.DataFrame(rows)[cols]
