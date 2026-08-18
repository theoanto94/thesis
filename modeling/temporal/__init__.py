"""Temporal (lag-based) forecasting track — separate from the Phase 2 pipeline.

Phase 2 models are deliberately structural (no cost-derived features). This
subpackage predicts the *total* real cost rate one year ahead from the
previous year's realized rate (a single t-1 lag) plus the structural
features, and asks one experimental question: does removing the COVID years
(2020, 2021) from training improve forecasts? Four models (persistence,
Ridge, LightGBM, XGBoost) are compared under both treatments on a common
recursive 2022-2023 protocol; the winner is refit through 2023 and rolled
forward to project 2024-2030 at the full postcode3 x age_group x gender
grain, feeding the EDA dashboard's projected years.
"""
