"""Temporal (lag-based) forecasting track — separate from the Phase 2 pipeline.

Where Phase 2 is deliberately structural, this track predicts the total real
cost rate one year ahead from the previous year's realized rate plus the
structural features, and asks whether dropping the COVID years from training
improves forecasts. Four models are compared under both treatments on a common
recursive 2022-2023 protocol; the winner is refit through 2023 and rolled
forward to project 2024-2030 for the EDA dashboard.
"""
