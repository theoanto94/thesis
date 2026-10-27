"""Temporal track orchestrator: experiments -> forecast.

Usage::

    uv run python -m modeling.temporal.run_all

Stage 2 opens ``identifier.db`` for writing, so stop the dashboards first.
"""

from __future__ import annotations

from . import experiments, forecast


def main() -> None:
    """Run the temporal experiments, then the 2024-2030 forecast."""
    print("=" * 70)
    print("Stage 1/2 — COVID-treatment experiments (7 configs, recursive eval)")
    print("=" * 70)
    experiments.main()

    print("\n" + "=" * 70)
    print("Stage 2/2 — refit winner, project 2024-2030, write DB")
    print("=" * 70)
    forecast.main()


if __name__ == "__main__":
    main()
