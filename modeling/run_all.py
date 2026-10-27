"""One-shot orchestration for the full Phase 2 pipeline: tune, train, stack,
evaluate.

    uv run python -m modeling.run_all
"""

from __future__ import annotations

import time

from . import evaluate, tier4_stacking, train, tune


def main() -> None:
    """Run tuning, training, stacking and evaluation in order."""
    t0 = time.time()
    print("=" * 70, "\n[1/4] Hyperparameter tuning (total target)\n", "=" * 70)
    tune.main()

    print("=" * 70, "\n[2/4] Training base models on all targets\n", "=" * 70)
    train.main()

    print("=" * 70, "\n[3/4] Stacking ensemble (OOF + Ridge meta)\n", "=" * 70)
    tier4_stacking.main()

    print("=" * 70, "\n[4/4] Held-out evaluation (2021-2023)\n", "=" * 70)
    evaluate.main()

    print(f"\nPipeline complete in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
