#!/usr/bin/env python3
"""Ablation study runner."""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from smarthouse.config import load_config
    from smarthouse.pipeline import (
        _load_pickle, _outputs_path,
        run_overlap, run_baseline, run_multiagent, run_validation,
    )
    from smarthouse.evaluation.metrics import run_ablation

    config = load_config()

    # Ensure all windows exist
    if not _outputs_path(config, "real_overlap_windows.pkl").exists():
        run_overlap(config)
    if not _outputs_path(config, "baseline_windows.pkl").exists():
        run_baseline(config)
    if not _outputs_path(config, "multiagent_windows_raw.pkl").exists():
        run_multiagent(config)
    if not _outputs_path(config, "multiagent_windows_validated.pkl").exists():
        run_validation(config)

    real_windows = _load_pickle(_outputs_path(config, "real_overlap_windows.pkl"))
    baseline_windows = _load_pickle(_outputs_path(config, "baseline_windows.pkl"))
    raw_windows = _load_pickle(_outputs_path(config, "multiagent_windows_raw.pkl"))
    validated_windows = _load_pickle(_outputs_path(config, "multiagent_windows_validated.pkl"))

    ablation_df = run_ablation(
        real_windows, baseline_windows, raw_windows, validated_windows, config
    )

    out_path = _outputs_path(config, "ablation_results.csv")
    ablation_df.to_csv(str(out_path), index=False)
    logger.info("Ablation results saved to %s", out_path)
    print(ablation_df.to_string())


if __name__ == "__main__":
    main()
