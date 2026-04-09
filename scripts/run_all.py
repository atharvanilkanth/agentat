#!/usr/bin/env python3
"""CLI entrypoint for the smarthouse pipeline."""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart-house pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stages = [
        "run-all", "prep", "overlap", "templates",
        "baseline", "multiagent", "validation", "evaluation", "visualization"
    ]
    for stage in stages:
        subparsers.add_parser(stage, help=f"Run {stage} stage")

    args = parser.parse_args()

    from smarthouse.config import load_config
    config = load_config()

    from smarthouse import pipeline

    stage_map = {
        "run-all": pipeline.run_all,
        "prep": pipeline.run_prep,
        "overlap": pipeline.run_overlap,
        "templates": pipeline.run_templates,
        "baseline": pipeline.run_baseline,
        "multiagent": pipeline.run_multiagent,
        "validation": pipeline.run_validation,
        "evaluation": pipeline.run_evaluation,
        "visualization": pipeline.run_visualization,
    }

    fn = stage_map[args.command]
    try:
        fn(config)
    except Exception as e:
        logger.exception("Pipeline stage '%s' failed: %s", args.command, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
