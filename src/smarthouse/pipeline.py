"""Pipeline orchestration for all stages."""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from smarthouse.config import Config

logger = logging.getLogger(__name__)


def _outputs_path(config: Config, filename: str) -> Path:
    outputs = Path(config.outputs_dir)
    if not outputs.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        outputs = repo_root / outputs
    return outputs / filename


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved %s (%d items)", path.name, len(obj) if hasattr(obj, "__len__") else 1)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_prep(config: Config) -> tuple:
    """Stage 1: Load/generate dataset and normalize sensors."""
    from smarthouse.data.loader import load_dataset
    from smarthouse.data.normalizer import normalize_events_df

    logger.info("=== Stage: prep ===")
    events_df, intervals_df, sensor_room_map = load_dataset(config)
    events_df = normalize_events_df(events_df)

    # Save outputs
    out_dir = _outputs_path(config, "").parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor_map_path = _outputs_path(config, "sensor_room_map.json")
    with open(sensor_map_path, "w") as f:
        json.dump(sensor_room_map, f, indent=2)
    logger.info("Saved sensor_room_map.json")

    intervals_path = _outputs_path(config, "activity_intervals.parquet")
    intervals_df.to_parquet(str(intervals_path), index=False)
    logger.info("Saved activity_intervals.parquet (%d rows)", len(intervals_df))

    events_path = _outputs_path(config, "events.parquet")
    events_df.to_parquet(str(events_path), index=False)
    logger.info("Saved events.parquet (%d rows)", len(events_df))

    return events_df, intervals_df, sensor_room_map


def run_overlap(config: Config) -> List[Dict[str, Any]]:
    """Stage 2: Extract overlap windows."""
    from smarthouse.overlap.extractor import build_real_overlap_windows

    logger.info("=== Stage: overlap ===")

    # Load prerequisite data
    events_path = _outputs_path(config, "events.parquet")
    intervals_path = _outputs_path(config, "activity_intervals.parquet")
    sensor_map_path = _outputs_path(config, "sensor_room_map.json")

    if not events_path.exists():
        run_prep(config)

    events_df = pd.read_parquet(str(events_path))
    intervals_df = pd.read_parquet(str(intervals_path))
    with open(sensor_map_path) as f:
        sensor_room_map = json.load(f)

    real_windows = build_real_overlap_windows(
        events_df, intervals_df, sensor_room_map, config
    )

    _save_pickle(real_windows, _outputs_path(config, "real_overlap_windows.pkl"))
    logger.info("Overlap stage complete: %d windows", len(real_windows))
    return real_windows


def run_templates(config: Config) -> Dict[str, List[dict]]:
    """Stage 3: Build activity templates."""
    from smarthouse.templates.builder import build_activity_templates

    logger.info("=== Stage: templates ===")

    events_path = _outputs_path(config, "events.parquet")
    intervals_path = _outputs_path(config, "activity_intervals.parquet")

    if not events_path.exists():
        run_prep(config)

    events_df = pd.read_parquet(str(events_path))
    intervals_df = pd.read_parquet(str(intervals_path))

    templates = build_activity_templates(events_df, intervals_df, config)
    _save_pickle(templates, _outputs_path(config, "activity_templates.pkl"))
    logger.info("Templates stage complete: %d activities", len(templates))
    return templates


def run_baseline(config: Config) -> List[Dict[str, Any]]:
    """Stage 4: Generate baseline windows."""
    from smarthouse.generators.baseline import generate_baseline_windows

    logger.info("=== Stage: baseline ===")

    real_path = _outputs_path(config, "real_overlap_windows.pkl")
    tmpl_path = _outputs_path(config, "activity_templates.pkl")

    if not real_path.exists():
        run_overlap(config)
    if not tmpl_path.exists():
        run_templates(config)

    real_windows = _load_pickle(real_path)
    templates = _load_pickle(tmpl_path)

    baseline_windows = generate_baseline_windows(real_windows, templates, config)
    _save_pickle(baseline_windows, _outputs_path(config, "baseline_windows.pkl"))
    logger.info("Baseline stage complete: %d windows", len(baseline_windows))
    return baseline_windows


def run_multiagent(config: Config) -> List[Dict[str, Any]]:
    """Stage 5: Generate raw multi-agent windows."""
    from smarthouse.generators.multiagent import generate_multiagent_windows_raw

    logger.info("=== Stage: multiagent ===")

    real_path = _outputs_path(config, "real_overlap_windows.pkl")
    tmpl_path = _outputs_path(config, "activity_templates.pkl")

    if not real_path.exists():
        run_overlap(config)
    if not tmpl_path.exists():
        run_templates(config)

    real_windows = _load_pickle(real_path)
    templates = _load_pickle(tmpl_path)

    raw_windows = generate_multiagent_windows_raw(real_windows, templates, config)
    _save_pickle(raw_windows, _outputs_path(config, "multiagent_windows_raw.pkl"))
    logger.info("Multi-agent stage complete: %d raw windows", len(raw_windows))
    return raw_windows


def run_validation(config: Config) -> List[Dict[str, Any]]:
    """Stage 6: Validate and repair multi-agent windows."""
    from smarthouse.validation.repair import run_validation_repair

    logger.info("=== Stage: validation ===")

    raw_path = _outputs_path(config, "multiagent_windows_raw.pkl")
    tmpl_path = _outputs_path(config, "activity_templates.pkl")

    if not raw_path.exists():
        run_multiagent(config)
    if not tmpl_path.exists():
        run_templates(config)

    raw_windows = _load_pickle(raw_path)
    templates = _load_pickle(tmpl_path)

    validated_windows = run_validation_repair(raw_windows, templates, config)
    _save_pickle(validated_windows, _outputs_path(config, "multiagent_windows_validated.pkl"))
    logger.info("Validation stage complete: %d windows", len(validated_windows))
    return validated_windows


def run_evaluation(config: Config) -> Dict[str, Any]:
    """Stage 7: Compute evaluation metrics."""
    from smarthouse.evaluation.metrics import run_ablation
    from smarthouse.evaluation.classifier import run_classifier_experiment

    logger.info("=== Stage: evaluation ===")

    paths = {
        "real": "real_overlap_windows.pkl",
        "baseline": "baseline_windows.pkl",
        "raw": "multiagent_windows_raw.pkl",
        "validated": "multiagent_windows_validated.pkl",
    }

    # Ensure prerequisites
    if not _outputs_path(config, paths["real"]).exists():
        run_overlap(config)
    if not _outputs_path(config, paths["baseline"]).exists():
        run_baseline(config)
    if not _outputs_path(config, paths["raw"]).exists():
        run_multiagent(config)
    if not _outputs_path(config, paths["validated"]).exists():
        run_validation(config)

    real_windows = _load_pickle(_outputs_path(config, paths["real"]))
    baseline_windows = _load_pickle(_outputs_path(config, paths["baseline"]))
    raw_windows = _load_pickle(_outputs_path(config, paths["raw"]))
    validated_windows = _load_pickle(_outputs_path(config, paths["validated"]))

    ablation_df = run_ablation(
        real_windows, baseline_windows, raw_windows, validated_windows, config
    )
    ablation_path = _outputs_path(config, "ablation_results.csv")
    ablation_df.to_csv(str(ablation_path), index=False)
    logger.info("Saved ablation_results.csv")

    clf_results = run_classifier_experiment(real_windows, baseline_windows, validated_windows)

    metrics = {
        "ablation": ablation_df.to_dict("records"),
        "classifier": clf_results,
    }

    def _json_default(obj):
        import math
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return str(obj)

    metrics_path = _outputs_path(config, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Saved evaluation_metrics.json")

    return metrics


def run_visualization(config: Config) -> None:
    """Stage 8: Generate visualizations."""
    from smarthouse.visualization.plots import generate_qualitative_examples

    logger.info("=== Stage: visualization ===")

    real_windows = _load_pickle(_outputs_path(config, "real_overlap_windows.pkl"))
    baseline_windows = _load_pickle(_outputs_path(config, "baseline_windows.pkl"))
    validated_windows = _load_pickle(_outputs_path(config, "multiagent_windows_validated.pkl"))

    outputs = Path(config.outputs_dir)
    if not outputs.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        outputs = repo_root / outputs
    outputs_dir = str(outputs)

    generate_qualitative_examples(
        real_windows, baseline_windows, validated_windows, outputs_dir, n=5
    )
    logger.info("Visualization stage complete")


def run_all(config: Config) -> None:
    """Run all pipeline stages in order."""
    logger.info("=== Running full pipeline ===")
    run_prep(config)
    run_overlap(config)
    run_templates(config)
    run_baseline(config)
    run_multiagent(config)
    run_validation(config)
    run_evaluation(config)
    run_visualization(config)
    logger.info("=== Pipeline complete ===")
