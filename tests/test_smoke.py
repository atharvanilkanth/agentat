"""Smoke test for the full pipeline on a small sample."""
import pytest
import logging
from unittest.mock import patch

logger = logging.getLogger(__name__)


@pytest.fixture
def small_config():
    from smarthouse.config import Config
    return Config(
        dataset_name="cairo",
        data_dir="data/cairo",
        event_file="data.csv",
        activity_file="activities.csv",
        selected_activities=["Breakfast", "Dinner", "Work_in_Office", "Sleep", "Wake"],
        min_overlap_seconds=30.0,
        min_events=1,
        time_bin_seconds=30,
        random_seed=42,
        outputs_dir="outputs",
    )


def test_synthetic_data_generation(small_config):
    from smarthouse.data.loader import generate_synthetic_data

    events_df, intervals_df, sensor_room_map = generate_synthetic_data(small_config, n_days=5)

    assert len(events_df) > 0, "Should have events"
    assert len(intervals_df) > 0, "Should have intervals"
    assert "timestamp" in events_df.columns
    assert "sensor_id" in events_df.columns
    assert "interval_id" in intervals_df.columns
    assert "resident_id" in intervals_df.columns
    assert isinstance(sensor_room_map, dict)
    assert len(sensor_room_map) > 0


def test_overlap_extraction(small_config):
    from smarthouse.data.loader import generate_synthetic_data
    from smarthouse.overlap.extractor import build_real_overlap_windows

    events_df, intervals_df, sensor_room_map = generate_synthetic_data(small_config, n_days=5)
    windows = build_real_overlap_windows(events_df, intervals_df, sensor_room_map, small_config)

    assert isinstance(windows, list)
    if windows:
        w = windows[0]
        assert "window_id" in w
        assert "overlap_start" in w
        assert "overlap_end" in w
        assert "events_a_overlap" in w
        assert "events_b_overlap" in w


def test_templates(small_config):
    from smarthouse.data.loader import generate_synthetic_data
    from smarthouse.templates.builder import build_activity_templates

    events_df, intervals_df, _ = generate_synthetic_data(small_config, n_days=5)
    templates = build_activity_templates(events_df, intervals_df, small_config)

    assert isinstance(templates, dict)
    assert len(templates) > 0


def test_full_pipeline_smoke(small_config, tmp_path):
    """Run the full pipeline on a small sample."""
    import pickle
    import json
    from pathlib import Path
    from unittest.mock import patch

    # Patch outputs dir to tmp_path
    small_config_patched = small_config.__class__(
        **{f: getattr(small_config, f) for f in small_config.__dataclass_fields__
           if f != "outputs_dir"},
        outputs_dir=str(tmp_path),
    )

    # Patch pipeline to use tmp outputs
    from smarthouse import pipeline
    original_outputs_path = pipeline._outputs_path

    def patched_outputs_path(cfg, filename):
        return tmp_path / filename

    with patch.object(pipeline, "_outputs_path", patched_outputs_path):
        # Also patch load_dataset to use synthetic
        from smarthouse.data import loader
        with patch.object(loader, "load_dataset",
                          side_effect=lambda cfg: loader.generate_synthetic_data(cfg, n_days=3)):
            pipeline.run_all(small_config_patched)

    # Check that key outputs exist
    assert (tmp_path / "real_overlap_windows.pkl").exists()
    assert (tmp_path / "activity_templates.pkl").exists()
    assert (tmp_path / "baseline_windows.pkl").exists()
    assert (tmp_path / "multiagent_windows_raw.pkl").exists()
    assert (tmp_path / "multiagent_windows_validated.pkl").exists()
    assert (tmp_path / "evaluation_metrics.json").exists()
