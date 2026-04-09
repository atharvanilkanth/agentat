"""Unit tests for overlap logic."""
import pytest
from datetime import datetime, timedelta
import pandas as pd

from smarthouse.overlap.extractor import (
    find_overlapping_pairs,
    filter_quality_windows,
)


def make_intervals(data):
    return pd.DataFrame(data, columns=["interval_id", "resident_id", "activity_label", "start_time", "end_time"])


def test_no_overlap():
    t0 = datetime(2023, 1, 1, 8, 0, 0)
    intervals = make_intervals([
        ("IV00001", "R1", "Breakfast", t0, t0 + timedelta(minutes=30)),
        ("IV00002", "R2", "Breakfast", t0 + timedelta(hours=2), t0 + timedelta(hours=2, minutes=30)),
    ])
    pairs = find_overlapping_pairs(intervals)
    assert pairs.empty or len(pairs) == 0


def test_overlap_detected():
    t0 = datetime(2023, 1, 1, 8, 0, 0)
    intervals = make_intervals([
        ("IV00001", "R1", "Breakfast", t0, t0 + timedelta(minutes=45)),
        ("IV00002", "R2", "Dinner", t0 + timedelta(minutes=15), t0 + timedelta(minutes=60)),
    ])
    pairs = find_overlapping_pairs(intervals)
    assert len(pairs) == 1
    assert pairs.iloc[0]["overlap_duration_seconds"] == pytest.approx(1800.0)


def test_quality_filter_duration():
    windows = [
        {"overlap_duration_seconds": 60, "events_a_overlap": [1, 2, 3], "events_b_overlap": [1, 2, 3]},
        {"overlap_duration_seconds": 200, "events_a_overlap": [1, 2, 3], "events_b_overlap": [1, 2, 3]},
    ]
    filtered = filter_quality_windows(windows, min_duration_seconds=120, min_events_per_resident=2)
    assert len(filtered) == 1
    assert filtered[0]["overlap_duration_seconds"] == 200


def test_quality_filter_events():
    windows = [
        {"overlap_duration_seconds": 200, "events_a_overlap": [1], "events_b_overlap": [1, 2, 3]},
        {"overlap_duration_seconds": 200, "events_a_overlap": [1, 2, 3], "events_b_overlap": [1, 2, 3]},
    ]
    filtered = filter_quality_windows(windows, min_duration_seconds=120, min_events_per_resident=2)
    assert len(filtered) == 1


def test_single_resident_no_pairs():
    t0 = datetime(2023, 1, 1, 8, 0, 0)
    intervals = make_intervals([
        ("IV00001", "R1", "Breakfast", t0, t0 + timedelta(minutes=30)),
        ("IV00002", "R1", "Dinner", t0 + timedelta(minutes=10), t0 + timedelta(minutes=50)),
    ])
    pairs = find_overlapping_pairs(intervals)
    assert pairs.empty or len(pairs) == 0
