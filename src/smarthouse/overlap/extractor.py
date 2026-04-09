"""Overlap window extraction between residents' activity intervals."""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smarthouse.config import Config

logger = logging.getLogger(__name__)


def find_overlapping_pairs(intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Find pairs of intervals (different residents) that overlap in time."""
    residents = intervals_df["resident_id"].unique()
    if len(residents) < 2:
        logger.warning("Only one resident found; no overlapping pairs possible.")
        return pd.DataFrame()

    records = []
    df = intervals_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # For each pair of resident combinations
    for i, r1 in enumerate(residents):
        for r2 in residents[i + 1:]:
            df_a = df[df["resident_id"] == r1]
            df_b = df[df["resident_id"] == r2]

            for _, row_a in df_a.iterrows():
                for _, row_b in df_b.iterrows():
                    overlap_start = max(row_a["start_time"], row_b["start_time"])
                    overlap_end = min(row_a["end_time"], row_b["end_time"])
                    if overlap_start < overlap_end:
                        duration = (overlap_end - overlap_start).total_seconds()
                        records.append({
                            "interval_id_a": row_a["interval_id"],
                            "interval_id_b": row_b["interval_id"],
                            "resident_a": r1,
                            "resident_b": r2,
                            "activity_a": row_a["activity_label"],
                            "activity_b": row_b["activity_label"],
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_duration_seconds": duration,
                        })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _events_for_interval(
    events_df: pd.DataFrame,
    interval_row: pd.Series,
) -> List[dict]:
    """Get events for a specific activity interval."""
    mask = (
        (events_df["resident_label"] == interval_row["resident_id"])
        & (events_df["timestamp"] >= interval_row["start_time"])
        & (events_df["timestamp"] <= interval_row["end_time"])
    )
    sub = events_df[mask].copy()
    return sub.to_dict("records")


def _events_in_window(events: List[dict], start: datetime, end: datetime) -> List[dict]:
    """Filter events to those within [start, end]."""
    return [e for e in events if start <= e["timestamp"] <= end]


def extract_overlap_windows(
    events_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    sensor_room_map: Dict[str, str],
    config: Config,
) -> List[Dict[str, Any]]:
    """Extract all overlap windows from the dataset."""
    pairs = find_overlapping_pairs(intervals_df)
    if pairs.empty:
        logger.warning("No overlapping pairs found.")
        return []

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    intervals_df = intervals_df.copy()
    intervals_df["start_time"] = pd.to_datetime(intervals_df["start_time"])
    intervals_df["end_time"] = pd.to_datetime(intervals_df["end_time"])

    # Index intervals for fast lookup
    iv_index = intervals_df.set_index("interval_id")

    windows = []
    for _, pair in pairs.iterrows():
        row_a = iv_index.loc[pair["interval_id_a"]]
        row_b = iv_index.loc[pair["interval_id_b"]]

        events_a = _events_for_interval(events_df, row_a)
        events_b = _events_for_interval(events_df, row_b)

        overlap_start = pair["overlap_start"]
        overlap_end = pair["overlap_end"]

        events_a_overlap = _events_in_window(events_a, overlap_start, overlap_end)
        events_b_overlap = _events_in_window(events_b, overlap_start, overlap_end)

        merged = sorted(
            events_a_overlap + events_b_overlap,
            key=lambda e: (e["timestamp"], e["resident_label"], e["sensor_id"])
        )

        rooms_a = list({e["room"] for e in events_a if e["room"]})
        rooms_b = list({e["room"] for e in events_b if e["room"]})

        window = {
            "window_id": str(uuid.uuid4()),
            "resident_a": pair["resident_a"],
            "activity_a": pair["activity_a"],
            "resident_b": pair["resident_b"],
            "activity_b": pair["activity_b"],
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_duration_seconds": pair["overlap_duration_seconds"],
            "events_a": events_a,
            "events_b": events_b,
            "events_a_overlap": events_a_overlap,
            "events_b_overlap": events_b_overlap,
            "merged_events": merged,
            "rooms_a": rooms_a,
            "rooms_b": rooms_b,
            "interval_id_a": pair["interval_id_a"],
            "interval_id_b": pair["interval_id_b"],
        }
        windows.append(window)

    logger.info("Extracted %d raw overlap windows", len(windows))
    return windows


def filter_quality_windows(
    windows: List[Dict[str, Any]],
    min_duration_seconds: float = 120.0,
    min_events_per_resident: int = 2,
) -> List[Dict[str, Any]]:
    """Filter windows based on quality criteria."""
    filtered = []
    for w in windows:
        if w["overlap_duration_seconds"] < min_duration_seconds:
            continue
        if len(w["events_a_overlap"]) < min_events_per_resident:
            continue
        if len(w["events_b_overlap"]) < min_events_per_resident:
            continue
        filtered.append(w)
    logger.info(
        "Quality filtering: %d -> %d windows", len(windows), len(filtered)
    )
    return filtered


def build_real_overlap_windows(
    events_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    sensor_room_map: Dict[str, str],
    config: Config,
) -> List[Dict[str, Any]]:
    """Build the final gold set of overlap windows."""
    raw_windows = extract_overlap_windows(
        events_df, intervals_df, sensor_room_map, config
    )
    quality_windows = filter_quality_windows(
        raw_windows,
        min_duration_seconds=config.MIN_OVERLAP_SECONDS,
        min_events_per_resident=config.MIN_EVENTS,
    )
    logger.info("Final gold set: %d overlap windows", len(quality_windows))
    return quality_windows
