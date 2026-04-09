"""Single-agent baseline window generator."""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from smarthouse.config import Config
from smarthouse.templates.builder import sample_template

logger = logging.getLogger(__name__)


def _materialize_template(
    template: dict,
    start_time: datetime,
    target_duration: float,
    resident_id: str,
    activity: str,
    rng: np.random.Generator,
) -> List[dict]:
    """Materialize a relative template into absolute events, scaled to target_duration."""
    rel_events = template["relative_events"]
    if not rel_events:
        return []

    src_duration = template["duration_seconds"]
    if src_duration <= 0:
        src_duration = 1.0
    scale = target_duration / src_duration

    events = []
    for re in rel_events:
        jitter = float(rng.normal(0, 0.5))
        offset = max(0.0, re["rel_seconds"] * scale + jitter)
        ts = start_time + timedelta(seconds=offset)
        events.append({
            "timestamp": ts,
            "sensor_id": re["sensor_id"],
            "sensor_state": re["sensor_state"],
            "room": re["room"],
            "activity_label": activity,
            "resident_label": resident_id,
            "event_type": "synthetic",
        })
    return sorted(events, key=lambda e: e["timestamp"])


def generate_baseline_windows(
    real_windows: List[Dict[str, Any]],
    templates: Dict[str, List[dict]],
    config: Config,
) -> List[Dict[str, Any]]:
    """
    Generate baseline windows using Policy A: keep resident with longer activity.
    Single-agent baseline: only one resident's activity stream is kept.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    baseline_windows = []

    for rw in real_windows:
        # Policy A: keep resident with longer activity duration
        events_a_full = rw["events_a"]
        events_b_full = rw["events_b"]
        dur_a_full = (
            max((e["timestamp"] for e in events_a_full), default=rw["overlap_end"])
            - min((e["timestamp"] for e in events_a_full), default=rw["overlap_start"])
        ).total_seconds() if events_a_full else 0
        dur_b_full = (
            max((e["timestamp"] for e in events_b_full), default=rw["overlap_end"])
            - min((e["timestamp"] for e in events_b_full), default=rw["overlap_start"])
        ).total_seconds() if events_b_full else 0

        if dur_a_full >= dur_b_full:
            kept_activity = rw["activity_a"]
            kept_resident = rw["resident_a"]
        else:
            kept_activity = rw["activity_b"]
            kept_resident = rw["resident_b"]

        tmpl = sample_template(kept_activity, templates, rng)
        if tmpl is None:
            logger.debug("No template for activity %s, skipping", kept_activity)
            continue

        overlap_duration = rw["overlap_duration_seconds"]
        events = _materialize_template(
            tmpl,
            rw["overlap_start"],
            max(overlap_duration, 60.0),
            kept_resident,
            kept_activity,
            rng,
        )

        baseline_window = {
            "window_id": str(uuid.uuid4()),
            "source_window_id": rw["window_id"],
            "resident_a": kept_resident,
            "activity_a": kept_activity,
            "resident_b": None,
            "activity_b": None,
            "overlap_start": rw["overlap_start"],
            "overlap_end": rw["overlap_end"],
            "overlap_duration_seconds": overlap_duration,
            "events_a": events,
            "events_b": [],
            "events_a_overlap": events,
            "events_b_overlap": [],
            "merged_events": events,
            "rooms_a": list({e["room"] for e in events}),
            "rooms_b": [],
            "interval_id_a": rw["interval_id_a"],
            "interval_id_b": rw["interval_id_b"],
            "generator": "baseline",
            "violations": [],
        }
        baseline_windows.append(baseline_window)

    logger.info("Generated %d baseline windows", len(baseline_windows))
    return baseline_windows
