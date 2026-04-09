"""Multi-agent window generator."""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from smarthouse.config import Config
from smarthouse.templates.builder import sample_template

logger = logging.getLogger(__name__)


def _materialize_template_for_resident(
    template: dict,
    start_time: datetime,
    target_duration: float,
    resident_id: str,
    activity: str,
    rng: np.random.Generator,
) -> List[dict]:
    """Materialize template into absolute events with per-resident identity preserved."""
    rel_events = template["relative_events"]
    if not rel_events:
        return []

    src_duration = template["duration_seconds"]
    if src_duration <= 0:
        src_duration = 1.0
    scale = target_duration / src_duration

    events = []
    for re in rel_events:
        jitter = float(rng.normal(0, 1.0))
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


def generate_multiagent_windows_raw(
    real_windows: List[Dict[str, Any]],
    templates: Dict[str, List[dict]],
    config: Config,
) -> List[Dict[str, Any]]:
    """
    Generate raw multi-agent windows by synthesizing both residents' streams
    from templates, then merging and sorting by timestamp.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 2)
    raw_windows = []

    for rw in real_windows:
        tmpl_a = sample_template(rw["activity_a"], templates, rng)
        tmpl_b = sample_template(rw["activity_b"], templates, rng)

        if tmpl_a is None or tmpl_b is None:
            logger.debug(
                "No template for %s or %s, skipping",
                rw["activity_a"], rw["activity_b"],
            )
            continue

        overlap_duration = rw["overlap_duration_seconds"]

        events_a = _materialize_template_for_resident(
            tmpl_a,
            rw["overlap_start"],
            max(overlap_duration, 60.0),
            rw["resident_a"],
            rw["activity_a"],
            rng,
        )
        events_b = _materialize_template_for_resident(
            tmpl_b,
            rw["overlap_start"],
            max(overlap_duration, 60.0),
            rw["resident_b"],
            rw["activity_b"],
            rng,
        )

        # Deterministic merge: sort by (timestamp, resident_label, sensor_id)
        merged = sorted(
            events_a + events_b,
            key=lambda e: (e["timestamp"], e["resident_label"], e["sensor_id"])
        )

        raw_window = {
            "window_id": str(uuid.uuid4()),
            "source_window_id": rw["window_id"],
            "resident_a": rw["resident_a"],
            "activity_a": rw["activity_a"],
            "resident_b": rw["resident_b"],
            "activity_b": rw["activity_b"],
            "overlap_start": rw["overlap_start"],
            "overlap_end": rw["overlap_end"],
            "overlap_duration_seconds": overlap_duration,
            "events_a": events_a,
            "events_b": events_b,
            "events_a_overlap": events_a,
            "events_b_overlap": events_b,
            "merged_events": merged,
            "rooms_a": list({e["room"] for e in events_a}),
            "rooms_b": list({e["room"] for e in events_b}),
            "interval_id_a": rw["interval_id_a"],
            "interval_id_b": rw["interval_id_b"],
            "generator": "multiagent_raw",
            "violations": [],
        }
        raw_windows.append(raw_window)

    logger.info("Generated %d raw multi-agent windows", len(raw_windows))
    return raw_windows
