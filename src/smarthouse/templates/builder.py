"""Activity template building from real overlap windows."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smarthouse.config import Config

logger = logging.getLogger(__name__)


def _relativize_events(events: List[dict]) -> List[dict]:
    """Convert event timestamps to relative seconds from first event."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e["timestamp"])
    t0 = events[0]["timestamp"]
    rel_events = []
    for e in events:
        rel = {
            "rel_seconds": (e["timestamp"] - t0).total_seconds(),
            "sensor_id": e["sensor_id"],
            "sensor_state": e["sensor_state"],
            "room": e.get("room", "unknown"),
        }
        rel_events.append(rel)
    return rel_events


def build_activity_templates(
    events_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    config: Config,
) -> Dict[str, List[dict]]:
    """Build per-activity event templates from intervals."""
    templates: Dict[str, List[dict]] = {}

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    intervals_df = intervals_df.copy()
    intervals_df["start_time"] = pd.to_datetime(intervals_df["start_time"])
    intervals_df["end_time"] = pd.to_datetime(intervals_df["end_time"])

    for _, iv in intervals_df.iterrows():
        activity = iv["activity_label"]
        if activity not in config.SELECTED_ACTIVITIES:
            continue

        mask = (
            (events_df["resident_label"] == iv["resident_id"])
            & (events_df["timestamp"] >= iv["start_time"])
            & (events_df["timestamp"] <= iv["end_time"])
        )
        ev = events_df[mask].to_dict("records")
        if len(ev) < 2:
            continue

        duration = (iv["end_time"] - iv["start_time"]).total_seconds()
        rel_events = _relativize_events(ev)
        rooms = list({e["room"] for e in ev})
        sensors = list({e["sensor_id"] for e in ev})

        template = {
            "activity": activity,
            "relative_events": rel_events,
            "duration_seconds": duration,
            "rooms": rooms,
            "sensors": sensors,
            "event_count": len(ev),
        }
        templates.setdefault(activity, []).append(template)

    # Log stats
    for activity, tmpl_list in templates.items():
        durations = [t["duration_seconds"] for t in tmpl_list]
        counts = [t["event_count"] for t in tmpl_list]
        logger.info(
            "Activity %s: %d templates, mean_duration=%.1fs, mean_events=%.1f",
            activity, len(tmpl_list),
            float(np.mean(durations)) if durations else 0,
            float(np.mean(counts)) if counts else 0,
        )

    return templates


def sample_template(
    activity: str,
    templates: Dict[str, List[dict]],
    rng: np.random.Generator,
) -> Optional[dict]:
    """Sample one template for the given activity."""
    tmpl_list = templates.get(activity, [])
    if not tmpl_list:
        return None
    idx = int(rng.integers(0, len(tmpl_list)))
    return tmpl_list[idx]


def get_template_stats(templates: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Get statistics per activity from templates."""
    stats = {}
    for activity, tmpl_list in templates.items():
        if not tmpl_list:
            continue
        durations = [t["duration_seconds"] for t in tmpl_list]
        counts = [t["event_count"] for t in tmpl_list]
        all_sensors = [s for t in tmpl_list for s in t["sensors"]]
        sensor_freq: Dict[str, int] = {}
        for s in all_sensors:
            sensor_freq[s] = sensor_freq.get(s, 0) + 1
        common_sensors = sorted(sensor_freq, key=lambda x: -sensor_freq[x])[:5]

        all_rooms = [r for t in tmpl_list for r in t["rooms"]]
        room_freq: Dict[str, int] = {}
        for r in all_rooms:
            room_freq[r] = room_freq.get(r, 0) + 1
        dominant_room = max(room_freq, key=lambda x: room_freq[x]) if room_freq else "unknown"

        stats[activity] = {
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "mean_event_count": float(np.mean(counts)),
            "dominant_room": dominant_room,
            "common_sensors": common_sensors,
        }
    return stats
