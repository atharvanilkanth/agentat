"""Deterministic repair module for validation violations."""
import copy
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from smarthouse.config import Config
from smarthouse.templates.builder import sample_template
from smarthouse.validation.rules import validate_window

logger = logging.getLogger(__name__)


def repair_teleportation(
    window: Dict[str, Any],
    events: List[dict],
    rng: np.random.Generator,
) -> List[dict]:
    """Fix teleportation by removing the offending event."""
    if not events:
        return events

    fixed = []
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    fixed.append(sorted_events[0])

    for i in range(1, len(sorted_events)):
        prev = fixed[-1]
        curr = sorted_events[i]
        dt = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        if (
            dt < 5.0
            and prev.get("room") != curr.get("room")
            and prev.get("room") not in ("unknown", None)
        ):
            # Skip the teleporting event
            continue
        fixed.append(curr)

    return fixed


def repair_exclusive_conflict(
    window: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fix bathroom exclusivity by time-shifting B's bathroom events."""
    w = copy.deepcopy(window)
    events_a = w.get("events_a_overlap", [])
    events_b = w.get("events_b_overlap", [])

    bathroom_times_a = sorted([
        e["timestamp"] for e in events_a if e.get("room") == "bathroom"
    ])
    if not bathroom_times_a:
        return w

    latest_a = max(bathroom_times_a)

    new_events_b = []
    for e in events_b:
        if e.get("room") == "bathroom":
            # Shift to after resident A leaves bathroom
            shift = float(rng.uniform(60, 300))
            e = dict(e)
            e["timestamp"] = latest_a + timedelta(seconds=shift)
        new_events_b.append(e)

    new_events_b.sort(key=lambda e: e["timestamp"])
    w["events_b_overlap"] = new_events_b

    # Rebuild merged
    w["merged_events"] = sorted(
        w["events_a_overlap"] + w["events_b_overlap"],
        key=lambda e: (e["timestamp"], e.get("resident_label", ""), e.get("sensor_id", ""))
    )
    return w


def repair_contradictory_states(events: List[dict]) -> List[dict]:
    """Fix contradictory states by deduplicating rapid same-state events."""
    if not events:
        return events

    fixed = []
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    fixed.append(sorted_events[0])

    for i in range(1, len(sorted_events)):
        prev = fixed[-1]
        curr = sorted_events[i]
        dt = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        if (
            dt < 1.0
            and prev.get("sensor_id") == curr.get("sensor_id")
            and prev.get("sensor_state") == curr.get("sensor_state")
        ):
            continue
        fixed.append(curr)

    return fixed


def repair_activity_collapse(
    window: Dict[str, Any],
    templates: Dict[str, List[dict]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Re-generate events from template if activity has collapsed (no events)."""
    from smarthouse.generators.multiagent import _materialize_template_for_resident

    w = copy.deepcopy(window)

    for resident_key, activity_key, events_key in [
        ("resident_a", "activity_a", "events_a_overlap"),
        ("resident_b", "activity_b", "events_b_overlap"),
    ]:
        events = w.get(events_key, [])
        if not events:
            activity = w.get(activity_key)
            resident = w.get(resident_key)
            if not activity or not resident:
                continue
            tmpl = sample_template(activity, templates, rng)
            if tmpl is None:
                continue
            duration = w["overlap_duration_seconds"]
            new_events = _materialize_template_for_resident(
                tmpl,
                w["overlap_start"],
                max(duration, 60.0),
                resident,
                activity,
                rng,
            )
            w[events_key] = new_events

    # Rebuild merged
    w["merged_events"] = sorted(
        w.get("events_a_overlap", []) + w.get("events_b_overlap", []),
        key=lambda e: (e["timestamp"], e.get("resident_label", ""), e.get("sensor_id", ""))
    )
    return w


def repair_window(
    window: Dict[str, Any],
    templates: Dict[str, List[dict]],
    config: Config,
    rng: np.random.Generator,
    max_attempts: int = 3,
) -> Optional[Dict[str, Any]]:
    """Attempt to repair a window with violations. Returns repaired window or None."""
    w = copy.deepcopy(window)

    for attempt in range(max_attempts):
        violations = validate_window(w)
        if not violations:
            return w

        vtypes = {v["type"] for v in violations}

        if "missing_resident_events" in vtypes:
            w = repair_activity_collapse(w, templates, rng)

        if "teleportation" in vtypes:
            w["events_a_overlap"] = repair_teleportation(w, w.get("events_a_overlap", []), rng)
            w["events_b_overlap"] = repair_teleportation(w, w.get("events_b_overlap", []), rng)

        if "exclusive_resource" in vtypes:
            w = repair_exclusive_conflict(w, rng)

        if "contradictory_state" in vtypes:
            w["events_a_overlap"] = repair_contradictory_states(w.get("events_a_overlap", []))
            w["events_b_overlap"] = repair_contradictory_states(w.get("events_b_overlap", []))

        # Rebuild merged after repairs
        w["merged_events"] = sorted(
            w.get("events_a_overlap", []) + w.get("events_b_overlap", []),
            key=lambda e: (e["timestamp"], e.get("resident_label", ""), e.get("sensor_id", ""))
        )

    # Final check
    final_violations = validate_window(w)
    w["violations"] = final_violations
    w["has_violations"] = len(final_violations) > 0

    if final_violations:
        logger.debug(
            "Window %s still has %d violations after repair",
            w.get("window_id"), len(final_violations)
        )
    return w


def run_validation_repair(
    raw_windows: List[Dict[str, Any]],
    templates: Dict[str, List[dict]],
    config: Config,
) -> List[Dict[str, Any]]:
    """Validate and repair all windows. Returns validated windows."""
    rng = np.random.default_rng(config.RANDOM_SEED + 3)
    validated = []
    n_clean = 0
    n_repaired = 0
    n_failed = 0

    for w in raw_windows:
        violations = validate_window(w)
        if not violations:
            w = dict(w)
            w["violations"] = []
            w["has_violations"] = False
            validated.append(w)
            n_clean += 1
            continue

        repaired = repair_window(w, templates, config, rng)
        if repaired is not None:
            validated.append(repaired)
            if repaired.get("has_violations"):
                n_failed += 1
            else:
                n_repaired += 1
        else:
            n_failed += 1

    logger.info(
        "Validation/repair: %d clean, %d repaired, %d failed (total=%d)",
        n_clean, n_repaired, n_failed, len(validated)
    )
    return validated
