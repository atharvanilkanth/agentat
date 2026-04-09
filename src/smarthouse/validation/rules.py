"""Coordination rule validation for overlap windows."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

BATHROOM_EXCLUSIVE_SENSORS = {"M003", "M004", "D001"}


def check_teleportation(resident_events: List[dict]) -> List[dict]:
    """
    Check for teleportation: resident appearing in two different rooms
    within an impossibly short time (< 5 seconds).
    """
    violations = []
    sorted_events = sorted(resident_events, key=lambda e: e["timestamp"])
    for i in range(1, len(sorted_events)):
        prev = sorted_events[i - 1]
        curr = sorted_events[i]
        dt = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        if (
            dt < 5.0
            and prev.get("room") != curr.get("room")
            and prev.get("room") not in ("unknown", None)
            and curr.get("room") not in ("unknown", None)
        ):
            violations.append({
                "type": "teleportation",
                "resident": curr.get("resident_label"),
                "from_room": prev.get("room"),
                "to_room": curr.get("room"),
                "time_delta_seconds": dt,
                "timestamp": curr["timestamp"],
            })
    return violations


def check_exclusive_resource(window: Dict[str, Any]) -> List[dict]:
    """Check that bathroom is not used by both residents simultaneously."""
    violations = []
    events_a = window.get("events_a_overlap", [])
    events_b = window.get("events_b_overlap", [])

    bathroom_times_a = [
        e["timestamp"] for e in events_a
        if e.get("room") == "bathroom"
    ]
    bathroom_times_b = [
        e["timestamp"] for e in events_b
        if e.get("room") == "bathroom"
    ]

    if bathroom_times_a and bathroom_times_b:
        for ta in bathroom_times_a:
            for tb in bathroom_times_b:
                if abs((ta - tb).total_seconds()) < 30.0:
                    violations.append({
                        "type": "exclusive_resource",
                        "resource": "bathroom",
                        "resident_a": window.get("resident_a"),
                        "resident_b": window.get("resident_b"),
                        "time_a": ta,
                        "time_b": tb,
                    })
                    break
            else:
                continue
            break

    return violations


def check_activity_room_alignment(window: Dict[str, Any]) -> List[dict]:
    """Check that events occur in rooms consistent with the activity."""
    from smarthouse.data.loader import ACTIVITY_ROOM_MAP

    violations = []
    for resident_key, events_key in [
        ("activity_a", "events_a_overlap"),
        ("activity_b", "events_b_overlap"),
    ]:
        activity = window.get(resident_key)
        events = window.get(events_key, [])
        if not activity or not events:
            continue
        expected_rooms = ACTIVITY_ROOM_MAP.get(activity, [])
        if not expected_rooms:
            continue
        for e in events:
            room = e.get("room", "unknown")
            if room not in ("unknown", None) and room not in expected_rooms:
                violations.append({
                    "type": "activity_room_mismatch",
                    "activity": activity,
                    "expected_rooms": expected_rooms,
                    "actual_room": room,
                    "sensor_id": e.get("sensor_id"),
                    "timestamp": e["timestamp"],
                })
    return violations


def check_overlap_preservation(window: Dict[str, Any]) -> List[dict]:
    """Check that both residents have events in the overlap window."""
    violations = []
    events_a = window.get("events_a_overlap", [])
    events_b = window.get("events_b_overlap", [])

    if not events_a:
        violations.append({
            "type": "missing_resident_events",
            "resident": window.get("resident_a"),
            "detail": "No events for resident A in overlap window",
        })
    if not events_b:
        violations.append({
            "type": "missing_resident_events",
            "resident": window.get("resident_b"),
            "detail": "No events for resident B in overlap window",
        })
    return violations


def check_contradictory_states(resident_events: List[dict]) -> List[dict]:
    """Check for sensor showing contradictory states at the same time."""
    violations = []
    sorted_events = sorted(resident_events, key=lambda e: e["timestamp"])

    # Group by sensor, look for rapid state oscillations
    sensor_states: Dict[str, list] = {}
    for e in sorted_events:
        sid = e.get("sensor_id")
        if not sid:
            continue
        sensor_states.setdefault(sid, []).append(
            (e["timestamp"], e.get("sensor_state"))
        )

    for sid, state_list in sensor_states.items():
        for i in range(1, len(state_list)):
            prev_t, prev_s = state_list[i - 1]
            curr_t, curr_s = state_list[i]
            dt = (curr_t - prev_t).total_seconds()
            # Same state repeated immediately is fine; contradictory is same-sensor, same state within 1s
            if dt < 1.0 and prev_s == curr_s and prev_s in ("ON", "OPEN"):
                violations.append({
                    "type": "contradictory_state",
                    "sensor_id": sid,
                    "state": curr_s,
                    "time_delta_seconds": dt,
                    "timestamp": curr_t,
                })

    return violations


def validate_window(window: Dict[str, Any]) -> List[dict]:
    """Run all validation checks on a window. Returns list of violations."""
    violations = []

    events_a = window.get("events_a_overlap", [])
    events_b = window.get("events_b_overlap", [])

    if events_a:
        violations.extend(check_teleportation(events_a))
        violations.extend(check_contradictory_states(events_a))
    if events_b:
        violations.extend(check_teleportation(events_b))
        violations.extend(check_contradictory_states(events_b))

    violations.extend(check_exclusive_resource(window))
    violations.extend(check_activity_room_alignment(window))
    violations.extend(check_overlap_preservation(window))

    return violations


def tag_violations(window: Dict[str, Any]) -> Dict[str, Any]:
    """Tag a window with its violations."""
    w = dict(window)
    w["violations"] = validate_window(window)
    w["has_violations"] = len(w["violations"]) > 0
    return w
