"""Unit tests for validation rules."""
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List

from smarthouse.validation.rules import (
    check_teleportation,
    check_exclusive_resource,
    check_overlap_preservation,
    check_contradictory_states,
)


def make_event(timestamp, room, resident="R1", sensor_id="M001", state="ON"):
    return {
        "timestamp": timestamp,
        "room": room,
        "resident_label": resident,
        "sensor_id": sensor_id,
        "sensor_state": state,
        "activity_label": "Test",
        "event_type": "motion",
    }


def test_teleportation_detected():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    events = [
        make_event(t0, "bedroom1", sensor_id="M005"),
        make_event(t0 + timedelta(seconds=2), "kitchen", sensor_id="M013"),
    ]
    violations = check_teleportation(events)
    assert len(violations) == 1
    assert violations[0]["type"] == "teleportation"


def test_no_teleportation_same_room():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    events = [
        make_event(t0, "kitchen", sensor_id="M013"),
        make_event(t0 + timedelta(seconds=2), "kitchen", sensor_id="M014"),
    ]
    violations = check_teleportation(events)
    assert len(violations) == 0


def test_no_teleportation_enough_time():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    events = [
        make_event(t0, "bedroom1", sensor_id="M005"),
        make_event(t0 + timedelta(seconds=30), "kitchen", sensor_id="M013"),
    ]
    violations = check_teleportation(events)
    assert len(violations) == 0


def test_exclusive_resource_violation():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    window = {
        "resident_a": "R1",
        "resident_b": "R2",
        "events_a_overlap": [make_event(t0, "bathroom", resident="R1", sensor_id="M003")],
        "events_b_overlap": [make_event(t0 + timedelta(seconds=10), "bathroom", resident="R2", sensor_id="M004")],
    }
    violations = check_exclusive_resource(window)
    assert len(violations) == 1
    assert violations[0]["type"] == "exclusive_resource"


def test_no_exclusive_resource_different_times():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    window = {
        "resident_a": "R1",
        "resident_b": "R2",
        "events_a_overlap": [make_event(t0, "bathroom", resident="R1", sensor_id="M003")],
        "events_b_overlap": [make_event(t0 + timedelta(minutes=10), "bathroom", resident="R2", sensor_id="M004")],
    }
    violations = check_exclusive_resource(window)
    assert len(violations) == 0


def test_overlap_preservation_missing_b():
    window = {
        "resident_a": "R1",
        "resident_b": "R2",
        "events_a_overlap": [{"timestamp": datetime(2023,1,1)}],
        "events_b_overlap": [],
    }
    violations = check_overlap_preservation(window)
    assert any(v["type"] == "missing_resident_events" for v in violations)


def test_contradictory_states():
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    events = [
        make_event(t0, "kitchen", sensor_id="M013", state="ON"),
        make_event(t0 + timedelta(milliseconds=100), "kitchen", sensor_id="M013", state="ON"),
    ]
    violations = check_contradictory_states(events)
    assert len(violations) == 1
