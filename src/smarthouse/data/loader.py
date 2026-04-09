"""Cairo dataset loader with synthetic fallback generator."""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from smarthouse.config import Config

logger = logging.getLogger(__name__)

CAIRO_SENSOR_ROOM_MAP: Dict[str, str] = {
    # Motion sensors
    "M001": "hallway", "M002": "hallway",
    "M003": "bathroom", "M004": "bathroom",
    "M005": "bedroom1", "M006": "bedroom1",
    "M007": "bedroom2", "M008": "bedroom2",
    "M009": "office", "M010": "office",
    "M011": "dining", "M012": "dining",
    "M013": "kitchen", "M014": "kitchen",
    "M015": "living", "M016": "living",
    "M017": "laundry", "M018": "laundry",
    # Door sensors
    "D001": "bathroom", "D002": "bedroom1",
    "D003": "bedroom2", "D004": "office",
    "D005": "kitchen",
    # Cabinet sensors
    "C001": "kitchen", "C002": "kitchen",
    "C003": "laundry",
    # Item sensors
    "I001": "kitchen", "I002": "kitchen",
}

ACTIVITY_ROOM_MAP: Dict[str, List[str]] = {
    "Breakfast": ["kitchen", "dining"],
    "Dinner": ["kitchen", "dining"],
    "Work_in_Office": ["office"],
    "Sleep": ["bedroom1", "bedroom2"],
    "Wake": ["bedroom1", "bedroom2", "bathroom"],
    "Bed_to_Toilet": ["bedroom1", "bedroom2", "bathroom", "hallway"],
    "Laundry": ["laundry"],
}

# Room -> sensors mapping (derived from CAIRO_SENSOR_ROOM_MAP)
def _build_room_sensor_map() -> Dict[str, List[str]]:
    room_map: Dict[str, List[str]] = {}
    for sensor, room in CAIRO_SENSOR_ROOM_MAP.items():
        room_map.setdefault(room, []).append(sensor)
    return room_map

ROOM_SENSOR_MAP = _build_room_sensor_map()


def _get_sensor_type(sensor_id: str) -> str:
    if sensor_id.startswith("M"):
        return "motion"
    elif sensor_id.startswith("D"):
        return "door"
    elif sensor_id.startswith("C"):
        return "cabinet"
    elif sensor_id.startswith("I"):
        return "item"
    return "unknown"


def _generate_sensor_events(
    activity: str,
    start_time: datetime,
    end_time: datetime,
    resident_id: str,
    rng: np.random.Generator,
    n_events: int = None,
) -> List[dict]:
    """Generate sensor events for a single activity interval."""
    rooms = ACTIVITY_ROOM_MAP.get(activity, ["living"])
    duration = (end_time - start_time).total_seconds()
    if duration <= 0:
        return []

    if n_events is None:
        n_events = int(rng.integers(5, 21))

    events = []
    for _ in range(n_events):
        room = rooms[rng.integers(0, len(rooms))]
        sensors_in_room = ROOM_SENSOR_MAP.get(room, ["M001"])
        sensor_id = sensors_in_room[rng.integers(0, len(sensors_in_room))]
        sensor_type = _get_sensor_type(sensor_id)

        # State depends on sensor type
        if sensor_type == "motion":
            state = rng.choice(["ON", "OFF"])
        elif sensor_type in ("door", "cabinet"):
            state = rng.choice(["OPEN", "CLOSE"])
        else:
            state = rng.choice(["ON", "OFF"])

        offset = rng.uniform(0, duration)
        ts = start_time + timedelta(seconds=float(offset))

        events.append({
            "timestamp": ts,
            "sensor_id": sensor_id,
            "sensor_state": state,
            "activity_label": activity,
            "resident_label": resident_id,
            "room": room,
            "event_type": sensor_type,
        })

    events.sort(key=lambda e: e["timestamp"])
    return events


def _schedule_day(
    date: datetime,
    resident_id: str,
    offset_minutes: int,
    activities: List[str],
    rng: np.random.Generator,
) -> List[dict]:
    """Schedule activities for one resident for one day. Returns list of interval dicts."""
    intervals = []

    # Base schedule (hour, duration_minutes)
    base_schedule = {
        "Sleep": (22, 480),          # 10pm for 8h
        "Wake": (6, 30),             # 6am for 30min
        "Breakfast": (7, 45),        # 7am for 45min
        "Work_in_Office": (9, 480),  # 9am for 8h
        "Dinner": (18, 60),          # 6pm for 1h
        "Bed_to_Toilet": (3, 15),    # 3am for 15min
    }

    off = offset_minutes
    for activity in activities:
        if activity not in base_schedule:
            continue
        base_hour, base_duration = base_schedule[activity]
        # Add jitter
        jitter_min = int(rng.integers(-15, 16))
        start_hour = base_hour
        start_min = off + jitter_min
        # Handle Sleep wrapping
        if activity == "Sleep":
            start_dt = date + timedelta(hours=start_hour, minutes=start_min)
            # Sleep starts evening of previous day effectively, keep same day
        else:
            start_dt = date + timedelta(hours=start_hour, minutes=start_min)

        duration_jitter = int(rng.integers(-10, 11))
        duration = max(15, base_duration + duration_jitter)
        end_dt = start_dt + timedelta(minutes=duration)

        intervals.append({
            "resident_id": resident_id,
            "activity_label": activity,
            "start_time": start_dt,
            "end_time": end_dt,
        })

    return intervals


def generate_synthetic_data(
    config: Config,
    n_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Generate synthetic Cairo-like dataset."""
    logger.info("Cairo dataset not found, generating synthetic data for demonstration.")

    rng = np.random.default_rng(config.RANDOM_SEED)
    selected = config.SELECTED_ACTIVITIES

    residents = ["R1", "R2"]
    offsets = {"R1": 0, "R2": 20}  # R2 is 20 minutes offset from R1

    base_date = datetime(2023, 1, 1)
    all_intervals = []
    all_events = []
    interval_id_counter = 0

    for day_idx in range(n_days):
        date = base_date + timedelta(days=day_idx)
        for resident in residents:
            day_intervals = _schedule_day(
                date, resident, offsets[resident], selected, rng
            )
            for iv in day_intervals:
                interval_id = f"IV{interval_id_counter:05d}"
                interval_id_counter += 1
                iv["interval_id"] = interval_id
                all_intervals.append(iv)

                events = _generate_sensor_events(
                    iv["activity_label"],
                    iv["start_time"],
                    iv["end_time"],
                    resident,
                    rng,
                )
                all_events.extend(events)

    intervals_df = pd.DataFrame(all_intervals)[
        ["interval_id", "resident_id", "activity_label", "start_time", "end_time"]
    ]
    events_df = pd.DataFrame(all_events)[
        ["timestamp", "sensor_id", "sensor_state", "activity_label",
         "resident_label", "room", "event_type"]
    ]
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    intervals_df = intervals_df.sort_values("start_time").reset_index(drop=True)

    logger.info(
        "Synthetic data: %d events, %d intervals across %d days",
        len(events_df), len(intervals_df), n_days,
    )
    return events_df, intervals_df, CAIRO_SENSOR_ROOM_MAP


def _parse_cairo_events(data_path: Path) -> pd.DataFrame:
    """Try to parse Cairo data.csv format."""
    df = pd.read_csv(data_path, sep=r"\s+", header=None,
                     names=["date", "time", "sensor_id", "sensor_state",
                            "activity_label", "begin_end"],
                     on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"],
                                     format="%Y-%m-%d %H:%M:%S.%f",
                                     errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["room"] = df["sensor_id"].map(CAIRO_SENSOR_ROOM_MAP).fillna("unknown")
    df["event_type"] = df["sensor_id"].apply(_get_sensor_type)
    df["resident_label"] = "R1"  # Cairo is single-resident, label all R1
    return df[["timestamp", "sensor_id", "sensor_state", "activity_label",
               "resident_label", "room", "event_type"]]


def _parse_cairo_activities(act_path: Path) -> pd.DataFrame:
    """Try to parse Cairo activities.csv format."""
    df = pd.read_csv(act_path, sep=r"\s+", header=None,
                     names=["date", "time", "end_date", "end_time",
                            "activity_label"],
                     on_bad_lines="skip")
    df["start_time"] = pd.to_datetime(df["date"] + " " + df["time"],
                                      format="%Y-%m-%d %H:%M:%S",
                                      errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_date"] + " " + df["end_time"],
                                    format="%Y-%m-%d %H:%M:%S",
                                    errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df["resident_id"] = "R1"
    df["interval_id"] = [f"IV{i:05d}" for i in range(len(df))]
    return df[["interval_id", "resident_id", "activity_label", "start_time", "end_time"]]


def load_dataset(
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load Cairo dataset or fall back to synthetic generation."""
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / config.data_dir
    event_path = data_dir / config.event_file
    activity_path = data_dir / config.activity_file

    if event_path.exists() and activity_path.exists():
        logger.info("Loading real Cairo dataset from %s", data_dir)
        try:
            events_df = _parse_cairo_events(event_path)
            intervals_df = _parse_cairo_activities(activity_path)
            # Filter to selected activities
            sel = config.SELECTED_ACTIVITIES
            events_df = events_df[events_df["activity_label"].isin(sel)].copy()
            intervals_df = intervals_df[intervals_df["activity_label"].isin(sel)].copy()
            logger.info(
                "Loaded %d events, %d intervals from Cairo dataset",
                len(events_df), len(intervals_df),
            )
            return events_df, intervals_df, CAIRO_SENSOR_ROOM_MAP
        except Exception as e:
            logger.warning("Failed to parse Cairo dataset (%s), falling back to synthetic", e)

    return generate_synthetic_data(config)
