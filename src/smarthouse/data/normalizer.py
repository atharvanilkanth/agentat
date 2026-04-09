"""Sensor state normalization."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_state(sensor_id: str, state: str) -> str:
    """Normalize sensor state to canonical form."""
    if not isinstance(state, str):
        return "UNKNOWN"

    state_upper = state.strip().upper()

    if sensor_id.startswith("M"):
        # Motion sensor
        if state_upper in ("ON", "1", "TRUE", "ACTIVE"):
            return "ON"
        elif state_upper in ("OFF", "0", "FALSE", "INACTIVE"):
            return "OFF"
        return state_upper

    elif sensor_id.startswith("D") or sensor_id.startswith("C"):
        # Door/cabinet sensor
        if state_upper in ("OPEN", "1", "TRUE", "ON"):
            return "OPEN"
        elif state_upper in ("CLOSE", "CLOSED", "0", "FALSE", "OFF"):
            return "CLOSE"
        return state_upper

    elif sensor_id.startswith("I"):
        # Item sensor
        if state_upper in ("ON", "1", "TRUE"):
            return "ON"
        elif state_upper in ("OFF", "0", "FALSE"):
            return "OFF"
        return state_upper

    return state_upper


def normalize_events_df(df):
    """Normalize sensor_state column in events DataFrame."""
    import pandas as pd
    df = df.copy()
    df["sensor_state"] = df.apply(
        lambda row: normalize_state(row["sensor_id"], row["sensor_state"]),
        axis=1,
    )
    return df
