"""Evaluation metrics for synthetic data quality."""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from smarthouse.config import Config
from smarthouse.validation.rules import validate_window

logger = logging.getLogger(__name__)


def _get_active_rooms(windows: List[Dict[str, Any]]) -> List[str]:
    """Get all rooms activated in a set of windows."""
    rooms = []
    for w in windows:
        rooms.extend(w.get("rooms_a", []))
        rooms.extend(w.get("rooms_b", []))
    return rooms


def compute_concurrent_room_activation(
    windows: List[Dict[str, Any]],
    real_windows: List[Dict[str, Any]],
) -> float:
    """
    Compute the fraction of windows where both residents activate distinct rooms
    simultaneously (proxy for realism of concurrent activity).
    """
    if not windows:
        return 0.0

    count = 0
    for w in windows:
        rooms_a = set(w.get("rooms_a", []))
        rooms_b = set(w.get("rooms_b", []))
        if rooms_a and rooms_b and rooms_a != rooms_b:
            count += 1

    return count / len(windows)


def _room_distribution(windows: List[Dict[str, Any]], time_bin_seconds: int) -> np.ndarray:
    """Build a normalized room co-occurrence distribution."""
    from smarthouse.data.loader import CAIRO_SENSOR_ROOM_MAP
    all_rooms = sorted(set(CAIRO_SENSOR_ROOM_MAP.values()))
    room_idx = {r: i for i, r in enumerate(all_rooms)}
    n = len(all_rooms)
    matrix = np.zeros((n, n)) + 1e-10  # smoothing

    for w in windows:
        rooms_a = w.get("rooms_a", [])
        rooms_b = w.get("rooms_b", [])
        for ra in rooms_a:
            for rb in rooms_b:
                if ra in room_idx and rb in room_idx:
                    matrix[room_idx[ra], room_idx[rb]] += 1

    flat = matrix.flatten()
    flat = flat / flat.sum()
    return flat


def compute_joint_room_distribution_jsd(
    windows: List[Dict[str, Any]],
    real_windows: List[Dict[str, Any]],
    time_bin_seconds: int,
) -> float:
    """Compute Jensen-Shannon divergence between joint room distributions."""
    from scipy.spatial.distance import jensenshannon

    if not windows or not real_windows:
        return 1.0

    p = _room_distribution(real_windows, time_bin_seconds)
    q = _room_distribution(windows, time_bin_seconds)
    return float(jensenshannon(p, q))


def compute_overlap_preservation_rate(windows: List[Dict[str, Any]]) -> float:
    """Fraction of windows where both residents have events in the overlap."""
    if not windows:
        return 0.0
    count = sum(
        1 for w in windows
        if w.get("events_a_overlap") and w.get("events_b_overlap")
    )
    return count / len(windows)


def compute_violation_rate(windows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute per-violation-type rates."""
    if not windows:
        return {}

    type_counts: Dict[str, int] = {}
    for w in windows:
        violations = w.get("violations") or validate_window(w)
        for v in violations:
            vtype = v.get("type", "unknown")
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

    total = len(windows)
    return {vtype: count / total for vtype, count in type_counts.items()}


def compute_diversity_score(windows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute diversity metrics over the window set."""
    if not windows:
        return {"activity_pair_entropy": 0.0, "room_pair_entropy": 0.0}

    # Activity pair diversity
    pair_counts: Dict[str, int] = {}
    for w in windows:
        pair = f"{w.get('activity_a','?')}+{w.get('activity_b','?')}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    total = len(windows)
    pair_probs = np.array([c / total for c in pair_counts.values()])
    act_entropy = float(-np.sum(pair_probs * np.log(pair_probs + 1e-10)))

    # Room pair diversity
    room_pair_counts: Dict[str, int] = {}
    for w in windows:
        for ra in w.get("rooms_a", []):
            for rb in w.get("rooms_b", []):
                rp = f"{ra}+{rb}"
                room_pair_counts[rp] = room_pair_counts.get(rp, 0) + 1
    if not room_pair_counts:
        room_entropy = 0.0
    else:
        room_total = sum(room_pair_counts.values())
        room_probs = np.array([c / room_total for c in room_pair_counts.values()])
        room_entropy = float(max(0.0, -np.sum(room_probs * np.log(room_probs + 1e-10))))

    return {
        "activity_pair_entropy": act_entropy,
        "room_pair_entropy": room_entropy,
    }


def compute_all_metrics(
    variant_name: str,
    windows: List[Dict[str, Any]],
    real_windows: List[Dict[str, Any]],
    config: Config,
) -> Dict[str, Any]:
    """Compute all metrics for a variant."""
    metrics = {
        "variant": variant_name,
        "n_windows": len(windows),
        "concurrent_room_activation": compute_concurrent_room_activation(windows, real_windows),
        "jsd_room_distribution": compute_joint_room_distribution_jsd(
            windows, real_windows, config.time_bin_seconds
        ),
        "overlap_preservation_rate": compute_overlap_preservation_rate(windows),
        "violation_rates": compute_violation_rate(windows),
        "diversity": compute_diversity_score(windows),
    }
    logger.info("Metrics for %s: %s", variant_name, metrics)
    return metrics


def run_ablation(
    real_windows: List[Dict[str, Any]],
    baseline_windows: List[Dict[str, Any]],
    raw_windows: List[Dict[str, Any]],
    validated_windows: List[Dict[str, Any]],
    config: Config,
) -> pd.DataFrame:
    """Run ablation study across A/B/C/D variants."""
    variants = {
        "A_real": real_windows,
        "B_baseline": baseline_windows,
        "C_multiagent_raw": raw_windows,
        "D_multiagent_validated": validated_windows,
    }

    rows = []
    for name, windows in variants.items():
        metrics = compute_all_metrics(name, windows, real_windows, config)
        row = {
            "variant": name,
            "n_windows": metrics["n_windows"],
            "concurrent_room_activation": metrics["concurrent_room_activation"],
            "jsd_room_distribution": metrics["jsd_room_distribution"],
            "overlap_preservation_rate": metrics["overlap_preservation_rate"],
            "activity_pair_entropy": metrics["diversity"]["activity_pair_entropy"],
            "room_pair_entropy": metrics["diversity"]["room_pair_entropy"],
        }
        # Add violation rates
        for vtype, rate in metrics.get("violation_rates", {}).items():
            row[f"violation_{vtype}"] = rate
        rows.append(row)

    return pd.DataFrame(rows)
