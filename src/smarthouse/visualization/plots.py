"""Timeline plots for overlap windows."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

ROOM_COLORS = {
    "bedroom1": "#4e79a7",
    "bedroom2": "#59a14f",
    "bathroom": "#f28e2b",
    "kitchen": "#e15759",
    "dining": "#76b7b2",
    "office": "#edc948",
    "living": "#b07aa1",
    "hallway": "#ff9da7",
    "laundry": "#9c755f",
    "unknown": "#bab0ac",
}


def _plot_window_timeline(
    ax: plt.Axes,
    events: List[dict],
    title: str,
    overlap_start=None,
    overlap_end=None,
) -> None:
    """Plot event timeline on the given axes."""
    if not events:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No events", transform=ax.transAxes, ha="center")
        return

    timestamps = [e["timestamp"] for e in events]
    rooms = [e.get("room", "unknown") for e in events]
    colors = [ROOM_COLORS.get(r, "#bab0ac") for r in rooms]

    t_min = min(timestamps)
    rel_seconds = [(t - t_min).total_seconds() for t in timestamps]

    ax.scatter(rel_seconds, range(len(events)), c=colors, s=30, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Seconds")
    ax.set_yticks([])

    if overlap_start and overlap_end:
        os_rel = (overlap_start - t_min).total_seconds()
        oe_rel = (overlap_end - t_min).total_seconds()
        ax.axvspan(os_rel, oe_rel, alpha=0.15, color="green", label="overlap")

    # Legend
    seen_rooms = list(dict.fromkeys(rooms))
    patches = [
        mpatches.Patch(color=ROOM_COLORS.get(r, "#bab0ac"), label=r)
        for r in seen_rooms
    ]
    ax.legend(handles=patches, fontsize=6, loc="upper right")


def plot_timeline(
    window: Dict[str, Any],
    title: str,
    output_path: str,
) -> None:
    """Plot a single window's timeline and save to file."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    events_a = window.get("events_a_overlap", [])
    events_b = window.get("events_b_overlap", [])

    _plot_window_timeline(
        axes[0], events_a,
        f"Resident A ({window.get('activity_a', '?')})",
        window.get("overlap_start"), window.get("overlap_end"),
    )
    _plot_window_timeline(
        axes[1], events_b,
        f"Resident B ({window.get('activity_b', '?')})",
        window.get("overlap_start"), window.get("overlap_end"),
    )

    fig.suptitle(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved timeline to %s", output_path)


def plot_comparison(
    real_w: Dict[str, Any],
    baseline_w: Optional[Dict[str, Any]],
    proposed_w: Optional[Dict[str, Any]],
    output_path: str,
) -> None:
    """3-panel comparison: real vs baseline vs proposed."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    windows = [
        (real_w, "Real"),
        (baseline_w, "Baseline"),
        (proposed_w, "Proposed (Multi-Agent)"),
    ]

    for ax, (w, label) in zip(axes, windows):
        if w is None:
            ax.set_title(f"{label}\n(N/A)")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            continue
        events = (w.get("events_a_overlap", []) or []) + (w.get("events_b_overlap", []) or [])
        events = sorted(events, key=lambda e: e["timestamp"])
        if events:
            t_min = min(e["timestamp"] for e in events)
            rel = [(e["timestamp"] - t_min).total_seconds() for e in events]
            rooms = [e.get("room", "unknown") for e in events]
            colors = [ROOM_COLORS.get(r, "#bab0ac") for r in rooms]
            ax.scatter(rel, range(len(events)), c=colors, s=20)
        ax.set_title(label)
        ax.set_xlabel("Seconds")
        ax.set_yticks([])

    fig.suptitle("Window Comparison")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_repair(
    raw_w: Dict[str, Any],
    repaired_w: Dict[str, Any],
    violations: List[dict],
    output_path: str,
) -> None:
    """Before/after repair plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (w, label) in zip(axes, [(raw_w, "Before Repair"), (repaired_w, "After Repair")]):
        events = w.get("merged_events", [])
        if events:
            t_min = min(e["timestamp"] for e in events)
            rel = [(e["timestamp"] - t_min).total_seconds() for e in events]
            rooms = [e.get("room", "unknown") for e in events]
            colors = [ROOM_COLORS.get(r, "#bab0ac") for r in rooms]
            ax.scatter(rel, range(len(events)), c=colors, s=20)
        vcount = len(violations) if label == "Before Repair" else len(w.get("violations", []))
        ax.set_title(f"{label}\n({vcount} violations)")
        ax.set_xlabel("Seconds")
        ax.set_yticks([])

    fig.suptitle("Repair Comparison")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_qualitative_examples(
    real_windows: List[Dict[str, Any]],
    baseline_windows: List[Dict[str, Any]],
    validated_windows: List[Dict[str, Any]],
    outputs_dir: str,
    n: int = 5,
) -> None:
    """Generate n qualitative example figures."""
    out_dir = Path(outputs_dir) / "qualitative_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_real = min(n, len(real_windows))
    n_base = min(n, len(baseline_windows))
    n_val = min(n, len(validated_windows))

    for i in range(n_real):
        plot_timeline(
            real_windows[i],
            f"Real Window {i+1}",
            str(out_dir / f"real_window_{i+1:03d}.png"),
        )

    for i in range(max(n_real, n_base)):
        rw = real_windows[i] if i < n_real else None
        bw = baseline_windows[i] if i < n_base else None
        vw = validated_windows[i] if i < n_val else None
        if rw or bw or vw:
            plot_comparison(
                rw or {},
                bw,
                vw,
                str(out_dir / f"comparison_{i+1:03d}.png"),
            )

    logger.info("Saved qualitative examples to %s", out_dir)
