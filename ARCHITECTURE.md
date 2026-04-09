# Smart-Home Multi-Resident Synthetic Data Pipeline — Architecture

## Overview

This pipeline generates synthetic multi-resident smart-home sensor data by:

1. **Generating one synthetic sensor stream per resident** using activity templates sampled from real (or synthetic) data
2. **Merging the two streams** with a shared household clock
3. **Applying coordination rules** to detect physically-impossible or semantically-implausible events
4. **Repairing violations deterministically** to yield clean, realistic multi-resident windows

The goal is to prove that multi-agent generation + coordination rules produces more realistic data than a single-resident approximation (the baseline).

---

## Data Flow

```
Cairo dataset (data/cairo/)
    OR synthetic fallback
          │
          ▼
  ┌─────────────────────────────┐
  │  Stage 1 – Data Preparation │  (src/smarthouse/data/)
  │  loader.py + normalizer.py  │
  └──────────────┬──────────────┘
                 │ events.parquet
                 │ activity_intervals.parquet
                 │ sensor_room_map.json
                 ▼
  ┌──────────────────────────────┐
  │  Stage 2 – Overlap Extraction│  (src/smarthouse/overlap/)
  │  extractor.py                │
  └──────────────┬───────────────┘
                 │ real_overlap_windows.pkl
                 ▼
  ┌──────────────────────────────┐
  │  Stage 3 – Template Building │  (src/smarthouse/templates/)
  │  builder.py                  │
  └──────────────┬───────────────┘
                 │ activity_templates.pkl
          ┌──────┴──────┐
          ▼             ▼
  ┌───────────┐   ┌──────────────────┐
  │ Stage 4   │   │  Stage 5         │
  │ Baseline  │   │  Multi-Agent     │
  │ Generator │   │  Generator       │
  └─────┬─────┘   └────────┬─────────┘
        │                  │
   baseline_windows.pkl    │ multiagent_windows_raw.pkl
                           ▼
                  ┌─────────────────────┐
                  │  Stage 6 – Validation│  (src/smarthouse/validation/)
                  │  rules.py + repair.py│
                  └──────────┬──────────┘
                             │ multiagent_windows_validated.pkl
                             ▼
              ┌──────────────────────────────┐
              │  Stage 7 – Evaluation        │  (src/smarthouse/evaluation/)
              │  metrics.py + classifier.py  │
              └──────────────┬───────────────┘
                             │ evaluation_metrics.json
                             │ ablation_results.csv
                             ▼
              ┌──────────────────────────────┐
              │  Stage 8 – Visualization     │  (src/smarthouse/visualization/)
              │  plots.py                    │
              └──────────────┬───────────────┘
                             │ qualitative_examples/*.png
```

---

## Key Data Objects

### Overlap Window

The central data structure passed between stages:

```python
{
  "window_id": str,           # unique identifier
  "resident_a": str,          # e.g. "R1"
  "activity_a": str,          # e.g. "Dinner"
  "resident_b": str,          # e.g. "R2"
  "activity_b": str,          # e.g. "Work_in_Office"
  "overlap_start": datetime,
  "overlap_end": datetime,
  "overlap_duration_seconds": float,
  "events_a": List[dict],     # full interval events for resident A
  "events_b": List[dict],     # full interval events for resident B
  "events_a_overlap": List[dict],  # truncated to overlap period
  "events_b_overlap": List[dict],
  "merged_events": List[dict],     # sorted by timestamp, tie-break by resident
  "rooms_a": List[str],
  "rooms_b": List[str],
  "interval_id_a": str,
  "interval_id_b": str,
  "violations": List[dict],   # populated by rules.py
}
```

### Sensor Event

```python
{
  "timestamp": datetime,
  "sensor_id": str,
  "sensor_state": str,        # normalized: ON/OFF/OPEN/CLOSE
  "activity_label": str,
  "resident_label": str,
  "room": str,
  "event_type": str,          # motion/door/cabinet/item
}
```

---

## Coordination Rule System

### Category 1 — Same-Resident Consistency

**Teleportation check**: If the same resident triggers events in two spatially-distant rooms within less than `min_travel_seconds` (default 5 s), flag a `teleportation` violation.

**Contradictory-state check**: If the same sensor emits the same state twice in a row (e.g. `ON → ON`) with no intervening opposite state, flag a `contradictory_state` violation.

### Category 2 — Household Co-Occupancy Realism

Two residents may occupy different rooms simultaneously. Violations occur only when contradictory household-level states are impossible.

### Category 3 — Exclusive Resource Use

**Bathroom exclusivity**: If both residents have events assigned to the bathroom room within an overlapping time window, flag an `exclusive_resource` violation.

### Category 4 — Activity-Room Alignment

Each activity has an expected room set:

| Activity | Expected rooms |
|----------|---------------|
| Breakfast / Dinner | kitchen, dining |
| Work_in_Office | office |
| Sleep | bedroom1, bedroom2 |
| Wake | bedroom1, bedroom2, bathroom |
| Bed_to_Toilet | bedroom1, bedroom2, bathroom, hallway |
| Laundry | laundry |

If a generated resident stream has no events in the expected rooms, flag an `activity_room_mismatch` violation.

### Category 5 — Overlap Preservation

Both activities must remain "visible" in the merged window. If one resident has zero events in the overlap period, flag a `missing_resident_events` violation.

---

## Repair Strategies

Applied in order, up to `max_attempts = 3` iterations:

1. **Teleportation repair** — shift the offending event timestamp by a small random delta (5–30 s) to create a plausible travel time.
2. **Exclusive-resource repair** — delay the second resident's conflicting events by the duration needed to avoid overlap.
3. **Contradictory-state repair** — drop the duplicate state event.
4. **Activity-collapse repair** — if a resident's activity becomes invisible, resample key evidence events from the template and insert them into the overlap window.

After each repair pass all rules are re-evaluated. If violations remain after `max_attempts` the window is regenerated from scratch.

---

## Ablation Variants

| Variant | Method | Expected behaviour |
|---------|--------|--------------------|
| **A** Real | Ground truth | Best concurrent room activation |
| **B** Baseline | Drop one resident, single-agent synthesis | Zero concurrent room activation |
| **C** Multi-agent raw | Both residents, no repair | Concurrent activation restored; some violations |
| **D** Multi-agent validated | Both residents + repair | Best violation rate; concurrent activation retained |

---

## Module Reference

| Module | Responsibility |
|--------|---------------|
| `smarthouse/config.py` | Load `config/config.yaml`; expose typed settings |
| `smarthouse/data/loader.py` | Cairo loader + synthetic fallback; unified event table |
| `smarthouse/data/normalizer.py` | Sensor-state normalization |
| `smarthouse/overlap/extractor.py` | Overlap detection, quality filtering, gold-set construction |
| `smarthouse/templates/builder.py` | Per-activity template building and sampling |
| `smarthouse/generators/baseline.py` | Single-agent baseline windows |
| `smarthouse/generators/multiagent.py` | Multi-agent raw windows |
| `smarthouse/validation/rules.py` | Rule checks, violation tagging |
| `smarthouse/validation/repair.py` | Deterministic repair + revalidation loop |
| `smarthouse/evaluation/metrics.py` | All 5 metric families + ablation table |
| `smarthouse/evaluation/classifier.py` | Transfer-learning classifier experiment |
| `smarthouse/visualization/plots.py` | Timeline plots, comparison plots, before/after repair |
| `smarthouse/pipeline.py` | Stage orchestration |
| `scripts/run_all.py` | CLI entrypoint |
| `scripts/ablation.py` | Standalone ablation runner |

