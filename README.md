# agentat — Multi-Resident Smart-Home Synthetic Data Pipeline

A reproducible end-to-end prototype that proves:

> **Generating one synthetic sensor stream per resident and merging them with coordination rules produces more realistic multi-resident smart-home data than a single-resident approximation.**

Built on the [Cairo CASAS dataset](http://casas.wsu.edu/) format. Falls back to a fully-synthetic Cairo-like dataset when real data is not present.

---

## Table of Contents

- [Setup](#setup)
- [Cairo input format](#cairo-input-format)
- [Running the pipeline](#running-the-pipeline)
- [Stage-by-stage commands](#stage-by-stage-commands)
- [Outputs produced](#outputs-produced)
- [Ablation study](#ablation-study)
- [Optional classifier experiment](#optional-classifier-experiment)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/atharvanilkanth/agentat.git
cd agentat

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

Python ≥ 3.9 is required. All dependencies are listed in `requirements.txt`.

---

## Cairo input format

The pipeline accepts the [CASAS Cairo](http://casas.wsu.edu/datasets/) multi-resident dataset.  
Place the files under `data/cairo/`:

```
data/cairo/
├── data.csv        # sensor events
└── activities.csv  # activity label annotations
```

### `data.csv` — sensor events

| Column | Example | Notes |
|--------|---------|-------|
| date | `2011-04-06` | |
| time | `13:41:16.000000` | |
| sensor_id | `M019` | motion / door / cabinet / item |
| sensor_state | `ON` / `OPEN` | |
| activity_label *(optional)* | `Laundry` | may be blank |
| resident *(optional)* | `R1` | may be blank |

### `activities.csv` — annotated intervals

| Column | Example |
|--------|---------|
| resident_id | `R1` |
| activity_label | `Dinner` |
| start_time | `2011-04-06 19:00:00` |
| end_time | `2011-04-06 19:34:00` |

**If these files are absent**, the pipeline automatically generates a synthetic Cairo-like dataset (2 residents, 30 days) and logs a notice. You can run the full pipeline with no real data at all.

---

## Running the pipeline

```bash
# Full end-to-end run
python scripts/run_all.py run-all

# Or via Make
make run
```

This executes all 8 stages and writes every artifact to `outputs/`.

---

## Stage-by-stage commands

```bash
python scripts/run_all.py prep           # Stage 1 – data preparation
python scripts/run_all.py overlap        # Stage 2 – overlap extraction
python scripts/run_all.py templates      # Stage 3 – activity templates
python scripts/run_all.py baseline       # Stage 4 – single-agent baseline
python scripts/run_all.py multiagent     # Stage 5 – multi-agent generation
python scripts/run_all.py validation     # Stage 6 – rules + repair
python scripts/run_all.py evaluation     # Stage 7 – metrics + ablation
python scripts/run_all.py visualization  # Stage 8 – timeline plots
```

Each stage loads its inputs from `outputs/` (produced by earlier stages) and writes its own outputs there, so stages can be re-run independently.

---

## Outputs produced

All artifacts are written to the `outputs/` directory.

| File | Description | Format |
|------|-------------|--------|
| `sensor_room_map.json` | Sensor → room mapping | JSON |
| `activity_intervals.parquet` | Resident activity intervals | Parquet |
| `events.parquet` | Unified sensor event table | Parquet |
| `real_overlap_windows.pkl` | Real concurrent-activity windows (gold set) | Pickle |
| `activity_templates.pkl` | Per-activity event templates | Pickle |
| `baseline_windows.pkl` | Single-agent synthetic windows | Pickle |
| `multiagent_windows_raw.pkl` | Multi-agent windows before repair | Pickle |
| `multiagent_windows_validated.pkl` | Multi-agent windows after repair | Pickle |
| `evaluation_metrics.json` | All metrics across variants A–D | JSON |
| `ablation_results.csv` | Ablation comparison table | CSV |
| `qualitative_examples/` | Timeline PNGs (real / baseline / proposed) | PNG |

### Unified event table schema

```
timestamp, sensor_id, sensor_state, activity_label, resident_label, room, event_type
```

### Activity intervals schema

```
interval_id, resident_id, activity_label, start_time, end_time
```

---

## Ablation study

Run the four variants (A–D) and produce a comparison table + plots:

```bash
python scripts/ablation.py
```

Or equivalently, the `evaluation` stage within `run-all` executes the ablation automatically.

| Variant | Description |
|---------|-------------|
| **A** – Real | Measured on real overlap windows |
| **B** – Baseline | Single-agent synthetic (drop one resident) |
| **C** – Multi-agent raw | Both residents merged, no repair |
| **D** – Multi-agent validated | Both residents merged + validation + repair |

Metrics reported:

1. Concurrent room activation match
2. Joint room-activation distribution JSD vs real
3. Activity overlap preservation rate
4. Constraint violation rates (teleportation, room mismatch, exclusive resource, contradictory state)
5. Diversity scores (activity-pair entropy, room-pair entropy)

---

## Optional classifier experiment

Included in the `evaluation` stage. Trains a Random Forest on synthetic windows and tests on real:

- **Baseline-trained** classifier: trained on variant B, tested on real windows
- **Validated-trained** classifier: trained on variant D, tested on real windows

Results appear in `outputs/evaluation_metrics.json` under the `classifier` key.

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a full data-flow diagram and rule-system description.

```
config/config.yaml
    │
    ▼
data/loader.py ──► events.parquet, activity_intervals.parquet, sensor_room_map.json
    │
    ▼
overlap/extractor.py ──► real_overlap_windows.pkl
    │
    ▼
templates/builder.py ──► activity_templates.pkl
    │
    ├──► generators/baseline.py  ──► baseline_windows.pkl
    │
    └──► generators/multiagent.py ──► multiagent_windows_raw.pkl
              │
              ▼
         validation/rules.py + repair.py ──► multiagent_windows_validated.pkl
              │
              ▼
         evaluation/metrics.py  ──► evaluation_metrics.json, ablation_results.csv
         visualization/plots.py ──► qualitative_examples/
```

---

## Running tests

```bash
pytest tests/ -v
```

16 tests cover: overlap logic, validation rules, and a full-pipeline smoke test.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: smarthouse` | Run `pip install -e .` from repo root |
| Pipeline fails on `evaluation` with empty windows | Ensure `validation` stage ran first; or run `run-all` |
| Real Cairo data not loading | Check column names match expected format; see loader log output for diagnostics |
| Plots not rendered (headless server) | Set `MPLBACKEND=Agg` before running, or `export MPLBACKEND=Agg` |
