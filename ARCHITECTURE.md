# Smart-Home Multi-Resident Synthetic Data Pipeline

## Overview
This pipeline generates synthetic multi-resident smart-home activity data based on the Cairo dataset format.

## Stages
1. **prep**: Load/generate Cairo dataset, normalize sensors
2. **overlap**: Extract concurrent activity windows between residents
3. **templates**: Build per-activity event templates
4. **baseline**: Single-agent baseline generation
5. **multiagent**: Multi-agent raw generation
6. **validation**: Rule-based validation and repair
7. **evaluation**: Compute metrics and ablation study
8. **visualization**: Generate timeline plots

## Key Design Decisions
- Falls back to synthetic data if Cairo dataset not present
- All randomness seeded for reproducibility
- Deterministic repair for reproducible outputs
