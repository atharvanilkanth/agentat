"""Configuration loader for the smarthouse pipeline."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


@dataclass
class Config:
    dataset_name: str
    data_dir: str
    event_file: str
    activity_file: str
    selected_activities: List[str]
    min_overlap_seconds: float
    min_events: int
    time_bin_seconds: int
    random_seed: int
    outputs_dir: str

    # Convenience aliases
    @property
    def SELECTED_ACTIVITIES(self) -> List[str]:
        return self.selected_activities

    @property
    def MIN_OVERLAP_SECONDS(self) -> float:
        return self.min_overlap_seconds

    @property
    def MIN_EVENTS(self) -> int:
        return self.min_events

    @property
    def RANDOM_SEED(self) -> int:
        return self.random_seed

    @property
    def OUTPUTS_DIR(self) -> str:
        return self.outputs_dir


def load_config(config_path: str = None) -> Config:
    """Load configuration from config.yaml."""
    if config_path is None:
        # Find config relative to repo root
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        dataset_name=raw["dataset"]["name"],
        data_dir=raw["dataset"]["data_dir"],
        event_file=raw["dataset"]["event_file"],
        activity_file=raw["dataset"]["activity_file"],
        selected_activities=raw["activities"]["selected"],
        min_overlap_seconds=raw["overlap"]["min_duration_seconds"],
        min_events=raw["overlap"]["min_events_per_resident"],
        time_bin_seconds=raw["time_bin_seconds"],
        random_seed=raw["random_seed"],
        outputs_dir=raw["outputs"]["dir"],
    )


# Default config instance
_default_config = None


def get_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
