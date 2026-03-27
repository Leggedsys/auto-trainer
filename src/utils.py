from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from models import TrainingConfig


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def build_training_config(raw_config: dict[str, Any]) -> TrainingConfig:
    isaaclab = raw_config["isaaclab"]
    training = raw_config["training"]
    return TrainingConfig(
        isaaclab_root_dir=isaaclab["root_dir"],
        isaaclab_launcher=isaaclab["launcher"],
        train_script=isaaclab["train_script"],
        task=training["task"],
        headless=bool(training.get("headless", True)),
        num_envs=int(training["num_envs"]),
        max_iterations=int(training["max_iterations"]),
    )
