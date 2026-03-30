from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from command_overrides import validate_command_overrides
from models import TrainingConfig
from reward_overrides import validate_reward_overrides
from trainer_overrides import validate_trainer_overrides


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def load_python_yaml(path: Path) -> dict[str, Any]:
    class TupleSafeLoader(yaml.SafeLoader):
        pass

    def construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    def construct_python_slice(loader, node):
        values = loader.construct_sequence(node)
        return slice(*values)

    TupleSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        construct_python_tuple,
    )
    TupleSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        construct_python_slice,
    )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=TupleSafeLoader)
    return data or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def build_training_config(raw_config: dict[str, Any]) -> TrainingConfig:
    target = raw_config["target"]
    isaaclab = raw_config["isaaclab"]
    training = raw_config["training"]
    trainer_override_mode = raw_config.get("trainer_override_mode", "none")
    command_override_mode = raw_config.get("command_override_mode", "none")
    reward_override_mode = raw_config.get("reward_override_mode", "none")
    trainer_overrides = raw_config.get("trainer_overrides", {})
    if not isinstance(trainer_overrides, dict):
        raise ValueError("trainer_overrides must be a mapping if provided")
    validate_trainer_overrides(trainer_overrides)
    return TrainingConfig(
        target_id=target["id"],
        project_root=Path(target["project_root"]),
        artifact_root=Path(target["artifact_root"]),
        isaaclab_root_dir=isaaclab["root_dir"],
        isaaclab_launcher=isaaclab["launcher"],
        train_script=isaaclab["train_script"],
        task=training["task"],
        headless=bool(training.get("headless", True)),
        num_envs=int(training["num_envs"]),
        max_iterations=int(training["max_iterations"]),
        trainer_override_mode=str(trainer_override_mode),
        trainer_overrides=_build_trainer_overrides_from_training(training),
        command_override_mode=str(command_override_mode),
        command_overrides=_build_command_overrides_from_config(raw_config),
        reward_override_mode=str(reward_override_mode),
        reward_overrides=_build_reward_overrides_from_config(raw_config),
    )


def _build_trainer_overrides_from_training(
    training: dict[str, Any],
) -> dict[str, float | int | bool]:
    overrides = {}
    for key in (
        "learning_rate",
        "entropy_coef",
        "clip_param",
        "num_mini_batches",
    ):
        if key in training:
            overrides[key] = training[key]
    validate_trainer_overrides(overrides)
    return overrides


def _build_command_overrides_from_config(
    raw_config: dict[str, Any],
) -> dict[str, tuple[float, float] | list[float]]:
    command_config = raw_config.get("command", {})
    if not isinstance(command_config, dict):
        raise ValueError("command must be a mapping if provided")
    overrides = {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        if key in command_config:
            overrides[key] = command_config[key]
    validate_command_overrides(overrides)
    return overrides


def _build_reward_overrides_from_config(
    raw_config: dict[str, Any],
) -> dict[str, float | int]:
    reward_config = raw_config.get("reward", {})
    if not isinstance(reward_config, dict):
        raise ValueError("reward must be a mapping if provided")
    overrides = {}
    for key in (
        "track_lin_vel_xy_exp.weight",
        "track_ang_vel_z_exp.weight",
        "feet_air_time.weight",
        "flat_orientation_l2.weight",
    ):
        if key in reward_config:
            overrides[key] = reward_config[key]
    validate_reward_overrides(overrides)
    return overrides
