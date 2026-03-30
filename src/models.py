from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    target_id: str
    project_root: Path
    artifact_root: Path
    isaaclab_root_dir: str
    isaaclab_launcher: str
    train_script: str
    task: str
    headless: bool
    num_envs: int
    max_iterations: int
    trainer_override_mode: str
    trainer_overrides: dict[str, float | int | bool]
    command_override_mode: str
    command_overrides: dict[str, tuple[float, float] | list[float]]
    reward_override_mode: str
    reward_overrides: dict[str, float | int]


@dataclass(slots=True)
class RunResult:
    run_id: str
    target_id: str
    run_dir: Path
    command: list[str]
    effective_input_path: Path
    trainer_override_resolution_path: Path
    trainer_override_verification_path: Path
    command_override_resolution_path: Path
    command_override_verification_path: Path
    reward_override_resolution_path: Path
    reward_override_verification_path: Path
    return_code: int
    config_snapshot_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    summary_path: Path
    summary: dict[str, Any]
