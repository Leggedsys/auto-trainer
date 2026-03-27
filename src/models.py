from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    isaaclab_root_dir: str
    isaaclab_launcher: str
    train_script: str
    task: str
    headless: bool
    num_envs: int
    max_iterations: int


@dataclass(slots=True)
class RunResult:
    run_id: str
    run_dir: Path
    command: list[str]
    return_code: int
    config_snapshot_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    summary_path: Path
    summary: dict[str, Any]
