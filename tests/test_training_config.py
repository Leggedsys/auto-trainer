from __future__ import annotations

from pathlib import Path

import pytest

from utils import build_training_config


def test_build_training_config_copies_trainer_overrides() -> None:
    raw_config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": "/tmp/project",
            "artifact_root": "/tmp/artifacts",
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "headless": True,
            "num_envs": 256,
            "max_iterations": 50,
            "learning_rate": 0.0003,
            "entropy_coef": 0.01,
            "clip_param": 0.2,
            "num_mini_batches": 4,
        },
        "trainer_override_mode": "none",
        "trainer_overrides": {},
        "command_override_mode": "hydra",
        "command": {
            "lin_vel_x": [0.0, 1.0],
            "ang_vel_z": [-1.0, 1.0],
        },
        "reward_override_mode": "hydra",
        "reward": {
            "track_lin_vel_xy_exp.weight": 1.5,
            "feet_air_time.weight": 0.25,
        },
    }

    config = build_training_config(raw_config)

    assert config.project_root == Path("/tmp/project")
    assert config.trainer_override_mode == "none"
    assert config.trainer_overrides == {
        "learning_rate": 0.0003,
        "entropy_coef": 0.01,
        "clip_param": 0.2,
        "num_mini_batches": 4,
    }
    assert config.command_override_mode == "hydra"
    assert config.command_overrides == {
        "lin_vel_x": [0.0, 1.0],
        "ang_vel_z": [-1.0, 1.0],
    }
    assert config.reward_override_mode == "hydra"
    assert config.reward_overrides == {
        "track_lin_vel_xy_exp.weight": 1.5,
        "feet_air_time.weight": 0.25,
    }


def test_build_training_config_rejects_non_mapping_trainer_overrides() -> None:
    raw_config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": "/tmp/project",
            "artifact_root": "/tmp/artifacts",
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "headless": True,
            "num_envs": 256,
            "max_iterations": 50,
        },
        "trainer_override_mode": "none",
        "trainer_overrides": ["bad"],
    }

    with pytest.raises(ValueError, match="trainer_overrides must be a mapping"):
        build_training_config(raw_config)


def test_build_training_config_rejects_out_of_bounds_trainer_override() -> None:
    raw_config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": "/tmp/project",
            "artifact_root": "/tmp/artifacts",
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "headless": True,
            "num_envs": 256,
            "max_iterations": 50,
            "learning_rate": 1.0,
        },
        "trainer_override_mode": "hydra",
        "trainer_overrides": {},
    }

    with pytest.raises(
        ValueError, match="Trainer override learning_rate out of bounds"
    ):
        build_training_config(raw_config)


def test_build_training_config_rejects_invalid_command_override_range() -> None:
    raw_config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": "/tmp/project",
            "artifact_root": "/tmp/artifacts",
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "headless": True,
            "num_envs": 256,
            "max_iterations": 50,
            "learning_rate": 0.0003,
        },
        "trainer_override_mode": "hydra",
        "trainer_overrides": {},
        "command_override_mode": "hydra",
        "command": {"lin_vel_x": [1.0, 0.0]},
    }

    with pytest.raises(ValueError, match="must satisfy low <= high"):
        build_training_config(raw_config)


def test_build_training_config_rejects_invalid_reward_override() -> None:
    raw_config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": "/tmp/project",
            "artifact_root": "/tmp/artifacts",
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "headless": True,
            "num_envs": 256,
            "max_iterations": 50,
            "learning_rate": 0.0003,
        },
        "trainer_override_mode": "hydra",
        "trainer_overrides": {},
        "command_override_mode": "hydra",
        "command": {"lin_vel_x": [0.0, 1.0]},
        "reward_override_mode": "hydra",
        "reward": {"feet_air_time.weight": 5.0},
    }

    with pytest.raises(
        ValueError, match="Reward override feet_air_time.weight out of bounds"
    ):
        build_training_config(raw_config)
