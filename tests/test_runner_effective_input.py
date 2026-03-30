from __future__ import annotations

from pathlib import Path

from runner import TrainingRunner
from utils import load_yaml


def build_config(*, artifact_root: Path):
    from models import TrainingConfig

    return TrainingConfig(
        target_id="test_target",
        project_root=Path("/tmp/project-root"),
        artifact_root=artifact_root,
        isaaclab_root_dir="/tmp/isaaclab",
        isaaclab_launcher="/tmp/isaaclab/isaaclab.sh",
        train_script="/tmp/isaaclab/train.py",
        task="Isaac-Velocity-Flat-Unitree-Go2-v0",
        headless=True,
        num_envs=256,
        max_iterations=50,
        trainer_override_mode="none",
        trainer_overrides={
            "learning_rate": 0.0003,
            "entropy_coef": 0.01,
        },
        command_override_mode="none",
        command_overrides={},
        reward_override_mode="none",
        reward_overrides={},
    )


def test_runner_writes_effective_input_artifact(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "train_go2.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    captured = {}

    def fake_generate_run_id() -> str:
        return "run_001"

    class CompletedProcess:
        returncode = 0

    def fake_subprocess_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        stdout_handle = kwargs["stdout"]
        stdout_handle.write(
            "Mean reward: 1.5\n"
            "Mean episode length: 10\n"
            "Mean value_function loss: 0.1\n"
            "Mean surrogate loss: 0.2\n"
            "Mean entropy loss: 0.3\n"
            "Learning iteration 5/50\n"
        )
        return CompletedProcess()

    monkeypatch.setattr("runner.generate_run_id", fake_generate_run_id)
    monkeypatch.setattr("runner.subprocess.run", fake_subprocess_run)

    runner = TrainingRunner(
        project_root=project_root, artifact_root=tmp_path / "artifacts"
    )
    result = runner.run(build_config(artifact_root=tmp_path / "artifacts"))

    effective_input = load_yaml(result.effective_input_path)
    trainer_override_resolution = load_yaml(result.trainer_override_resolution_path)
    trainer_override_verification = load_yaml(result.trainer_override_verification_path)

    assert result.effective_input_path.name == "effective_input.yaml"
    assert (
        result.trainer_override_resolution_path.name
        == "trainer_override_resolution.yaml"
    )
    assert effective_input["command"] == captured["command"]
    assert effective_input["cwd"] == str(project_root)
    assert effective_input["training"] == {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "headless": True,
        "num_envs": 256,
        "max_iterations": 50,
    }
    assert effective_input["trainer_override_mode"] == "none"
    assert effective_input["trainer_overrides"] == {
        "learning_rate": 0.0003,
        "entropy_coef": 0.01,
    }
    assert trainer_override_resolution["mode"] == "none"
    assert trainer_override_resolution["effective"] is False
    assert trainer_override_verification["effective"] is False
    assert effective_input["trainer_overrides_effective"] is False
    assert effective_input["command_override_mode"] == "none"
    assert effective_input["command_overrides"] == {}
    assert effective_input["reward_override_mode"] == "none"
    assert effective_input["reward_overrides"] == {}
    assert captured["env"]["RUN_DIR"] == str(result.run_dir)


def test_runner_appends_hydra_trainer_override_args(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "repo"
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "train_go2.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    captured = {}

    def fake_generate_run_id() -> str:
        return "run_002"

    class CompletedProcess:
        returncode = 0

    def fake_subprocess_run(command, **kwargs):
        captured["command"] = command
        run_dir = Path(kwargs["env"]["RUN_DIR"])
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        (params_dir / "agent.yaml").write_text(
            "algorithm:\n  learning_rate: 0.0003\n  entropy_coef: 0.01\n",
            encoding="utf-8",
        )
        kwargs["stdout"].write("Learning iteration 1/1\n")
        return CompletedProcess()

    monkeypatch.setattr("runner.generate_run_id", fake_generate_run_id)
    monkeypatch.setattr("runner.subprocess.run", fake_subprocess_run)

    config = build_config(artifact_root=tmp_path / "artifacts")
    config.trainer_override_mode = "hydra"
    config.trainer_overrides = {
        "learning_rate": 0.0003,
        "entropy_coef": 0.01,
    }

    runner = TrainingRunner(
        project_root=project_root, artifact_root=tmp_path / "artifacts"
    )
    result = runner.run(config)

    effective_input = load_yaml(result.effective_input_path)
    trainer_override_resolution = load_yaml(result.trainer_override_resolution_path)
    trainer_override_verification = load_yaml(result.trainer_override_verification_path)

    assert "agent.algorithm.learning_rate=0.0003" in captured["command"]
    assert "agent.algorithm.entropy_coef=0.01" in captured["command"]
    assert trainer_override_resolution["mode"] == "hydra"
    assert trainer_override_resolution["effective"] is True
    assert trainer_override_verification["effective"] is True
    assert set(trainer_override_verification["matched"]) == {
        "learning_rate",
        "entropy_coef",
    }
    assert effective_input["trainer_override_mode"] == "hydra"
    assert effective_input["trainer_overrides_effective"] is True


def test_runner_locates_agent_config_from_stdout_log(tmp_path: Path) -> None:
    runner = TrainingRunner(
        project_root=tmp_path / "repo", artifact_root=tmp_path / "artifacts"
    )
    stdout_log_path = tmp_path / "stdout.log"
    run_dir = tmp_path / "run"
    external_log_root = tmp_path / "logs" / "rsl_rl" / "unitree_go2_flat"
    agent_config_path = (
        external_log_root / "2026-03-30_17-48-09" / "params" / "agent.yaml"
    )
    agent_config_path.parent.mkdir(parents=True)
    agent_config_path.write_text(
        "algorithm:\n  learning_rate: 0.0005\n",
        encoding="utf-8",
    )
    stdout_log_path.write_text(
        "[INFO] Logging experiment in directory: "
        f"{external_log_root}\n"
        "Exact experiment name requested from command line: 2026-03-30_17-48-09\n",
        encoding="utf-8",
    )

    resolved_path = runner._locate_agent_config(stdout_log_path, run_dir)

    assert resolved_path == agent_config_path


def test_runner_locates_env_config_from_stdout_log(tmp_path: Path) -> None:
    runner = TrainingRunner(
        project_root=tmp_path / "repo", artifact_root=tmp_path / "artifacts"
    )
    stdout_log_path = tmp_path / "stdout.log"
    run_dir = tmp_path / "run"
    external_log_root = tmp_path / "logs" / "rsl_rl" / "unitree_go2_flat"
    env_config_path = external_log_root / "2026-03-30_17-48-09" / "params" / "env.yaml"
    env_config_path.parent.mkdir(parents=True)
    env_config_path.write_text(
        "commands:\n  base_velocity:\n    ranges:\n      lin_vel_x: [0.0, 1.0]\n",
        encoding="utf-8",
    )
    stdout_log_path.write_text(
        "[INFO] Logging experiment in directory: "
        f"{external_log_root}\n"
        "Exact experiment name requested from command line: 2026-03-30_17-48-09\n",
        encoding="utf-8",
    )

    resolved_path = runner._locate_env_config(stdout_log_path, run_dir)

    assert resolved_path == env_config_path
