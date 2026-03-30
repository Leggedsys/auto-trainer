from __future__ import annotations

import os
import subprocess
from pathlib import Path

from command_overrides import (
    resolve_command_overrides,
    save_command_override_resolution,
    verify_command_overrides_from_env_config,
)
from models import RunResult, TrainingConfig
from parser import parse_training_summary, save_summary
from reward_overrides import (
    resolve_reward_overrides,
    save_reward_override_resolution,
    verify_reward_overrides_from_env_config,
)
from trainer_overrides import (
    resolve_trainer_overrides,
    save_trainer_override_resolution,
    verify_trainer_overrides_from_agent_config,
)
from utils import generate_run_id, load_python_yaml, load_yaml, save_yaml


class TrainingRunner:
    def __init__(self, project_root: Path, artifact_root: Path) -> None:
        self.project_root = project_root
        self.artifact_root = artifact_root
        self.runs_dir = artifact_root / "runs"

    def run(self, config: TrainingConfig) -> RunResult:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = generate_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        config_snapshot_path = run_dir / "config_snapshot.yaml"
        effective_input_path = run_dir / "effective_input.yaml"
        trainer_override_resolution_path = run_dir / "trainer_override_resolution.yaml"
        trainer_override_verification_path = (
            run_dir / "trainer_override_verification.yaml"
        )
        command_override_resolution_path = run_dir / "command_override_resolution.yaml"
        command_override_verification_path = (
            run_dir / "command_override_verification.yaml"
        )
        reward_override_resolution_path = run_dir / "reward_override_resolution.yaml"
        reward_override_verification_path = (
            run_dir / "reward_override_verification.yaml"
        )
        stdout_log_path = run_dir / "stdout.log"
        stderr_log_path = run_dir / "stderr.log"
        summary_path = run_dir / "summary.json"

        save_yaml(
            config_snapshot_path,
            {
                "target": {
                    "id": config.target_id,
                    "artifact_root": str(config.artifact_root),
                },
                "isaaclab": {
                    "root_dir": config.isaaclab_root_dir,
                    "launcher": config.isaaclab_launcher,
                    "train_script": config.train_script,
                },
                "training": {
                    "task": config.task,
                    "headless": config.headless,
                    "num_envs": config.num_envs,
                    "max_iterations": config.max_iterations,
                },
            },
        )

        script_path = self.project_root / "scripts" / "train_go2.sh"
        command = [
            str(script_path),
            "--isaaclab-root",
            config.isaaclab_root_dir,
            "--launcher",
            config.isaaclab_launcher,
            "--train-script",
            config.train_script,
            "--task",
            config.task,
            "--num-envs",
            str(config.num_envs),
            "--max-iterations",
            str(config.max_iterations),
        ]
        trainer_override_resolution = resolve_trainer_overrides(
            config.trainer_overrides,
            config.trainer_override_mode,
        )
        command_override_resolution = resolve_command_overrides(
            config.command_overrides,
            config.command_override_mode,
        )
        reward_override_resolution = resolve_reward_overrides(
            config.reward_overrides,
            config.reward_override_mode,
        )
        command.extend(trainer_override_resolution.cli_args)
        command.extend(command_override_resolution.cli_args)
        command.extend(reward_override_resolution.cli_args)
        if config.headless:
            command.append("--headless")

        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)

        save_trainer_override_resolution(
            trainer_override_resolution_path,
            trainer_override_resolution,
        )
        save_command_override_resolution(
            command_override_resolution_path,
            command_override_resolution,
        )
        save_reward_override_resolution(
            reward_override_resolution_path,
            reward_override_resolution,
        )

        save_yaml(
            effective_input_path,
            {
                "command": command,
                "cwd": str(self.project_root),
                "training": {
                    "task": config.task,
                    "headless": config.headless,
                    "num_envs": config.num_envs,
                    "max_iterations": config.max_iterations,
                },
                "trainer_override_mode": config.trainer_override_mode,
                "trainer_overrides": config.trainer_overrides,
                "trainer_override_resolution": trainer_override_resolution.metadata,
                "trainer_overrides_effective": trainer_override_resolution.metadata[
                    "effective"
                ],
                "command_override_mode": config.command_override_mode,
                "command_overrides": config.command_overrides,
                "command_override_resolution": command_override_resolution.metadata,
                "command_overrides_effective": command_override_resolution.metadata[
                    "effective"
                ],
                "reward_override_mode": config.reward_override_mode,
                "reward_overrides": config.reward_overrides,
                "reward_override_resolution": reward_override_resolution.metadata,
                "reward_overrides_effective": reward_override_resolution.metadata[
                    "effective"
                ],
            },
        )

        with (
            stdout_log_path.open("w", encoding="utf-8") as stdout_file,
            stderr_log_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
                env=env,
            )

        summary = parse_training_summary(stdout_log_path=stdout_log_path)
        save_summary(summary_path=summary_path, summary=summary)

        agent_config_path = self._locate_agent_config(stdout_log_path, run_dir)
        if agent_config_path.exists():
            agent_config = load_yaml(agent_config_path)
            verification = verify_trainer_overrides_from_agent_config(
                config.trainer_overrides,
                agent_config,
            )
            verification["agent_config_path"] = str(agent_config_path)
        else:
            verification = {
                "matched": {},
                "mismatched": {},
                "missing": {
                    key: {"reason": "agent_config_artifact_missing"}
                    for key in config.trainer_overrides
                },
                "effective": False,
                "agent_config_path": str(agent_config_path),
            }
        save_yaml(trainer_override_verification_path, verification)

        env_config_path = self._locate_env_config(stdout_log_path, run_dir)
        if env_config_path.exists():
            env_config = load_python_yaml(env_config_path)
            command_verification = verify_command_overrides_from_env_config(
                config.command_overrides,
                env_config,
            )
            command_verification["env_config_path"] = str(env_config_path)
        else:
            command_verification = {
                "matched": {},
                "mismatched": {},
                "missing": {
                    key: {"reason": "env_config_artifact_missing"}
                    for key in config.command_overrides
                },
                "effective": False,
                "env_config_path": str(env_config_path),
            }
        save_yaml(command_override_verification_path, command_verification)

        if env_config_path.exists():
            reward_verification = verify_reward_overrides_from_env_config(
                config.reward_overrides,
                env_config,
            )
            reward_verification["env_config_path"] = str(env_config_path)
        else:
            reward_verification = {
                "matched": {},
                "mismatched": {},
                "missing": {
                    key: {"reason": "env_config_artifact_missing"}
                    for key in config.reward_overrides
                },
                "effective": False,
                "env_config_path": str(env_config_path),
            }
        save_yaml(reward_override_verification_path, reward_verification)

        return RunResult(
            run_id=run_id,
            target_id=config.target_id,
            run_dir=run_dir,
            command=command,
            effective_input_path=effective_input_path,
            trainer_override_resolution_path=trainer_override_resolution_path,
            trainer_override_verification_path=trainer_override_verification_path,
            command_override_resolution_path=command_override_resolution_path,
            command_override_verification_path=command_override_verification_path,
            reward_override_resolution_path=reward_override_resolution_path,
            reward_override_verification_path=reward_override_verification_path,
            return_code=completed.returncode,
            config_snapshot_path=config_snapshot_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            summary_path=summary_path,
            summary=summary,
        )

    def _locate_agent_config(self, stdout_log_path: Path, run_dir: Path) -> Path:
        local_path = run_dir / "params" / "agent.yaml"
        if local_path.exists():
            return local_path

        text = stdout_log_path.read_text(encoding="utf-8", errors="replace")
        log_root_path: Path | None = None
        run_name: str | None = None
        for line in text.splitlines():
            root_marker = "[INFO] Logging experiment in directory:"
            if root_marker in line:
                log_root_path = Path(line.split(root_marker, maxsplit=1)[1].strip())
            name_marker = "Exact experiment name requested from command line:"
            if name_marker in line:
                run_name = line.split(name_marker, maxsplit=1)[1].strip()

        if log_root_path is not None and run_name:
            candidate = log_root_path / run_name / "params" / "agent.yaml"
            if candidate.exists():
                return candidate

        return local_path

    def _locate_env_config(self, stdout_log_path: Path, run_dir: Path) -> Path:
        local_path = run_dir / "params" / "env.yaml"
        if local_path.exists():
            return local_path

        text = stdout_log_path.read_text(encoding="utf-8", errors="replace")
        log_root_path: Path | None = None
        run_name: str | None = None
        for line in text.splitlines():
            root_marker = "[INFO] Logging experiment in directory:"
            if root_marker in line:
                log_root_path = Path(line.split(root_marker, maxsplit=1)[1].strip())
            name_marker = "Exact experiment name requested from command line:"
            if name_marker in line:
                run_name = line.split(name_marker, maxsplit=1)[1].strip()

        if log_root_path is not None and run_name:
            candidate = log_root_path / run_name / "params" / "env.yaml"
            if candidate.exists():
                return candidate

        return local_path
