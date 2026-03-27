from __future__ import annotations

import os
import subprocess
from pathlib import Path

from models import RunResult, TrainingConfig
from parser import parse_training_summary, save_summary
from utils import generate_run_id, save_yaml


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
        if config.headless:
            command.append("--headless")

        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)

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

        return RunResult(
            run_id=run_id,
            target_id=config.target_id,
            run_dir=run_dir,
            command=command,
            return_code=completed.returncode,
            config_snapshot_path=config_snapshot_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            summary_path=summary_path,
            summary=summary,
        )
