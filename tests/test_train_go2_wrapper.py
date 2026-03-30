from __future__ import annotations

import subprocess
from pathlib import Path


def test_train_go2_wrapper_preserves_unknown_extra_args(tmp_path: Path) -> None:
    launcher = tmp_path / "isaaclab.sh"
    train_script = tmp_path / "train.py"
    output_path = tmp_path / "cmd.txt"
    wrapper = Path("scripts/train_go2.sh")

    launcher.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    train_script.write_text("# stub\n", encoding="utf-8")

    env = {"CAPTURE_PATH": str(output_path)}
    subprocess.run(
        [
            str(wrapper),
            "--launcher",
            str(launcher),
            "--train-script",
            str(train_script),
            "--task",
            "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "--num-envs",
            "256",
            "--max-iterations",
            "50",
            "agent.algorithm.learning_rate=0.0003",
            "agent.algorithm.entropy_coef=0.01",
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
    )

    captured_args = output_path.read_text(encoding="utf-8").splitlines()

    assert "-p" in captured_args
    assert str(train_script) in captured_args
    assert "agent.algorithm.learning_rate=0.0003" in captured_args
    assert "agent.algorithm.entropy_coef=0.01" in captured_args
