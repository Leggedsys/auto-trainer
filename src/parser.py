from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]|\x1B[@-_]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _extract_last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return float(matches[-1])


def _extract_last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return int(matches[-1])


def parse_training_summary(
    stdout_log_path: Path, training_log_dir: Path | None = None
) -> dict[str, Any]:
    del training_log_dir

    raw_text = stdout_log_path.read_text(encoding="utf-8", errors="replace")
    text = _strip_ansi(raw_text)

    return {
        "mean_reward": _extract_last_float(r"Mean reward:\s*(-?\d+(?:\.\d+)?)", text),
        "mean_episode_length": _extract_last_float(
            r"Mean episode length:\s*(-?\d+(?:\.\d+)?)", text
        ),
        "value_loss": _extract_last_float(
            r"Mean value_function loss:\s*(-?\d+(?:\.\d+)?)", text
        ),
        "policy_loss": _extract_last_float(
            r"Mean surrogate loss:\s*(-?\d+(?:\.\d+)?)", text
        ),
        "entropy": _extract_last_float(r"Mean entropy loss:\s*(-?\d+(?:\.\d+)?)", text),
        "iteration": _extract_last_int(r"Learning iteration\s+(\d+)\/\d+", text),
    }


def save_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
