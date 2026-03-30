from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


TRAINER_OVERRIDE_MODE_NONE = "none"
TRAINER_OVERRIDE_MODE_HYDRA = "hydra"
SUPPORTED_TRAINER_OVERRIDE_MODES = {
    TRAINER_OVERRIDE_MODE_NONE,
    TRAINER_OVERRIDE_MODE_HYDRA,
}
TRAINER_OVERRIDE_BOUNDS = {
    "learning_rate": (1e-6, 1e-2),
    "entropy_coef": (0.0, 0.1),
    "clip_param": (0.05, 0.4),
    "num_mini_batches": (1, 64),
}
HYDRA_KEY_MAP = {
    "learning_rate": "agent.algorithm.learning_rate",
    "entropy_coef": "agent.algorithm.entropy_coef",
    "clip_param": "agent.algorithm.clip_param",
    "num_mini_batches": "agent.algorithm.num_mini_batches",
    "max_iterations": "agent.max_iterations",
}
AGENT_CONFIG_VALUE_PATHS = {
    "learning_rate": ("algorithm", "learning_rate"),
    "entropy_coef": ("algorithm", "entropy_coef"),
    "clip_param": ("algorithm", "clip_param"),
    "num_mini_batches": ("algorithm", "num_mini_batches"),
    "max_iterations": ("max_iterations",),
}


@dataclass(slots=True)
class TrainerOverrideResolution:
    mode: str
    cli_args: list[str]
    metadata: dict[str, Any]


def describe_trainer_override_bounds() -> dict[str, tuple[float, float]]:
    return TRAINER_OVERRIDE_BOUNDS.copy()


def validate_trainer_overrides(
    trainer_overrides: dict[str, float | int | bool],
) -> None:
    for key, value in trainer_overrides.items():
        bounds = TRAINER_OVERRIDE_BOUNDS.get(key)
        if bounds is None:
            raise ValueError(f"Unsupported trainer override field: {key}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Trainer override {key} must be numeric")
        lower, upper = bounds
        if value < lower or value > upper:
            raise ValueError(
                f"Trainer override {key} out of bounds: {value} not in [{lower}, {upper}]"
            )


def resolve_trainer_overrides(
    trainer_overrides: dict[str, float | int | bool],
    mode: str,
) -> TrainerOverrideResolution:
    validate_trainer_overrides(trainer_overrides)

    if mode not in SUPPORTED_TRAINER_OVERRIDE_MODES:
        raise ValueError(f"Unsupported trainer override mode: {mode}")

    if mode == TRAINER_OVERRIDE_MODE_NONE:
        return TrainerOverrideResolution(
            mode=mode,
            cli_args=[],
            metadata={
                "mode": mode,
                "supported": False,
                "effective": False,
                "validated": True,
                "requested_overrides": dict(trainer_overrides),
                "hydra_overrides": {},
            },
        )

    hydra_overrides = {}
    cli_args = []
    unsupported_keys = []
    for key, value in trainer_overrides.items():
        hydra_key = HYDRA_KEY_MAP.get(key)
        if hydra_key is None:
            unsupported_keys.append(key)
            continue
        hydra_overrides[hydra_key] = value
        cli_args.append(f"{hydra_key}={value}")

    return TrainerOverrideResolution(
        mode=mode,
        cli_args=cli_args,
        metadata={
            "mode": mode,
            "supported": not unsupported_keys,
            "effective": not unsupported_keys,
            "validated": True,
            "requested_overrides": dict(trainer_overrides),
            "hydra_overrides": hydra_overrides,
            "unsupported_keys": unsupported_keys,
        },
    )


def save_trainer_override_resolution(
    output_path: Path,
    resolution: TrainerOverrideResolution,
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolution.metadata, file, sort_keys=False)


def verify_trainer_overrides_from_agent_config(
    trainer_overrides: dict[str, float | int | bool],
    agent_config: dict[str, Any],
) -> dict[str, Any]:
    matched = {}
    mismatched = {}
    missing = {}

    for key, expected_value in trainer_overrides.items():
        path = AGENT_CONFIG_VALUE_PATHS.get(key)
        if path is None:
            missing[key] = {"reason": "no_verification_path"}
            continue

        current: Any = agent_config
        found = True
        for part in path:
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]

        if not found:
            missing[key] = {"reason": "path_missing", "path": list(path)}
            continue

        if current == expected_value:
            matched[key] = {"path": list(path), "value": current}
        else:
            mismatched[key] = {
                "path": list(path),
                "expected": expected_value,
                "actual": current,
            }

    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "effective": bool(trainer_overrides) and not mismatched and not missing,
    }
