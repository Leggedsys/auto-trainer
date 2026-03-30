from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


COMMAND_OVERRIDE_MODE_NONE = "none"
COMMAND_OVERRIDE_MODE_HYDRA = "hydra"
SUPPORTED_COMMAND_OVERRIDE_MODES = {
    COMMAND_OVERRIDE_MODE_NONE,
    COMMAND_OVERRIDE_MODE_HYDRA,
}
COMMAND_OVERRIDE_BOUNDS = {
    "lin_vel_x": (-1.5, 1.5),
    "lin_vel_y": (-1.5, 1.5),
    "ang_vel_z": (-2.0, 2.0),
    "heading": (-3.141592653589793, 3.141592653589793),
}
HYDRA_KEY_MAP = {
    "lin_vel_x": "env.commands.base_velocity.ranges.lin_vel_x",
    "lin_vel_y": "env.commands.base_velocity.ranges.lin_vel_y",
    "ang_vel_z": "env.commands.base_velocity.ranges.ang_vel_z",
    "heading": "env.commands.base_velocity.ranges.heading",
}
ENV_CONFIG_VALUE_PATHS = {
    "lin_vel_x": ("commands", "base_velocity", "ranges", "lin_vel_x"),
    "lin_vel_y": ("commands", "base_velocity", "ranges", "lin_vel_y"),
    "ang_vel_z": ("commands", "base_velocity", "ranges", "ang_vel_z"),
    "heading": ("commands", "base_velocity", "ranges", "heading"),
}


@dataclass(slots=True)
class CommandOverrideResolution:
    mode: str
    cli_args: list[str]
    metadata: dict[str, Any]


def describe_command_override_bounds() -> dict[str, tuple[float, float]]:
    return COMMAND_OVERRIDE_BOUNDS.copy()


def validate_command_overrides(
    command_overrides: dict[str, tuple[float, float] | list[float]],
) -> None:
    for key, value in command_overrides.items():
        bounds = COMMAND_OVERRIDE_BOUNDS.get(key)
        if bounds is None:
            raise ValueError(f"Unsupported command override field: {key}")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Command override {key} must be a 2-value range")
        low, high = value
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"Command override {key} values must be numeric")
        if low > high:
            raise ValueError(f"Command override {key} must satisfy low <= high")
        min_bound, max_bound = bounds
        if low < min_bound or high > max_bound:
            raise ValueError(
                f"Command override {key} out of bounds: [{low}, {high}] not in [{min_bound}, {max_bound}]"
            )


def resolve_command_overrides(
    command_overrides: dict[str, tuple[float, float] | list[float]],
    mode: str,
) -> CommandOverrideResolution:
    validate_command_overrides(command_overrides)

    if mode not in SUPPORTED_COMMAND_OVERRIDE_MODES:
        raise ValueError(f"Unsupported command override mode: {mode}")

    if mode == COMMAND_OVERRIDE_MODE_NONE:
        return CommandOverrideResolution(
            mode=mode,
            cli_args=[],
            metadata={
                "mode": mode,
                "supported": False,
                "effective": False,
                "validated": True,
                "requested_overrides": dict(command_overrides),
                "hydra_overrides": {},
            },
        )

    hydra_overrides = {}
    cli_args = []
    for key, value in command_overrides.items():
        hydra_key = HYDRA_KEY_MAP[key]
        hydra_value = f"[{value[0]}, {value[1]}]"
        hydra_overrides[hydra_key] = list(value)
        cli_args.append(f"{hydra_key}={hydra_value}")

    return CommandOverrideResolution(
        mode=mode,
        cli_args=cli_args,
        metadata={
            "mode": mode,
            "supported": True,
            "effective": True,
            "validated": True,
            "requested_overrides": dict(command_overrides),
            "hydra_overrides": hydra_overrides,
            "unsupported_keys": [],
        },
    )


def save_command_override_resolution(
    output_path: Path,
    resolution: CommandOverrideResolution,
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolution.metadata, file, sort_keys=False)


def verify_command_overrides_from_env_config(
    command_overrides: dict[str, tuple[float, float] | list[float]],
    env_config: dict[str, Any],
) -> dict[str, Any]:
    matched = {}
    mismatched = {}
    missing = {}

    for key, expected_value in command_overrides.items():
        path = ENV_CONFIG_VALUE_PATHS.get(key)
        if path is None:
            missing[key] = {"reason": "no_verification_path"}
            continue

        current: Any = env_config
        found = True
        for part in path:
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]

        if not found:
            missing[key] = {"reason": "path_missing", "path": list(path)}
            continue

        normalized_current = (
            list(current) if isinstance(current, (tuple, list)) else current
        )
        normalized_expected = list(expected_value)
        if normalized_current == normalized_expected:
            matched[key] = {"path": list(path), "value": normalized_current}
        else:
            mismatched[key] = {
                "path": list(path),
                "expected": normalized_expected,
                "actual": normalized_current,
            }

    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "effective": bool(command_overrides) and not mismatched and not missing,
    }
