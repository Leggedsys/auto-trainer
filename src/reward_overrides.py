from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REWARD_OVERRIDE_MODE_NONE = "none"
REWARD_OVERRIDE_MODE_HYDRA = "hydra"
SUPPORTED_REWARD_OVERRIDE_MODES = {
    REWARD_OVERRIDE_MODE_NONE,
    REWARD_OVERRIDE_MODE_HYDRA,
}
REWARD_OVERRIDE_BOUNDS = {
    "track_lin_vel_xy_exp.weight": (0.0, 5.0),
    "track_ang_vel_z_exp.weight": (0.0, 5.0),
    "feet_air_time.weight": (0.0, 2.0),
    "flat_orientation_l2.weight": (-10.0, 0.0),
}
HYDRA_KEY_MAP = {key: f"env.rewards.{key}" for key in REWARD_OVERRIDE_BOUNDS}
ENV_CONFIG_VALUE_PATHS = {
    key: ("rewards", *key.split(".")) for key in REWARD_OVERRIDE_BOUNDS
}


@dataclass(slots=True)
class RewardOverrideResolution:
    mode: str
    cli_args: list[str]
    metadata: dict[str, Any]


def describe_reward_override_bounds() -> dict[str, tuple[float, float]]:
    return REWARD_OVERRIDE_BOUNDS.copy()


def validate_reward_overrides(
    reward_overrides: dict[str, float | int],
) -> None:
    for key, value in reward_overrides.items():
        bounds = REWARD_OVERRIDE_BOUNDS.get(key)
        if bounds is None:
            raise ValueError(f"Unsupported reward override field: {key}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Reward override {key} must be numeric")
        lower, upper = bounds
        if value < lower or value > upper:
            raise ValueError(
                f"Reward override {key} out of bounds: {value} not in [{lower}, {upper}]"
            )


def resolve_reward_overrides(
    reward_overrides: dict[str, float | int],
    mode: str,
) -> RewardOverrideResolution:
    validate_reward_overrides(reward_overrides)

    if mode not in SUPPORTED_REWARD_OVERRIDE_MODES:
        raise ValueError(f"Unsupported reward override mode: {mode}")

    if mode == REWARD_OVERRIDE_MODE_NONE:
        return RewardOverrideResolution(
            mode=mode,
            cli_args=[],
            metadata={
                "mode": mode,
                "supported": False,
                "effective": False,
                "validated": True,
                "requested_overrides": dict(reward_overrides),
                "hydra_overrides": {},
            },
        )

    hydra_overrides = {}
    cli_args = []
    for key, value in reward_overrides.items():
        hydra_key = HYDRA_KEY_MAP[key]
        hydra_overrides[hydra_key] = value
        cli_args.append(f"{hydra_key}={value}")

    return RewardOverrideResolution(
        mode=mode,
        cli_args=cli_args,
        metadata={
            "mode": mode,
            "supported": True,
            "effective": True,
            "validated": True,
            "requested_overrides": dict(reward_overrides),
            "hydra_overrides": hydra_overrides,
            "unsupported_keys": [],
        },
    )


def save_reward_override_resolution(
    output_path: Path,
    resolution: RewardOverrideResolution,
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolution.metadata, file, sort_keys=False)


def verify_reward_overrides_from_env_config(
    reward_overrides: dict[str, float | int],
    env_config: dict[str, Any],
) -> dict[str, Any]:
    matched = {}
    mismatched = {}
    missing = {}

    for key, expected_value in reward_overrides.items():
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
        "effective": bool(reward_overrides) and not mismatched and not missing,
    }
