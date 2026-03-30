from __future__ import annotations

import pytest

from command_overrides import (
    describe_command_override_bounds,
    resolve_command_overrides,
    verify_command_overrides_from_env_config,
)


def test_describe_command_override_bounds_returns_supported_fields() -> None:
    assert describe_command_override_bounds() == {
        "lin_vel_x": (-1.5, 1.5),
        "lin_vel_y": (-1.5, 1.5),
        "ang_vel_z": (-2.0, 2.0),
        "heading": (-3.141592653589793, 3.141592653589793),
    }


def test_resolve_command_overrides_hydra_mode_maps_supported_ranges() -> None:
    resolution = resolve_command_overrides(
        {
            "lin_vel_x": [0.0, 1.0],
            "ang_vel_z": [-1.0, 1.0],
        },
        "hydra",
    )

    assert resolution.metadata["effective"] is True
    assert resolution.cli_args == [
        "env.commands.base_velocity.ranges.lin_vel_x=[0.0, 1.0]",
        "env.commands.base_velocity.ranges.ang_vel_z=[-1.0, 1.0]",
    ]


def test_resolve_command_overrides_rejects_invalid_range_order() -> None:
    with pytest.raises(ValueError, match="must satisfy low <= high"):
        resolve_command_overrides({"lin_vel_x": [1.0, 0.0]}, "hydra")


def test_verify_command_overrides_from_env_config_matches_ranges() -> None:
    verification = verify_command_overrides_from_env_config(
        {
            "lin_vel_x": [0.0, 1.0],
            "heading": [0.0, 0.0],
        },
        {
            "commands": {
                "base_velocity": {
                    "ranges": {
                        "lin_vel_x": [0.0, 1.0],
                        "heading": [0.0, 0.0],
                    }
                }
            }
        },
    )

    assert verification["effective"] is True
    assert set(verification["matched"]) == {"lin_vel_x", "heading"}
