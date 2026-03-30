from __future__ import annotations

import pytest

from reward_overrides import (
    describe_reward_override_bounds,
    resolve_reward_overrides,
    verify_reward_overrides_from_env_config,
)


def test_describe_reward_override_bounds_returns_supported_fields() -> None:
    assert describe_reward_override_bounds() == {
        "track_lin_vel_xy_exp.weight": (0.0, 5.0),
        "track_ang_vel_z_exp.weight": (0.0, 5.0),
        "feet_air_time.weight": (0.0, 2.0),
        "flat_orientation_l2.weight": (-10.0, 0.0),
    }


def test_resolve_reward_overrides_hydra_mode_maps_supported_weights() -> None:
    resolution = resolve_reward_overrides(
        {
            "track_lin_vel_xy_exp.weight": 1.5,
            "feet_air_time.weight": 0.25,
        },
        "hydra",
    )

    assert resolution.metadata["effective"] is True
    assert resolution.cli_args == [
        "env.rewards.track_lin_vel_xy_exp.weight=1.5",
        "env.rewards.feet_air_time.weight=0.25",
    ]


def test_resolve_reward_overrides_rejects_out_of_bounds_value() -> None:
    with pytest.raises(
        ValueError, match="Reward override feet_air_time.weight out of bounds"
    ):
        resolve_reward_overrides({"feet_air_time.weight": 5.0}, "hydra")


def test_verify_reward_overrides_from_env_config_matches_weights() -> None:
    verification = verify_reward_overrides_from_env_config(
        {
            "track_lin_vel_xy_exp.weight": 1.5,
            "flat_orientation_l2.weight": -2.5,
        },
        {
            "rewards": {
                "track_lin_vel_xy_exp": {"weight": 1.5},
                "flat_orientation_l2": {"weight": -2.5},
            }
        },
    )

    assert verification["effective"] is True
    assert set(verification["matched"]) == {
        "track_lin_vel_xy_exp.weight",
        "flat_orientation_l2.weight",
    }
