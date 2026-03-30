from __future__ import annotations

import pytest

from trainer_overrides import (
    describe_trainer_override_bounds,
    resolve_trainer_overrides,
)


def test_describe_trainer_override_bounds_returns_supported_fields() -> None:
    assert describe_trainer_override_bounds() == {
        "learning_rate": (1e-6, 1e-2),
        "entropy_coef": (0.0, 0.1),
        "clip_param": (0.05, 0.4),
        "num_mini_batches": (1, 64),
    }


def test_resolve_trainer_overrides_none_mode_is_non_effective() -> None:
    resolution = resolve_trainer_overrides({"learning_rate": 0.0003}, "none")

    assert resolution.cli_args == []
    assert resolution.metadata["mode"] == "none"
    assert resolution.metadata["effective"] is False
    assert resolution.metadata["validated"] is True
    assert resolution.metadata["requested_overrides"] == {"learning_rate": 0.0003}


def test_resolve_trainer_overrides_hydra_mode_maps_supported_keys() -> None:
    resolution = resolve_trainer_overrides(
        {
            "learning_rate": 0.0003,
            "entropy_coef": 0.01,
            "num_mini_batches": 4,
        },
        "hydra",
    )

    assert resolution.metadata["effective"] is True
    assert resolution.metadata["unsupported_keys"] == []
    assert resolution.cli_args == [
        "agent.algorithm.learning_rate=0.0003",
        "agent.algorithm.entropy_coef=0.01",
        "agent.algorithm.num_mini_batches=4",
    ]


def test_resolve_trainer_overrides_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported trainer override mode"):
        resolve_trainer_overrides({}, "bad")


def test_resolve_trainer_overrides_rejects_unsupported_field() -> None:
    with pytest.raises(ValueError, match="Unsupported trainer override field"):
        resolve_trainer_overrides({"unknown": 1}, "hydra")


def test_resolve_trainer_overrides_rejects_out_of_bounds_value() -> None:
    with pytest.raises(
        ValueError, match="Trainer override learning_rate out of bounds"
    ):
        resolve_trainer_overrides({"learning_rate": 1.0}, "hydra")
