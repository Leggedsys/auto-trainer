from __future__ import annotations

from trainer_overrides import verify_trainer_overrides_from_agent_config


def test_verify_trainer_overrides_from_agent_config_matches_expected_values() -> None:
    verification = verify_trainer_overrides_from_agent_config(
        {
            "learning_rate": 0.0003,
            "clip_param": 0.2,
        },
        {
            "algorithm": {
                "learning_rate": 0.0003,
                "clip_param": 0.2,
            }
        },
    )

    assert verification["effective"] is True
    assert set(verification["matched"]) == {"learning_rate", "clip_param"}
    assert verification["mismatched"] == {}
    assert verification["missing"] == {}


def test_verify_trainer_overrides_from_agent_config_reports_mismatch() -> None:
    verification = verify_trainer_overrides_from_agent_config(
        {"learning_rate": 0.0003},
        {"algorithm": {"learning_rate": 0.001}},
    )

    assert verification["effective"] is False
    assert verification["matched"] == {}
    assert verification["missing"] == {}
    assert verification["mismatched"]["learning_rate"]["actual"] == 0.001
