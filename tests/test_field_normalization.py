from __future__ import annotations

from field_normalization import normalize_legged_locomotion_fields


def test_normalize_legged_locomotion_fields_maps_core_fields() -> None:
    normalized = normalize_legged_locomotion_fields(
        {
            "project": {
                "task_id": "Isaac-Velocity-Flat-Unitree-Go2-v0",
                "task_type": "ManagerBasedRLEnv",
            },
            "candidate_fields": [
                {
                    "field": "max_iterations",
                    "kind": "scalar",
                    "source_file": "agent cfg",
                    "default_value": 1500,
                    "candidate_for_planner": True,
                },
                {
                    "field": "algorithm.learning_rate",
                    "kind": "scalar",
                    "source_file": "agent cfg",
                    "default_value": 0.001,
                    "candidate_for_planner": True,
                },
                {
                    "field": "rewards.track_lin_vel_xy_exp.weight",
                    "kind": "scalar",
                    "source_file": "env cfg",
                    "default_value": 1.5,
                    "candidate_for_planner": True,
                },
                {
                    "field": "commands.base_velocity.ranges.lin_vel_x",
                    "kind": "range",
                    "source_file": "env cfg",
                    "default_value": [0.0, 1.0],
                    "candidate_for_planner": True,
                },
            ],
            "absent_field_families": [],
        }
    )

    normalized_fields = {
        item["normalized_field"]: item for item in normalized["normalized_fields"]
    }
    assert "training.max_iterations" in normalized_fields
    assert "training.learning_rate" in normalized_fields
    assert "reward.track_lin_vel_xy_exp.weight" in normalized_fields
    assert "command.lin_vel_x" in normalized_fields
    assert (
        normalized_fields["training.learning_rate"]["hydra_path"]
        == "agent.algorithm.learning_rate"
    )
    assert (
        normalized_fields["reward.track_lin_vel_xy_exp.weight"]["hydra_path"]
        == "env.rewards.track_lin_vel_xy_exp.weight"
    )


def test_normalize_legged_locomotion_fields_marks_absent_command_family() -> None:
    normalized = normalize_legged_locomotion_fields(
        {
            "project": {},
            "candidate_fields": [],
            "absent_field_families": ["command_ranges"],
        }
    )

    assert normalized["not_applicable_families"] == ["command"]


def test_normalize_legged_locomotion_fields_leaves_unmapped_algorithm_fields() -> None:
    normalized = normalize_legged_locomotion_fields(
        {
            "project": {},
            "candidate_fields": [
                {
                    "field": "algorithm.gamma",
                    "kind": "scalar",
                    "source_file": "agent cfg",
                    "default_value": 0.99,
                    "candidate_for_planner": True,
                }
            ],
            "absent_field_families": [],
        }
    )

    assert normalized["normalized_fields"] == []
    assert normalized["unmapped_source_fields"] == [
        {
            "source_field": "algorithm.gamma",
            "reason": "no_stable_legged_locomotion_mapping",
        }
    ]
