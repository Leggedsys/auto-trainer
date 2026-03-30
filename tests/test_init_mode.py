from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from main import (
    build_capability_status,
    build_onboarding_report,
    build_task_policy,
    run_init_mode,
    summarize_field_registry,
)
from utils import load_yaml


def test_summarize_field_registry_counts_enabled_fields() -> None:
    summary = summarize_field_registry(
        {
            "fields": [
                {
                    "field": "training.learning_rate",
                    "layer": "training",
                    "validation_source": "agent.yaml",
                    "enabled_for_planner": True,
                },
                {
                    "field": "reward.track_lin_vel_xy_exp.weight",
                    "layer": "reward",
                    "validation_source": "env.yaml",
                    "enabled_for_planner": False,
                },
            ]
        }
    )

    assert summary["field_count"] == 2
    assert summary["planner_enabled_count"] == 1
    assert summary["planner_enabled_fields"] == ["training.learning_rate"]
    assert summary["layers"] == {"training": 1, "reward": 1}
    assert summary["validation_sources"] == {"agent.yaml": 1, "env.yaml": 1}


def test_build_capability_status_marks_fields_as_wired() -> None:
    status = build_capability_status(
        {
            "fields": [
                {
                    "field": "training.learning_rate",
                    "layer": "training",
                    "hydra_path": "agent.algorithm.learning_rate",
                    "validation_source": "agent.yaml",
                    "enabled_for_planner": True,
                }
            ]
        },
        {
            "normalized_fields": [
                {
                    "normalized_field": "training.learning_rate",
                }
            ],
            "not_applicable_families": [],
        },
    )

    assert status["fields"][0]["known"] is True
    assert status["fields"][0]["mapped"] is True
    assert status["fields"][0]["resolvable"] is True
    assert status["fields"][0]["verifiable"] is True
    assert status["fields"][0]["planner_enabled"] is True
    assert status["layers"]["training"]["fully_wired_count"] == 1


def test_build_capability_status_uses_normalized_candidates() -> None:
    status = build_capability_status(
        {
            "fields": [
                {
                    "field": "training.learning_rate",
                    "layer": "training",
                    "hydra_path": "agent.algorithm.learning_rate",
                    "validation_source": "agent.yaml",
                    "enabled_for_planner": True,
                },
                {
                    "field": "command.lin_vel_x",
                    "layer": "command",
                    "hydra_path": "env.commands.base_velocity.ranges.lin_vel_x",
                    "validation_source": "env.yaml",
                    "enabled_for_planner": True,
                },
            ]
        },
        {
            "normalized_fields": [
                {
                    "normalized_field": "training.learning_rate",
                }
            ],
            "not_applicable_families": ["command"],
        },
    )

    assert status["fields"][0]["planner_enabled"] is True
    assert status["fields"][1]["planner_enabled"] is False
    assert status["fields"][1]["family_status"] == "not_applicable"


def test_build_task_policy_returns_structured_defaults() -> None:
    policy = build_task_policy(
        {"training": {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0"}}
    )

    assert policy["task_id"] == "Isaac-Velocity-Flat-Unitree-Go2-v0"
    assert policy["primary_metric"] == "mean_reward"
    assert policy["plateau_policy"]["type"] == "manual_review_required"


def test_build_onboarding_report_summarizes_layer_wiring() -> None:
    report = build_onboarding_report(
        project_identity={"target_id": "isaaclab_go2"},
        field_registry_summary={
            "planner_enabled_fields": ["training.learning_rate"],
        },
        capability_status={
            "layers": {
                "training": {
                    "field_count": 2,
                    "fully_wired_count": 2,
                    "planner_enabled_count": 2,
                },
                "reward": {
                    "field_count": 2,
                    "fully_wired_count": 1,
                    "planner_enabled_count": 1,
                },
            }
        },
        normalized_candidates={"normalized_fields": [{}, {}]},
        discovery_error=None,
    )

    assert report["summary"]["planner_enabled_field_count"] == 1
    assert report["summary"]["normalized_candidate_count"] == 2
    assert report["gaps"]["automatic_discovery"] == "implemented_in_restricted_mode"
    assert report["summary"]["fully_wired_layers"] == ["training"]
    assert report["summary"]["partially_wired_layers"] == ["reward"]


def test_run_init_mode_records_field_registry_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = {
        "target": {
            "id": "isaaclab_go2",
            "project_root": str(project_root),
            "artifact_root": str(artifact_root),
        },
        "isaaclab": {
            "root_dir": "/tmp/isaaclab",
            "launcher": "/tmp/isaaclab/isaaclab.sh",
            "train_script": "/tmp/isaaclab/train.py",
        },
        "training": {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0"},
    }
    config_path.write_text("target: {}\n", encoding="utf-8")

    args = Namespace(
        target_id="isaaclab_go2",
        project_root=str(project_root),
        artifact_root=str(artifact_root),
        task_registry_path=None,
        env_cfg_path=None,
        agent_cfg_path=None,
    )

    capability_package_dir = artifact_root / "capability_package"
    capability_package_dir.mkdir(parents=True)
    (capability_package_dir / "normalized_candidate_list.yaml").write_text(
        "project:\n"
        "  task_id: isaaclab_go2\n"
        "normalized_fields:\n"
        "  - normalized_field: training.learning_rate\n"
        "not_applicable_families: []\n"
        "unmapped_source_fields: []\n",
        encoding="utf-8",
    )

    run_init_mode(config_path, config, args)

    target_info = load_yaml(artifact_root / "target_info.yaml")
    project_identity = load_yaml(
        artifact_root / "capability_package" / "project_identity.yaml"
    )
    capability_status = load_yaml(
        artifact_root / "capability_package" / "capability_status.yaml"
    )
    task_policy = load_yaml(artifact_root / "capability_package" / "task_policy.yaml")
    onboarding_report = load_yaml(
        artifact_root / "capability_package" / "onboarding_report.yaml"
    )

    assert target_info["target"]["id"] == "isaaclab_go2"
    assert target_info["discovered"]["field_registry_exists"] is True
    assert target_info["field_registry"]["planner_enabled_count"] >= 1
    assert "training" in target_info["field_registry"]["layers"]
    assert project_identity["target_id"] == "isaaclab_go2"
    assert project_identity["workspace_root"] == str(artifact_root)
    assert "training" in capability_status["layers"]
    assert (
        target_info["capability_status"]["layer_summary"]["training"][
            "fully_wired_count"
        ]
        >= 1
    )
    assert task_policy["primary_metric"] == "mean_reward"
    assert onboarding_report["project"]["target_id"] == "isaaclab_go2"
