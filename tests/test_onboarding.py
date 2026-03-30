from __future__ import annotations

from pathlib import Path

from onboarding import (
    build_generated_field_registry,
    build_legged_locomotion_source_snippets,
    build_normalized_candidate_list,
    inject_legged_trainer_fallback_candidates,
    is_legged_locomotion_task,
    load_raw_discovery,
    run_field_discovery_and_normalization,
    save_generated_field_registry,
    save_normalized_candidate_list,
    update_capability_package_with_normalized_fields,
)
from utils import load_yaml


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_is_legged_locomotion_task_uses_velocity_heuristic() -> None:
    assert is_legged_locomotion_task("Isaac-Velocity-Flat-Unitree-Go2-v0") is True
    assert is_legged_locomotion_task("Template-Autotrain-v0") is False


def test_build_legged_locomotion_source_snippets_collects_available_files(
    tmp_path: Path,
) -> None:
    env_cfg = tmp_path / "env.py"
    agent_cfg = tmp_path / "agent.py"
    registry_init = tmp_path / "__init__.py"
    env_cfg.write_text("env cfg\n", encoding="utf-8")
    agent_cfg.write_text("agent cfg\n", encoding="utf-8")
    registry_init.write_text("registry\n", encoding="utf-8")

    snippets = build_legged_locomotion_source_snippets(
        project_root=tmp_path,
        task_id="Isaac-Velocity-Flat-Unitree-Go2-v0",
        env_cfg_path=env_cfg,
        agent_cfg_path=agent_cfg,
        registry_init_path=registry_init,
    )

    labels = [snippet["label"] for snippet in snippets]
    assert "task registration" in labels
    assert "env cfg" in labels
    assert "agent cfg" in labels


def test_onboarding_normalizes_raw_discovery() -> None:
    normalized = build_normalized_candidate_list(
        {
            "project": {
                "task_id": "Isaac-Velocity-Flat-Unitree-Go2-v0",
                "task_type": "ManagerBasedRLEnv",
            },
            "candidate_fields": [
                {
                    "field": "algorithm.learning_rate",
                    "kind": "scalar",
                    "source_file": "agent cfg",
                    "default_value": 0.001,
                    "candidate_for_planner": True,
                }
            ],
            "absent_field_families": ["command_ranges"],
        }
    )

    assert (
        normalized["normalized_fields"][0]["normalized_field"]
        == "training.learning_rate"
    )
    assert normalized["not_applicable_families"] == ["command"]


def test_onboarding_can_load_and_save_normalized_candidate_list(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.yaml"
    normalized_path = tmp_path / "normalized.yaml"
    raw_path.write_text(
        "project:\n"
        "  task_id: Isaac-Velocity-Flat-Unitree-Go2-v0\n"
        "  task_type: ManagerBasedRLEnv\n"
        "candidate_fields:\n"
        "  - field: algorithm.learning_rate\n"
        "    kind: scalar\n"
        "    source_file: agent cfg\n"
        "    default_value: 0.001\n"
        "    candidate_for_planner: true\n"
        "absent_field_families: []\n",
        encoding="utf-8",
    )

    raw = load_raw_discovery(raw_path)
    normalized = build_normalized_candidate_list(raw)
    save_normalized_candidate_list(normalized_path, normalized)
    loaded = load_yaml(normalized_path)

    assert (
        loaded["normalized_fields"][0]["normalized_field"] == "training.learning_rate"
    )


def test_onboarding_can_run_discovery_and_normalization() -> None:
    client = StubLLMClient(
        '{"project":{"task_id":"Isaac-Velocity-Flat-Unitree-Go2-v0","task_type":"ManagerBasedRLEnv"},'
        '"candidate_fields":[{"field":"algorithm.learning_rate","kind":"scalar","source_file":"agent cfg",'
        '"source_symbol":"cfg","default_value":0.001,"hydra_path":"algorithm.learning_rate",'
        '"candidate_for_planner":true,"layer":"training","reason":"test"}],'
        '"absent_field_families":["command_ranges"]}'
    )

    raw_discovery, normalized = run_field_discovery_and_normalization(
        task_id="Isaac-Velocity-Flat-Unitree-Go2-v0",
        task_type="ManagerBasedRLEnv",
        source_snippets=[],
        llm_client=client,
        agent_cfg_path=None,
    )

    assert raw_discovery["candidate_fields"][0]["field"] == "algorithm.learning_rate"
    assert (
        normalized["normalized_fields"][0]["normalized_field"]
        == "training.learning_rate"
    )


def test_inject_legged_trainer_fallback_candidates_adds_core_trainer_fields(
    tmp_path: Path,
) -> None:
    agent_cfg_path = tmp_path / "agent.py"
    agent_cfg_path.write_text(
        "max_iterations = 1500\n"
        "algorithm = dict(learning_rate=1.0e-3, entropy_coef=0.01, clip_param=0.2, num_mini_batches=4)\n",
        encoding="utf-8",
    )

    raw = inject_legged_trainer_fallback_candidates(
        {
            "candidate_fields": [],
            "absent_field_families": ["training"],
        },
        agent_cfg_path,
    )

    fields = {entry["field"] for entry in raw["candidate_fields"]}
    assert "max_iterations" in fields
    assert "algorithm.learning_rate" in fields
    assert raw["absent_field_families"] == []


def test_onboarding_updates_capability_package_with_normalized_fields(
    tmp_path: Path,
) -> None:
    capability_package_dir = tmp_path / "capability_package"
    capability_package_dir.mkdir()
    normalized = {
        "project": {"task_id": "Isaac-Velocity-Flat-Unitree-Go2-v0"},
        "normalized_fields": [
            {"normalized_field": "training.learning_rate"},
        ],
        "not_applicable_families": ["command"],
        "unmapped_source_fields": [],
    }

    update_capability_package_with_normalized_fields(capability_package_dir, normalized)

    loaded = load_yaml(capability_package_dir / "normalized_candidate_list.yaml")
    assert (
        loaded["normalized_fields"][0]["normalized_field"] == "training.learning_rate"
    )


def test_onboarding_can_build_and_save_generated_field_registry(
    tmp_path: Path,
) -> None:
    generated = build_generated_field_registry(
        {
            "project": {"task_id": "Isaac-Velocity-Flat-Unitree-Go2-v0"},
            "normalized_fields": [
                {
                    "source_field": "algorithm.learning_rate",
                    "normalized_field": "training.learning_rate",
                    "normalized_layer": "training",
                    "kind": "scalar",
                    "hydra_path": "agent.algorithm.learning_rate",
                    "default_value": 0.001,
                    "validation_source": "agent.yaml",
                    "candidate_for_planner": True,
                    "normalization_reason": "normalized_from_algorithm.learning_rate",
                    "source_file": "agent cfg",
                }
            ],
            "not_applicable_families": ["command"],
            "unmapped_source_fields": [],
        }
    )
    output_path = tmp_path / "field_registry.generated.yaml"
    save_generated_field_registry(output_path, generated)
    loaded = load_yaml(output_path)

    assert loaded["generated"] is True
    assert loaded["fields"][0]["field"] == "training.learning_rate"
