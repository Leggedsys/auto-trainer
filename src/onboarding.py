from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from field_discovery import discover_fields
from field_normalization import normalize_legged_locomotion_fields
from llm import LLMClient
from utils import load_yaml, save_yaml


def load_raw_discovery(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def build_normalized_candidate_list(raw_discovery: dict[str, Any]) -> dict[str, Any]:
    return normalize_legged_locomotion_fields(raw_discovery)


def save_normalized_candidate_list(path: Path, normalized: dict[str, Any]) -> None:
    save_yaml(path, normalized)


def update_capability_package_with_normalized_fields(
    capability_package_dir: Path,
    normalized: dict[str, Any],
) -> None:
    save_yaml(capability_package_dir / "normalized_candidate_list.yaml", normalized)


def build_generated_field_registry(normalized: dict[str, Any]) -> dict[str, Any]:
    fields = []
    project = normalized.get("project", {})
    for entry in normalized.get("normalized_fields", []):
        fields.append(
            {
                "field": entry["normalized_field"],
                "layer": entry["normalized_layer"],
                "hydra_path": entry.get("hydra_path"),
                "kind": entry.get("kind", "unknown"),
                "bounds": None,
                "role": "discovered_candidate",
                "description": entry.get(
                    "normalization_reason", "generated from normalized candidate"
                ),
                "effect_direction": {
                    "increase": "unknown",
                    "decrease": "unknown",
                },
                "couplings": [],
                "validation_source": entry.get("validation_source"),
                "enabled_for_planner": bool(entry.get("candidate_for_planner", False)),
                "default_value": entry.get("default_value"),
                "source_field": entry.get("source_field"),
                "source_file": entry.get("source_file"),
            }
        )
    return {
        "version": 1,
        "target_id": project.get("task_id"),
        "task": project.get("task_id"),
        "generated": True,
        "fields": fields,
        "not_applicable_families": normalized.get("not_applicable_families", []),
        "unmapped_source_fields": normalized.get("unmapped_source_fields", []),
    }


def save_generated_field_registry(
    path: Path, generated_registry: dict[str, Any]
) -> None:
    save_yaml(path, generated_registry)


def is_legged_locomotion_task(task_id: str) -> bool:
    return "Velocity" in task_id


def build_legged_locomotion_source_snippets(
    *,
    project_root: Path,
    task_id: str,
    env_cfg_path: Path,
    agent_cfg_path: Path | None,
    registry_init_path: Path | None,
) -> list[dict[str, str]]:
    snippets = []
    if registry_init_path is not None and registry_init_path.exists():
        snippets.append(
            {
                "label": "task registration",
                "path": str(registry_init_path),
                "content": registry_init_path.read_text(encoding="utf-8"),
            }
        )
    snippets.append(
        {
            "label": "env cfg",
            "path": str(env_cfg_path),
            "content": env_cfg_path.read_text(encoding="utf-8"),
        }
    )
    if agent_cfg_path is not None and agent_cfg_path.exists():
        snippets.append(
            {
                "label": "agent cfg",
                "path": str(agent_cfg_path),
                "content": agent_cfg_path.read_text(encoding="utf-8"),
            }
        )

    shared_velocity_cfg = Path(
        "/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
    )
    if shared_velocity_cfg.exists():
        snippets.append(
            {
                "label": "shared locomotion velocity cfg",
                "path": str(shared_velocity_cfg),
                "content": shared_velocity_cfg.read_text(encoding="utf-8"),
            }
        )
    return snippets


def inject_legged_trainer_fallback_candidates(
    raw_discovery: dict[str, Any],
    agent_cfg_path: Path | None,
) -> dict[str, Any]:
    if agent_cfg_path is None or not agent_cfg_path.exists():
        return raw_discovery

    text = agent_cfg_path.read_text(encoding="utf-8")
    candidate_fields = list(raw_discovery.get("candidate_fields", []))
    existing_fields = {entry.get("field") for entry in candidate_fields}

    field_patterns = {
        "max_iterations": r"max_iterations\s*=\s*([0-9]+)",
        "algorithm.learning_rate": r"learning_rate\s*=\s*([0-9.eE+-]+)",
        "algorithm.entropy_coef": r"entropy_coef\s*=\s*([0-9.eE+-]+)",
        "algorithm.clip_param": r"clip_param\s*=\s*([0-9.eE+-]+)",
        "algorithm.num_mini_batches": r"num_mini_batches\s*=\s*([0-9]+)",
    }

    for field, pattern in field_patterns.items():
        if field in existing_fields:
            continue
        match = re.search(pattern, text)
        if not match:
            continue
        raw_value = match.group(1)
        value: int | float
        if "." in raw_value or "e" in raw_value.lower():
            value = float(raw_value)
        else:
            value = int(raw_value)
        candidate_fields.append(
            {
                "field": field,
                "layer": "training",
                "kind": "scalar",
                "source_file": str(agent_cfg_path),
                "source_symbol": agent_cfg_path.stem,
                "default_value": value,
                "hydra_path": field,
                "candidate_for_planner": True,
                "reason": "trainer_core_field_from_rule_fallback",
            }
        )

    raw_discovery["candidate_fields"] = candidate_fields
    raw_discovery["absent_field_families"] = [
        family
        for family in raw_discovery.get("absent_field_families", [])
        if family != "training"
    ]
    return raw_discovery


def run_field_discovery_and_normalization(
    *,
    task_id: str,
    task_type: str,
    source_snippets: list[dict[str, str]],
    llm_client: LLMClient,
    agent_cfg_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_discovery = discover_fields(
        task_id=task_id,
        task_type=task_type,
        source_snippets=source_snippets,
        llm_client=llm_client,
    )
    raw_discovery = inject_legged_trainer_fallback_candidates(
        raw_discovery,
        agent_cfg_path,
    )
    normalized = build_normalized_candidate_list(raw_discovery)
    return raw_discovery, normalized
