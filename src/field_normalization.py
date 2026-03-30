from __future__ import annotations

from typing import Any


def normalize_legged_locomotion_fields(
    raw_discovery: dict[str, Any],
) -> dict[str, Any]:
    candidate_fields = raw_discovery.get("candidate_fields", [])
    normalized_fields = []
    unmapped_source_fields = []
    absent_families = []

    for family in raw_discovery.get("absent_field_families", []):
        if family in {"command_ranges", "command"}:
            absent_families.append("command")
        else:
            absent_families.append(family)

    for entry in candidate_fields:
        source_field = entry["field"]
        normalized = _normalize_source_field(entry)
        if normalized is None:
            unmapped_source_fields.append(
                {
                    "source_field": source_field,
                    "reason": "no_stable_legged_locomotion_mapping",
                }
            )
            continue
        normalized_fields.append(normalized)

    return {
        "project": raw_discovery.get("project", {}),
        "normalized_fields": normalized_fields,
        "not_applicable_families": absent_families,
        "unmapped_source_fields": unmapped_source_fields,
    }


def _normalize_source_field(entry: dict[str, Any]) -> dict[str, Any] | None:
    source_field = entry["field"]
    if source_field == "max_iterations":
        return _build_normalized_entry(
            entry,
            normalized_field="training.max_iterations",
            hydra_path="agent.max_iterations",
            validation_source="agent.yaml",
        )
    if source_field == "scene.num_envs":
        return _build_normalized_entry(
            entry,
            normalized_field="training.num_envs",
            hydra_path="env.scene.num_envs",
            validation_source="env.yaml",
        )
    if source_field.startswith("algorithm."):
        tail = source_field.split("algorithm.", maxsplit=1)[1]
        if tail in {
            "learning_rate",
            "entropy_coef",
            "clip_param",
            "num_mini_batches",
        }:
            return _build_normalized_entry(
                entry,
                normalized_field=f"training.{tail}",
                hydra_path=f"agent.algorithm.{tail}",
                validation_source="agent.yaml",
            )
        return None
    if source_field.startswith("rewards.") and source_field.endswith(".weight"):
        reward_name = source_field[len("rewards.") :]
        return _build_normalized_entry(
            entry,
            normalized_field=f"reward.{reward_name}",
            hydra_path=f"env.rewards.{reward_name}",
            validation_source="env.yaml",
        )
    if source_field.startswith("commands.base_velocity.ranges."):
        tail = source_field.split("commands.base_velocity.ranges.", maxsplit=1)[1]
        if tail in {"lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"}:
            return _build_normalized_entry(
                entry,
                normalized_field=f"command.{tail}",
                hydra_path=f"env.commands.base_velocity.ranges.{tail}",
                validation_source="env.yaml",
            )
    return None


def _build_normalized_entry(
    entry: dict[str, Any],
    *,
    normalized_field: str,
    hydra_path: str,
    validation_source: str,
) -> dict[str, Any]:
    normalized_layer = normalized_field.split(".", maxsplit=1)[0]
    return {
        "source_field": entry["field"],
        "normalized_field": normalized_field,
        "normalized_layer": normalized_layer,
        "kind": entry.get("kind", "unknown"),
        "source_file": entry.get("source_file"),
        "hydra_path": hydra_path,
        "default_value": entry.get("default_value"),
        "validation_source": validation_source,
        "candidate_for_planner": bool(entry.get("candidate_for_planner", False)),
        "normalization_reason": f"normalized_from_{entry['field']}",
    }
