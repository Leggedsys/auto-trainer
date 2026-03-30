from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm import LLMClient
from patcher import describe_numeric_bounds
from utils import load_yaml


ALLOWED_FIELDS = {
    "training.num_envs",
    "training.max_iterations",
    "training.learning_rate",
    "training.entropy_coef",
    "training.clip_param",
    "training.num_mini_batches",
    "command.lin_vel_x",
    "command.lin_vel_y",
    "command.ang_vel_z",
    "command.heading",
    "reward.track_lin_vel_xy_exp.weight",
    "reward.track_ang_vel_z_exp.weight",
    "reward.feet_air_time.weight",
    "reward.flat_orientation_l2.weight",
}


def load_planner_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_planner_prompt(
    *,
    template: str,
    goal: str,
    current_config: dict[str, Any],
    summary: dict[str, Any],
    field_registry_path: Path,
    feedback: str,
) -> str:
    bounds_lines = []
    for field, (lower, upper) in describe_numeric_bounds().items():
        bounds_lines.append(f"- {field}: [{lower}, {upper}]")
    field_registry = load_yaml(field_registry_path)
    field_lines = []
    for entry in field_registry.get("fields", []):
        if entry.get("enabled_for_planner"):
            field_lines.append(
                "- {field} | layer={layer} | kind={kind} | role={role} | bounds={bounds} | "
                "description={description} | increase={increase} | decrease={decrease} | couplings={couplings}".format(
                    field=entry["field"],
                    layer=entry["layer"],
                    kind=entry["kind"],
                    role=entry["role"],
                    bounds=entry["bounds"],
                    description=entry["description"],
                    increase=entry["effect_direction"]["increase"],
                    decrease=entry["effect_direction"]["decrease"],
                    couplings=entry["couplings"],
                )
            )
    return template.format(
        goal=goal,
        current_config=json.dumps(current_config, indent=2, sort_keys=True),
        summary=json.dumps(summary, indent=2, sort_keys=True),
        numeric_bounds="\n".join(bounds_lines),
        field_registry="\n".join(field_lines),
        feedback=feedback,
    )


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    hypothesis = plan.get("hypothesis")
    changes = plan.get("changes")
    expected_effect = plan.get("expected_effect")

    if not isinstance(hypothesis, str):
        raise ValueError("Planner output missing string field: hypothesis")
    if not isinstance(expected_effect, str):
        raise ValueError("Planner output missing string field: expected_effect")
    if not isinstance(changes, list):
        raise ValueError("Planner output missing list field: changes")
    if len(changes) > 3:
        raise ValueError("Planner output exceeds maximum of 3 changes")

    validated_changes: list[dict[str, Any]] = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Each planner change must be a JSON object")

        field = change.get("field")
        reason = change.get("reason")
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"Planner proposed non-whitelisted field: {field}")
        if "new_value" not in change:
            raise ValueError(f"Planner change missing new_value for field: {field}")
        if not isinstance(reason, str):
            raise ValueError(f"Planner change missing string reason for field: {field}")

        validated_changes.append(
            {
                "field": field,
                "new_value": change["new_value"],
                "reason": reason,
            }
        )

    return {
        "hypothesis": hypothesis,
        "changes": validated_changes,
        "expected_effect": expected_effect,
    }


def plan_next_experiment(
    *,
    goal: str,
    current_config: dict[str, Any],
    summary: dict[str, Any],
    llm_client: LLMClient,
    prompt_path: Path,
    field_registry_path: Path,
    feedback: str = "None",
) -> dict[str, Any]:
    template = load_planner_prompt(prompt_path)
    prompt = build_planner_prompt(
        template=template,
        goal=goal,
        current_config=current_config,
        summary=summary,
        field_registry_path=field_registry_path,
        feedback=feedback,
    )
    response_text = llm_client.generate(prompt)
    try:
        plan = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner LLM output is not valid JSON. Raw output: {response_text[:500]}"
        ) from exc
    return _validate_plan(plan)


def save_planner_output(output_path: Path, plan: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")
