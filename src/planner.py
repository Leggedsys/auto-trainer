from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm import LLMClient
from patcher import describe_numeric_bounds


ALLOWED_FIELDS = {
    "training.num_envs",
    "training.max_iterations",
    "training.headless",
}


def load_planner_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_planner_prompt(
    *,
    template: str,
    goal: str,
    current_config: dict[str, Any],
    summary: dict[str, Any],
    feedback: str,
) -> str:
    bounds_lines = []
    for field, (lower, upper) in describe_numeric_bounds().items():
        bounds_lines.append(f"- {field}: [{lower}, {upper}]")
    return template.format(
        goal=goal,
        current_config=json.dumps(current_config, indent=2, sort_keys=True),
        summary=json.dumps(summary, indent=2, sort_keys=True),
        numeric_bounds="\n".join(bounds_lines),
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
    feedback: str = "None",
) -> dict[str, Any]:
    template = load_planner_prompt(prompt_path)
    prompt = build_planner_prompt(
        template=template,
        goal=goal,
        current_config=current_config,
        summary=summary,
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
