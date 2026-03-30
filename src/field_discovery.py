from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm import LLMClient


ALLOWED_ABSENT_FIELD_FAMILIES = {
    "training",
    "command",
    "reward",
    "command_ranges",
}


def build_field_discovery_prompt(
    *,
    task_id: str,
    task_type: str,
    source_snippets: list[dict[str, str]],
) -> str:
    sections = []
    for snippet in source_snippets:
        sections.append(
            f"[{snippet['label']}]\npath: {snippet['path']}\n{snippet['content']}"
        )

    return (
        "You are analyzing an Isaac Lab task project for auto-trainer onboarding.\n\n"
        "Identify candidate planner-editable fields from the provided source files and config dumps.\n"
        "Return JSON only.\n\n"
        "Rules:\n"
        "- Identify only fields that are explicitly defined in the provided files.\n"
        "- Prefer trainer hyperparameters, reward weights, and command ranges.\n"
        "- If a field family is absent, do not invent fields from that family.\n"
        "- Do not infer hidden fields that are not visible in the provided code or dumped configs.\n"
        "- Use task-specific field names from the source, not pre-normalized platform names.\n\n"
        "- absent_field_families may only contain values from this set: training, command, reward, command_ranges.\n\n"
        "Return this exact JSON shape:\n"
        "{\n"
        '  "project": {"task_id": "string", "task_type": "string"},\n'
        '  "candidate_fields": [\n'
        "    {\n"
        '      "field": "string",\n'
        '      "layer": "training|reward|command|other",\n'
        '      "kind": "scalar|range|boolean|enum|unknown",\n'
        '      "source_file": "string",\n'
        '      "source_symbol": "string",\n'
        '      "default_value": "any JSON value",\n'
        '      "hydra_path": "string or null",\n'
        '      "candidate_for_planner": true,\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ],\n"
        '  "absent_field_families": ["string"]\n'
        "}\n\n"
        f"task_id: {task_id}\n"
        f"task_type: {task_type}\n\n"
        "Inputs:\n\n" + "\n\n".join(sections)
    )


def discover_fields(
    *,
    task_id: str,
    task_type: str,
    source_snippets: list[dict[str, str]],
    llm_client: LLMClient,
) -> dict[str, Any]:
    prompt = build_field_discovery_prompt(
        task_id=task_id,
        task_type=task_type,
        source_snippets=source_snippets,
    )
    response = llm_client.generate(prompt)
    try:
        payload = json.loads(_extract_json_object(response))
    except Exception as exc:
        raise ValueError(f"Field discovery response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Field discovery response must be a JSON object")
    if not isinstance(payload.get("candidate_fields"), list):
        raise ValueError("Field discovery response must include candidate_fields list")
    absent_field_families = payload.get("absent_field_families", [])
    if not isinstance(absent_field_families, list):
        raise ValueError(
            "Field discovery response must include absent_field_families list"
        )
    for family in absent_field_families:
        if family not in ALLOWED_ABSENT_FIELD_FAMILIES:
            raise ValueError(f"Unsupported absent field family: {family}")
    _stabilize_legged_locomotion_discovery(payload)
    return payload


def load_source_snippet(path: Path, label: str) -> dict[str, str]:
    return {
        "label": label,
        "path": str(path),
        "content": path.read_text(encoding="utf-8"),
    }


def _extract_json_object(response: str) -> str:
    stripped = response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Field discovery response did not contain a JSON object")
    return stripped[start : end + 1]


def _stabilize_legged_locomotion_discovery(payload: dict[str, Any]) -> None:
    candidate_fields = payload.get("candidate_fields", [])
    absent_field_families = payload.get("absent_field_families", [])

    if any(
        entry.get("field") == "max_iterations"
        or str(entry.get("field", "")).startswith("algorithm.")
        for entry in candidate_fields
    ):
        payload["absent_field_families"] = [
            family for family in absent_field_families if family != "training"
        ]

    if any(
        str(entry.get("field", "")).startswith("commands.base_velocity.ranges.")
        for entry in candidate_fields
    ):
        payload["absent_field_families"] = [
            family
            for family in payload.get("absent_field_families", [])
            if family not in {"command", "command_ranges"}
        ]

    if any(
        str(entry.get("field", "")).startswith("rewards.") for entry in candidate_fields
    ):
        payload["absent_field_families"] = [
            family
            for family in payload.get("absent_field_families", [])
            if family != "reward"
        ]
