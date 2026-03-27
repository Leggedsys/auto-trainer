from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from utils import save_yaml


EXACT_ALLOWED_FIELDS = {
    "training.num_envs",
    "training.max_iterations",
    "training.learning_rate",
    "training.entropy_coef",
    "training.clip_param",
    "training.num_mini_batches",
}
ALLOWED_PREFIXES = ("reward.", "command.")
DENIED_PREFIXES = (
    "task.",
    "robot.",
    "observation.",
    "physics.",
    "isaaclab.",
    "python.",
    "code.",
    "path.",
)
NUMERIC_BOUNDS = {
    "training.num_envs": (1, 1024),
    "training.max_iterations": (1, 200),
    "training.learning_rate": (1e-6, 1e-2),
    "training.entropy_coef": (0.0, 0.1),
    "training.clip_param": (0.05, 0.4),
    "training.num_mini_batches": (1, 64),
}


def describe_numeric_bounds() -> dict[str, tuple[float, float]]:
    return NUMERIC_BOUNDS.copy()


def _is_allowed_field(field: str) -> bool:
    if any(field.startswith(prefix) for prefix in DENIED_PREFIXES):
        return False
    if field in EXACT_ALLOWED_FIELDS:
        return True
    return any(field.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _set_nested_value(config: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        next_node = node.get(part)
        if next_node is None:
            node[part] = {}
            next_node = node[part]
        if not isinstance(next_node, dict):
            raise ValueError(f"Cannot set nested field on non-dict path: {field}")
        node = next_node
    node[parts[-1]] = value


def _validate_value_bounds(field: str, value: Any) -> None:
    bounds = NUMERIC_BOUNDS.get(field)
    if bounds is None:
        return
    lower, upper = bounds
    if not isinstance(value, (int, float)):
        raise ValueError(f"Field {field} must be numeric")
    if value < lower or value > upper:
        raise ValueError(
            f"Field {field} out of bounds: {value} not in [{lower}, {upper}]"
        )


def apply_plan_to_config(
    current_config: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    changes = plan.get("changes")
    if not isinstance(changes, list):
        raise ValueError("Planner changes must be a list")
    if len(changes) > 3:
        raise ValueError("Planner changes exceed maximum of 3")

    next_config = deepcopy(current_config)
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Each planner change must be an object")

        field = change.get("field")
        if not isinstance(field, str):
            raise ValueError("Planner change field must be a string")
        if not _is_allowed_field(field):
            raise ValueError(f"Illegal planner field: {field}")
        if "new_value" not in change:
            raise ValueError(f"Planner change missing new_value: {field}")

        _validate_value_bounds(field, change["new_value"])

        _set_nested_value(next_config, field, change["new_value"])

    return next_config


def save_next_config(next_config: dict[str, Any], output_path: Path) -> None:
    save_yaml(output_path, next_config)
