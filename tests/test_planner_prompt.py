from __future__ import annotations

from pathlib import Path

from planner import build_planner_prompt


def test_build_planner_prompt_includes_field_registry_context(tmp_path: Path) -> None:
    registry_path = tmp_path / "field_registry.yaml"
    registry_path.write_text(
        "fields:\n"
        "  - field: reward.track_lin_vel_xy_exp.weight\n"
        "    layer: reward\n"
        "    kind: scalar\n"
        "    role: primary_task_reward\n"
        "    bounds: [0.0, 5.0]\n"
        "    description: primary linear velocity tracking reward\n"
        "    effect_direction:\n"
        "      increase: stronger linear tracking\n"
        "      decrease: weaker linear tracking\n"
        "    couplings: [command.lin_vel_x]\n"
        "    enabled_for_planner: true\n",
        encoding="utf-8",
    )

    prompt = build_planner_prompt(
        template="{field_registry}",
        goal="test goal",
        current_config={"reward": {"track_lin_vel_xy_exp": {"weight": 1.5}}},
        summary={"mean_reward": -1.0},
        field_registry_path=registry_path,
        feedback="None",
    )

    assert "reward.track_lin_vel_xy_exp.weight" in prompt
    assert "primary_task_reward" in prompt
    assert "stronger linear tracking" in prompt
    assert "command.lin_vel_x" in prompt
