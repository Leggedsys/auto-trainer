from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_value(value: Any) -> str:
    return "null" if value is None else str(value)


def _is_better(summary: dict[str, Any], previous_summary: dict[str, Any] | None) -> str:
    if previous_summary is None:
        return "No previous round is available, so this round is the baseline."

    current_reward = summary.get("mean_reward")
    previous_reward = previous_summary.get("mean_reward")
    current_length = summary.get("mean_episode_length")
    previous_length = previous_summary.get("mean_episode_length")

    if isinstance(current_reward, (int, float)) and isinstance(
        previous_reward, (int, float)
    ):
        if current_reward > previous_reward:
            return (
                f"Yes. Mean reward improved from {previous_reward} to {current_reward}."
            )
        if current_reward < previous_reward:
            return (
                f"No. Mean reward dropped from {previous_reward} to {current_reward}."
            )

    if isinstance(current_length, (int, float)) and isinstance(
        previous_length, (int, float)
    ):
        if current_length > previous_length:
            return (
                "Mixed. Reward is inconclusive, but mean episode length increased from "
                f"{previous_length} to {current_length}."
            )
        if current_length < previous_length:
            return (
                "Mixed. Reward is inconclusive, but mean episode length decreased from "
                f"{previous_length} to {current_length}."
            )

    return "Unclear. The key metrics do not show a decisive change."


def _largest_problem(summary: dict[str, Any]) -> str:
    mean_reward = summary.get("mean_reward")
    episode_length = summary.get("mean_episode_length")
    value_loss = summary.get("value_loss")

    if isinstance(mean_reward, (int, float)) and mean_reward < 0:
        return f"The largest problem is that mean reward is still negative at {mean_reward}."
    if isinstance(episode_length, (int, float)) and episode_length < 200:
        return f"The largest problem is short episode length at {episode_length}."
    if isinstance(value_loss, (int, float)) and value_loss > 0.05:
        return f"The largest problem is elevated value loss at {value_loss}."
    return "The largest problem is not obvious from the current summary alone."


def _suggested_direction(
    summary: dict[str, Any], previous_summary: dict[str, Any] | None
) -> str:
    mean_reward = summary.get("mean_reward")
    episode_length = summary.get("mean_episode_length")

    if previous_summary is None:
        return "Use this round as the baseline, then change only one or two training knobs at a time."
    if isinstance(mean_reward, (int, float)) and mean_reward < 0:
        return "Focus on safer optimizer and rollout-size adjustments that may improve reward without changing the task or robot."
    if isinstance(episode_length, (int, float)) and episode_length >= 1000:
        return "Episode length looks saturated, so focus on improving reward quality rather than only extending training horizon."
    return "Keep changes small and continue comparing reward, episode length, and losses across rounds."


def build_analysis(
    summary: dict[str, Any], previous_summary: dict[str, Any] | None
) -> str:
    lines = [
        "# Run Analysis",
        "",
        "## Is This Round Better?",
        _is_better(summary, previous_summary),
        "",
        "## Biggest Problem",
        _largest_problem(summary),
        "",
        "## Suggested Direction",
        _suggested_direction(summary, previous_summary),
        "",
        "## Metrics",
        f"- mean_reward: {_format_value(summary.get('mean_reward'))}",
        f"- mean_episode_length: {_format_value(summary.get('mean_episode_length'))}",
        f"- value_loss: {_format_value(summary.get('value_loss'))}",
        f"- policy_loss: {_format_value(summary.get('policy_loss'))}",
        f"- entropy: {_format_value(summary.get('entropy'))}",
        f"- iteration: {_format_value(summary.get('iteration'))}",
    ]
    return "\n".join(lines) + "\n"


def save_analysis(output_path: Path, analysis: str) -> None:
    output_path.write_text(analysis, encoding="utf-8")
