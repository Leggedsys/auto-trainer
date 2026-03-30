from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from analyst import build_analysis, save_analysis
from llm import build_default_llm_client
from onboarding import (
    build_generated_field_registry,
    build_legged_locomotion_source_snippets,
    is_legged_locomotion_task,
    run_field_discovery_and_normalization,
    save_generated_field_registry,
    update_capability_package_with_normalized_fields,
)
from patcher import apply_plan_to_config, describe_numeric_bounds, save_next_config
from planner import plan_next_experiment, save_planner_output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from runner import TrainingRunner
from utils import build_training_config, load_yaml, save_yaml

console = Console()


def summarize_field_registry(field_registry: dict[str, Any]) -> dict[str, Any]:
    fields = field_registry.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("field_registry.yaml must contain a list field 'fields'")

    enabled_fields = []
    layers: dict[str, int] = {}
    validation_sources: dict[str, int] = {}
    for entry in fields:
        if not isinstance(entry, dict):
            raise ValueError("Each field registry entry must be an object")
        field = entry.get("field")
        layer = entry.get("layer")
        validation_source = entry.get("validation_source")
        enabled_for_planner = bool(entry.get("enabled_for_planner", False))
        if not isinstance(field, str) or not isinstance(layer, str):
            raise ValueError(
                "Each field registry entry must define string field and layer"
            )
        layers[layer] = layers.get(layer, 0) + 1
        if isinstance(validation_source, str):
            validation_sources[validation_source] = (
                validation_sources.get(validation_source, 0) + 1
            )
        if enabled_for_planner:
            enabled_fields.append(field)

    return {
        "field_count": len(fields),
        "planner_enabled_count": len(enabled_fields),
        "planner_enabled_fields": enabled_fields,
        "layers": layers,
        "validation_sources": validation_sources,
    }


def build_capability_status(
    field_registry: dict[str, Any],
    normalized_candidates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = field_registry.get("fields", [])
    field_status = []
    layer_status: dict[str, dict[str, Any]] = {}
    normalized_lookup = {}
    not_applicable_families = set()
    if normalized_candidates is not None:
        normalized_lookup = {
            entry["normalized_field"]: entry
            for entry in normalized_candidates.get("normalized_fields", [])
        }
        not_applicable_families = set(
            normalized_candidates.get("not_applicable_families", [])
        )

    for entry in fields:
        field = entry["field"]
        layer = entry["layer"]
        hydra_path = entry.get("hydra_path")
        validation_source = entry.get("validation_source")
        planner_enabled = bool(entry.get("enabled_for_planner", False))
        normalized_entry = normalized_lookup.get(field)
        family_not_applicable = layer in not_applicable_families

        status = {
            "field": field,
            "layer": layer,
            "known": True,
            "mapped": (hydra_path is not None)
            and (normalized_entry is not None or normalized_candidates is None),
            "resolvable": (hydra_path is not None)
            and (normalized_entry is not None or normalized_candidates is None),
            "verifiable": isinstance(validation_source, str)
            and (normalized_entry is not None or normalized_candidates is None),
            "planner_enabled": planner_enabled and normalized_entry is not None,
            "family_status": "not_applicable"
            if family_not_applicable
            else "applicable",
            "evidence": {
                "hydra_path": hydra_path,
                "validation_source": validation_source,
                "normalized": normalized_entry is not None,
            },
        }
        field_status.append(status)

        layer_entry = layer_status.setdefault(
            layer,
            {
                "field_count": 0,
                "planner_enabled_count": 0,
                "fully_wired_count": 0,
            },
        )
        layer_entry["field_count"] += 1
        if status["planner_enabled"]:
            layer_entry["planner_enabled_count"] += 1
        if status["mapped"] and status["resolvable"] and status["verifiable"]:
            layer_entry["fully_wired_count"] += 1

    return {
        "fields": field_status,
        "layers": layer_status,
    }


def build_task_policy(current_config: dict[str, Any]) -> dict[str, Any]:
    task = current_config.get("training", {}).get("task")
    return {
        "task_id": task,
        "primary_metric": "mean_reward",
        "secondary_metrics": [
            "mean_episode_length",
            "policy_loss",
            "value_loss",
            "entropy",
        ],
        "plateau_policy": {
            "type": "manual_review_required",
            "note": "Plateau criteria are not yet automatically inferred for new tasks.",
        },
        "stop_conditions": {
            "reward_target": None,
            "manual_stop": True,
            "budget_exhaustion": True,
        },
        "unacceptable_behaviors": [
            "illegal field access",
            "changes without verification path",
            "changes that require Isaac Lab source edits",
        ],
    }


def build_onboarding_report(
    *,
    project_identity: dict[str, Any],
    field_registry_summary: dict[str, Any] | None,
    capability_status: dict[str, Any] | None,
    normalized_candidates: dict[str, Any] | None,
    discovery_error: dict[str, Any] | None,
) -> dict[str, Any]:
    layer_summary = capability_status["layers"] if capability_status is not None else {}
    fully_wired_layers = []
    partial_layers = []
    for layer, stats in layer_summary.items():
        if stats["field_count"] == stats["fully_wired_count"]:
            fully_wired_layers.append(layer)
        else:
            partial_layers.append(layer)

    planner_enabled_fields = (
        field_registry_summary["planner_enabled_fields"]
        if field_registry_summary is not None
        else []
    )
    normalized_count = (
        len(normalized_candidates.get("normalized_fields", []))
        if normalized_candidates is not None
        else 0
    )
    return {
        "project": project_identity,
        "summary": {
            "planner_enabled_field_count": len(planner_enabled_fields),
            "normalized_candidate_count": normalized_count,
            "fully_wired_layers": fully_wired_layers,
            "partially_wired_layers": partial_layers,
        },
        "gaps": {
            "automatic_discovery": "failed"
            if discovery_error is not None
            else "implemented_in_restricted_mode",
            "automatic_resolver_generation": "not_implemented",
            "two_round_acceptance": "not_run_in_init",
        },
        "discovery_error": discovery_error,
        "next_actions": [
            "review capability_status.yaml for field-level wiring status",
            "review field_registry.yaml for semantic coverage",
            "run at least two validation rounds before declaring onboarding complete",
        ],
    }


def print_section_title(title: str) -> None:
    console.print()
    console.print(Panel(Text(title, style="bold cyan"), border_style="bright_blue"))


def print_startup_banner(
    *,
    rounds: int,
    config_path: Path,
    headless: Any,
    llm_name: str,
    target_id: str,
    project_root: Path,
    artifact_root: Path,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")
    table.add_row("Task", "Isaac-Velocity-Flat-Unitree-Go2-v0")
    table.add_row("Rounds", str(rounds))
    table.add_row("Config", str(config_path))
    table.add_row("Target", target_id)
    table.add_row("Project", str(project_root))
    table.add_row("Artifacts", str(artifact_root))
    table.add_row("Headless", str(headless))
    table.add_row("LLM", llm_name)
    console.print(Panel(table, title="Go2 Auto Trainer", border_style="bright_blue"))


def print_bounds_banner() -> None:
    bounds = describe_numeric_bounds()
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Bounds", style="yellow")
    for field, (lower, upper) in bounds.items():
        table.add_row(field, f"[{lower}, {upper}]")
    console.print(
        Panel(table, title="Planner Patch Bounds Enabled", border_style="magenta")
    )


def print_stage(message: str) -> None:
    console.print(f"[bold cyan]>[/bold cyan] {message}")


def print_round_report(
    *,
    round_index: int,
    run_id: str,
    run_dir: Path,
    return_code: int,
    summary: dict[str, Any],
    planner: dict[str, Any],
    config_snapshot_path: Path,
    effective_input_path: Path,
    trainer_override_resolution_path: Path,
    trainer_override_verification_path: Path,
    command_override_resolution_path: Path,
    command_override_verification_path: Path,
    reward_override_resolution_path: Path,
    reward_override_verification_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    summary_path: Path,
    planner_output_path: Path,
    next_config_path: Path,
    analysis_path: Path,
) -> None:
    print_section_title(f"Round {round_index}")

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", justify="right")
    header.add_column(style="white")
    header.add_row("Run ID", run_id)
    header.add_row("Run Dir", str(run_dir))
    header.add_row(
        "Return Code",
        f"[bold {'green' if return_code == 0 else 'red'}]{return_code}[/bold {'green' if return_code == 0 else 'red'}]",
    )
    console.print(Panel(header, border_style="bright_blue"))

    result_table = Table(show_header=False, box=None, padding=(0, 1))
    result_table.add_column(style="cyan")
    result_table.add_column(style="white")
    metric_labels = {
        "mean_reward": "reward",
        "mean_episode_length": "ep_len",
        "value_loss": "value_loss",
        "policy_loss": "policy",
        "entropy": "entropy",
        "iteration": "iter",
    }
    for key, value in summary.items():
        label = metric_labels.get(key, key)
        result_table.add_row(label, str(value))
    console.print(Panel(result_table, title="Result", border_style="green"))

    plan_table = Table(show_header=False, box=None, padding=(0, 1))
    plan_table.add_column(style="cyan")
    plan_table.add_column(style="white")
    plan_table.add_row("hypothesis", planner["hypothesis"])
    if planner["changes"]:
        for index, change in enumerate(planner["changes"], start=1):
            plan_table.add_row(
                f"change {index}", f"{change['field']} -> {change['new_value']}"
            )
            plan_table.add_row(f"reason {index}", change["reason"])
    else:
        plan_table.add_row("changes", "none")
    plan_table.add_row("expected", planner["expected_effect"])
    console.print(Panel(plan_table, title="Plan", border_style="magenta"))

    artifact_table = Table(show_header=False, box=None, padding=(0, 1))
    artifact_table.add_column(style="cyan")
    artifact_table.add_column(style="white")
    artifact_table.add_row("config_snapshot", str(config_snapshot_path))
    artifact_table.add_row("effective_input", str(effective_input_path))
    artifact_table.add_row(
        "trainer_override_resolution", str(trainer_override_resolution_path)
    )
    artifact_table.add_row(
        "trainer_override_verification", str(trainer_override_verification_path)
    )
    artifact_table.add_row(
        "command_override_resolution", str(command_override_resolution_path)
    )
    artifact_table.add_row(
        "command_override_verification", str(command_override_verification_path)
    )
    artifact_table.add_row(
        "reward_override_resolution", str(reward_override_resolution_path)
    )
    artifact_table.add_row(
        "reward_override_verification", str(reward_override_verification_path)
    )
    artifact_table.add_row("stdout", str(stdout_log_path))
    artifact_table.add_row("stderr", str(stderr_log_path))
    artifact_table.add_row("summary", str(summary_path))
    artifact_table.add_row("planner_output", str(planner_output_path))
    artifact_table.add_row("next_config", str(next_config_path))
    artifact_table.add_row("analysis", str(analysis_path))
    console.print(Panel(artifact_table, title="Artifacts", border_style="yellow"))


def print_experiment_overview(round_summaries: list[dict[str, Any]]) -> None:
    print_section_title("Experiment Overview")
    best_round_index: int | None = None
    best_reward: float | None = None
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Round", style="cyan")
    table.add_column("Reward", style="white")
    table.add_column("Status", style="white")

    for entry in round_summaries:
        reward = entry["summary"].get("mean_reward")
        status = entry["status"]
        status_style = (
            "bold green"
            if status == "improved"
            else "bold yellow"
            if status == "unchanged"
            else "bold red"
            if status == "worse"
            else "bold cyan"
        )
        table.add_row(
            str(entry["round"]),
            str(reward),
            f"[{status_style}]{status}[/{status_style}]",
        )
        if isinstance(reward, (int, float)) and (
            best_reward is None or reward > best_reward
        ):
            best_reward = float(reward)
            best_round_index = int(entry["round"])

    console.print(table)

    if best_round_index is not None:
        console.print(
            Panel(
                f"[bold green]{best_round_index}[/bold green]",
                title="Best Round",
                border_style="green",
            )
        )
    else:
        console.print(Panel("unavailable", title="Best Round", border_style="red"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Go2 automatic training rounds")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of experiment rounds to run",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless training regardless of config file",
    )
    parser.add_argument("--num-envs", type=int, help="Override training.num_envs")
    parser.add_argument(
        "--max-iterations", type=int, help="Override training.max_iterations"
    )
    parser.add_argument("--init", action="store_true", help="Initialize target paths")
    parser.add_argument("--target-id", help="Target identifier for init mode")
    parser.add_argument("--project-root", help="Target project root for init mode")
    parser.add_argument(
        "--artifact-root", help="External artifact root for init mode or override"
    )
    parser.add_argument(
        "--task-registry-path",
        help="Task registry __init__.py path for restricted onboarding init",
    )
    parser.add_argument(
        "--env-cfg-path",
        help="Environment config path for restricted onboarding init",
    )
    parser.add_argument(
        "--agent-cfg-path",
        help="Agent config path for restricted onboarding init",
    )
    return parser.parse_args()


def save_status_file(run_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    save_yaml(run_dir / filename, payload)


def save_patch_error(
    run_dir: Path,
    round_index: int,
    planner: dict[str, Any],
    error: str,
    planner_feedback: str,
) -> None:
    save_status_file(
        run_dir,
        "patch_error.yaml",
        {
            "status": "patch_rejected",
            "round": round_index,
            "error": error,
            "planner_feedback": planner_feedback,
            "planner": planner,
        },
    )


def build_interrupt_analysis_prompt(
    round_index: int,
    current_config: dict[str, Any],
    previous_summary: dict[str, Any] | None,
    partial_summary: dict[str, Any] | None,
) -> str:
    return (
        "You are analyzing an interrupted Go2 training experiment. Return JSON only with keys "
        '"what_happened", "risk", and "next_step". '
        f"Round: {round_index}\n"
        f"Current config: {json.dumps(current_config, indent=2, sort_keys=True)}\n"
        f"Previous summary: {json.dumps(previous_summary, indent=2, sort_keys=True) if previous_summary is not None else 'null'}\n"
        f"Partial summary: {json.dumps(partial_summary, indent=2, sort_keys=True) if partial_summary is not None else 'null'}"
    )


def handle_interrupt(
    *,
    run_dir: Path | None,
    round_index: int,
    current_config: dict[str, Any],
    previous_summary: dict[str, Any] | None,
    llm_client: Any,
) -> None:
    console.print()
    console.print(
        Panel(
            "Interrupted by user. Saving interruption artifacts...",
            title="Interrupted",
            border_style="red",
        )
    )

    if run_dir is None:
        console.print(
            "[bold yellow]No active run directory yet; nothing to persist.[/bold yellow]"
        )
        return

    partial_summary_path = run_dir / "summary.json"
    partial_summary = None
    if partial_summary_path.exists():
        try:
            partial_summary = load_yaml(partial_summary_path)
        except Exception:
            partial_summary = None

    save_status_file(
        run_dir,
        "interrupt_status.yaml",
        {
            "status": "interrupted",
            "round": round_index,
            "message": "Experiment interrupted by user",
        },
    )

    try:
        prompt = build_interrupt_analysis_prompt(
            round_index=round_index,
            current_config=current_config,
            previous_summary=previous_summary,
            partial_summary=partial_summary,
        )
        response_text = llm_client.generate(prompt)
        analysis_payload = json.loads(response_text)
    except Exception as exc:
        analysis_payload = {
            "what_happened": "Experiment interrupted by user before the round completed.",
            "risk": f"Could not get LLM interruption analysis: {exc}",
            "next_step": "Review stdout.log and stderr.log, then restart from the last complete run.",
        }

    with (run_dir / "interrupt_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(analysis_payload, file, indent=2)
        file.write("\n")

    console.print(
        f"[bold yellow]Saved interrupt status:[/bold yellow] {run_dir / 'interrupt_status.yaml'}"
    )
    console.print(
        f"[bold yellow]Saved interrupt analysis:[/bold yellow] {run_dir / 'interrupt_analysis.json'}"
    )


def run_init_mode(
    config_path: Path, current_config: dict[str, Any], args: argparse.Namespace
) -> None:
    target = current_config.setdefault("target", {})
    target_id = args.target_id or target.get("id") or "isaaclab_go2"
    project_root = args.project_root or target.get("project_root")
    if not project_root:
        raise ValueError("Init mode requires --project-root or target.project_root")
    artifact_root = args.artifact_root or target.get(
        "artifact_root", f"/root/auto-trainer-artifacts/{target_id}"
    )

    project_root_path = Path(project_root)
    if not project_root_path.exists():
        raise ValueError(f"Project root does not exist: {project_root}")

    target["id"] = target_id
    target["project_root"] = str(project_root_path)
    target["artifact_root"] = artifact_root
    if args.task_registry_path:
        target["task_registry_path"] = args.task_registry_path
    if args.env_cfg_path:
        target["env_cfg_path"] = args.env_cfg_path
    if args.agent_cfg_path:
        target["agent_cfg_path"] = args.agent_cfg_path
    save_yaml(config_path, current_config)

    artifact_root_path = Path(artifact_root)
    artifact_root_path.mkdir(parents=True, exist_ok=True)
    capability_package_dir = artifact_root_path / "capability_package"
    capability_package_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    field_registry_path = repo_root / "configs" / "field_registry.yaml"
    generated_field_registry_path = (
        capability_package_dir / "field_registry.generated.yaml"
    )
    field_registry_exists = field_registry_path.exists()
    field_registry_summary = None
    capability_status = None
    normalized_candidates = None
    discovery_error = None
    normalized_candidate_list_path = (
        capability_package_dir / "normalized_candidate_list.yaml"
    )
    if normalized_candidate_list_path.exists():
        normalized_candidates = load_yaml(normalized_candidate_list_path)

    project_identity = {
        "target_id": target_id,
        "project_root": str(project_root_path),
        "task_id": current_config.get("training", {}).get("task"),
        "workspace_root": str(artifact_root_path),
        "artifact_root": artifact_root,
        "isaaclab_root": current_config.get("isaaclab", {}).get("root_dir"),
    }

    task_id = current_config.get("training", {}).get("task")
    env_cfg_entry = target.get("env_cfg_path")
    agent_cfg_entry = target.get("agent_cfg_path")
    registry_entry = target.get("task_registry_path")
    if (
        isinstance(task_id, str)
        and is_legged_locomotion_task(task_id)
        and isinstance(env_cfg_entry, str)
        and Path(env_cfg_entry).exists()
    ):
        source_snippets = build_legged_locomotion_source_snippets(
            project_root=project_root_path,
            task_id=task_id,
            env_cfg_path=Path(env_cfg_entry),
            agent_cfg_path=Path(agent_cfg_entry) if agent_cfg_entry else None,
            registry_init_path=Path(registry_entry) if registry_entry else None,
        )
        try:
            raw_discovery, normalized_candidates = (
                run_field_discovery_and_normalization(
                    task_id=task_id,
                    task_type="ManagerBasedRLEnv",
                    source_snippets=source_snippets,
                    llm_client=build_default_llm_client(),
                    agent_cfg_path=Path(agent_cfg_entry) if agent_cfg_entry else None,
                )
            )
            save_yaml(capability_package_dir / "raw_discovery.yaml", raw_discovery)
            update_capability_package_with_normalized_fields(
                capability_package_dir,
                normalized_candidates,
            )
            generated_field_registry = build_generated_field_registry(
                normalized_candidates
            )
            save_generated_field_registry(
                generated_field_registry_path,
                generated_field_registry,
            )
            discovered_task_id = raw_discovery.get("project", {}).get("task_id")
            if isinstance(discovered_task_id, str) and discovered_task_id:
                project_identity["task_id"] = discovered_task_id
        except Exception as exc:
            discovery_error = {
                "task_id": task_id,
                "error": str(exc),
            }
            save_yaml(capability_package_dir / "discovery_error.yaml", discovery_error)

    active_field_registry = None
    if generated_field_registry_path.exists():
        active_field_registry = load_yaml(generated_field_registry_path)
    elif field_registry_exists:
        active_field_registry = load_yaml(field_registry_path)

    if active_field_registry is not None:
        field_registry_summary = summarize_field_registry(active_field_registry)
        capability_status = build_capability_status(
            active_field_registry,
            normalized_candidates,
        )

    task_policy = build_task_policy(current_config)
    onboarding_report = build_onboarding_report(
        project_identity=project_identity,
        field_registry_summary=field_registry_summary,
        capability_status=capability_status,
        normalized_candidates=normalized_candidates,
        discovery_error=discovery_error,
    )
    save_yaml(capability_package_dir / "project_identity.yaml", project_identity)
    save_yaml(capability_package_dir / "task_policy.yaml", task_policy)
    save_yaml(capability_package_dir / "onboarding_report.yaml", onboarding_report)
    if capability_status is not None:
        save_yaml(capability_package_dir / "capability_status.yaml", capability_status)

    init_info = {
        "target": {
            "id": target_id,
            "project_root": str(project_root_path),
            "artifact_root": artifact_root,
        },
        "isaaclab": current_config.get("isaaclab", {}),
        "training": {
            "task": current_config.get("training", {}).get("task"),
        },
        "discovered": {
            "trainer_repo": str(repo_root),
            "project_root_exists": project_root_path.exists(),
            "isaaclab_root": current_config.get("isaaclab", {}).get("root_dir"),
            "launcher": current_config.get("isaaclab", {}).get("launcher"),
            "train_script": current_config.get("isaaclab", {}).get("train_script"),
            "launcher_exists": Path(
                current_config.get("isaaclab", {}).get("launcher", "")
            ).exists(),
            "train_script_exists": Path(
                current_config.get("isaaclab", {}).get("train_script", "")
            ).exists(),
            "field_registry_path": str(field_registry_path),
            "field_registry_exists": field_registry_exists,
            "capability_package_dir": str(capability_package_dir),
        },
    }
    if field_registry_summary is not None:
        init_info["field_registry"] = field_registry_summary
    if capability_status is not None:
        init_info["capability_status"] = {
            "layer_summary": capability_status["layers"],
        }
    save_yaml(artifact_root_path / "target_info.yaml", init_info)

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="bold cyan", justify="right")
    info_table.add_column(style="white")
    info_table.add_row("Config", str(config_path))
    info_table.add_row("Target", target_id)
    info_table.add_row("Project", str(project_root_path))
    info_table.add_row("Artifacts", artifact_root)
    info_table.add_row("Info File", str(artifact_root_path / "target_info.yaml"))
    info_table.add_row("Capability Dir", str(capability_package_dir))
    if field_registry_summary is not None:
        info_table.add_row(
            "Planner Fields",
            str(field_registry_summary["planner_enabled_count"]),
        )
        info_table.add_row("Field Layers", str(field_registry_summary["layers"]))
    if capability_status is not None:
        info_table.add_row(
            "Capability Layers",
            str(capability_status["layers"]),
        )
    console.print(Panel(info_table, title="Init Complete", border_style="green"))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    current_config: dict[str, Any] = load_yaml(config_path)
    if args.init:
        run_init_mode(config_path, current_config, args)
        return

    if args.headless:
        current_config.setdefault("training", {})["headless"] = True
    if args.num_envs is not None:
        current_config.setdefault("training", {})["num_envs"] = args.num_envs
    if args.max_iterations is not None:
        current_config.setdefault("training", {})["max_iterations"] = (
            args.max_iterations
        )
    if args.artifact_root is not None:
        current_config.setdefault("target", {})["artifact_root"] = args.artifact_root
    if args.project_root is not None:
        current_config.setdefault("target", {})["project_root"] = args.project_root

    initial_config = build_training_config(current_config)
    runner = TrainingRunner(
        project_root=project_root,
        artifact_root=initial_config.artifact_root,
    )
    planner_prompt_path = project_root / "prompts" / "planner.txt"
    llm_client = build_default_llm_client()
    previous_summary: dict[str, Any] | None = None
    round_summaries: list[dict[str, Any]] = []
    active_run_dir: Path | None = None

    print_startup_banner(
        rounds=args.rounds,
        config_path=config_path,
        target_id=initial_config.target_id,
        project_root=initial_config.project_root,
        artifact_root=initial_config.artifact_root,
        headless=current_config.get("training", {}).get("headless"),
        llm_name=llm_client.__class__.__name__,
    )
    print_bounds_banner()

    try:
        for round_index in range(args.rounds):
            print_stage(f"Preparing round {round_index}")
            config = build_training_config(current_config)
            with console.status(
                f"[bold cyan]Running training for round {round_index}[/bold cyan]",
                spinner="dots",
            ):
                result = runner.run(config)

            active_run_dir = result.run_dir
            console.print(
                f"[bold green]✓ Training finished for round {round_index}[/bold green]"
            )
            analysis = build_analysis(result.summary, previous_summary)
            analysis_path = result.run_dir / "analysis.md"
            save_analysis(analysis_path, analysis)

            planner_goal = current_config.get("planner", {}).get(
                "goal",
                "Improve training quality using only allowed training configuration changes.",
            )
            planner_feedback = "None"
            planner = None
            for attempt in range(3):
                with console.status(
                    f"[bold cyan]Generating planner suggestion for round {round_index}[/bold cyan]",
                    spinner="aesthetic",
                ):
                    planner = plan_next_experiment(
                        goal=planner_goal,
                        current_config=current_config,
                        summary=result.summary,
                        llm_client=llm_client,
                        prompt_path=planner_prompt_path,
                        field_registry_path=project_root
                        / "configs"
                        / "field_registry.yaml",
                        feedback=planner_feedback,
                    )

                try:
                    with console.status(
                        f"[bold cyan]Applying safe patch for round {round_index}[/bold cyan]",
                        spinner="line",
                    ):
                        next_config = apply_plan_to_config(current_config, planner)
                    break
                except ValueError as exc:
                    planner_feedback = (
                        "Your previous proposal was rejected. "
                        f"Reason: {exc}. Return a corrected JSON plan that respects all bounds."
                    )
                    console.print(
                        Panel(
                            f"{exc}\nRetrying planner with bound feedback (attempt {min(attempt + 2, 3)}/3).",
                            title="Patch Rejected",
                            border_style="red",
                        )
                    )
                    if attempt == 2:
                        save_patch_error(
                            result.run_dir,
                            round_index,
                            planner,
                            str(exc),
                            planner_feedback,
                        )
                        console.print(
                            Panel(
                                "Round stopped safely. Current run artifacts were preserved.",
                                title="Planner Failed After Retries",
                                border_style="red",
                            )
                        )
                        return

            assert planner is not None

            planner_output_path = result.run_dir / "planner_output.json"
            save_planner_output(planner_output_path, planner)
            next_config_path = result.run_dir / "next_config.yaml"
            save_next_config(next_config, next_config_path)

            status = "next config generated"
            if previous_summary is not None:
                current_reward = result.summary.get("mean_reward")
                previous_reward = previous_summary.get("mean_reward")
                if isinstance(current_reward, (int, float)) and isinstance(
                    previous_reward, (int, float)
                ):
                    if current_reward > previous_reward:
                        status = "improved"
                    elif current_reward < previous_reward:
                        status = "worse"
                    else:
                        status = "unchanged"

            print_round_report(
                round_index=round_index,
                run_id=result.run_id,
                run_dir=result.run_dir,
                return_code=result.return_code,
                summary=result.summary,
                planner=planner,
                config_snapshot_path=result.config_snapshot_path,
                effective_input_path=result.effective_input_path,
                trainer_override_resolution_path=result.trainer_override_resolution_path,
                trainer_override_verification_path=result.trainer_override_verification_path,
                command_override_resolution_path=result.command_override_resolution_path,
                command_override_verification_path=result.command_override_verification_path,
                reward_override_resolution_path=result.reward_override_resolution_path,
                reward_override_verification_path=result.reward_override_verification_path,
                stdout_log_path=result.stdout_log_path,
                stderr_log_path=result.stderr_log_path,
                summary_path=result.summary_path,
                planner_output_path=planner_output_path,
                next_config_path=next_config_path,
                analysis_path=analysis_path,
            )

            round_summaries.append(
                {
                    "round": round_index,
                    "summary": result.summary,
                    "status": status,
                }
            )

            previous_summary = result.summary
            current_config = next_config
    except KeyboardInterrupt:
        handle_interrupt(
            run_dir=active_run_dir,
            round_index=round_index,
            current_config=current_config,
            previous_summary=previous_summary,
            llm_client=llm_client,
        )
        return

    print_experiment_overview(round_summaries)


if __name__ == "__main__":
    main()
