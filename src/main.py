from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from analyst import build_analysis, save_analysis
from llm import build_default_llm_client
from patcher import apply_plan_to_config, describe_numeric_bounds, save_next_config
from planner import plan_next_experiment, save_planner_output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from runner import TrainingRunner
from utils import build_training_config, load_yaml, save_yaml

console = Console()


def print_section_title(title: str) -> None:
    console.print()
    console.print(Panel(Text(title, style="bold cyan"), border_style="bright_blue"))


def print_startup_banner(
    *, rounds: int, config_path: Path, headless: Any, llm_name: str
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")
    table.add_row("Task", "Isaac-Velocity-Flat-Unitree-Go2-v0")
    table.add_row("Rounds", str(rounds))
    table.add_row("Config", str(config_path))
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


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    current_config: dict[str, Any] = load_yaml(config_path)
    if args.headless:
        current_config.setdefault("training", {})["headless"] = True
    if args.num_envs is not None:
        current_config.setdefault("training", {})["num_envs"] = args.num_envs
    if args.max_iterations is not None:
        current_config.setdefault("training", {})["max_iterations"] = (
            args.max_iterations
        )

    runner = TrainingRunner(project_root=project_root)
    planner_prompt_path = project_root / "prompts" / "planner.txt"
    llm_client = build_default_llm_client()
    previous_summary: dict[str, Any] | None = None
    round_summaries: list[dict[str, Any]] = []
    active_run_dir: Path | None = None

    print_startup_banner(
        rounds=args.rounds,
        config_path=config_path,
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
