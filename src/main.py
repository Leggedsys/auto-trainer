from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from analyst import build_analysis, save_analysis
from llm import build_default_llm_client
from patcher import apply_plan_to_config, save_next_config
from planner import plan_next_experiment, save_planner_output
from runner import TrainingRunner
from utils import build_training_config, load_yaml, save_yaml


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"


def use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in {None, "dumb"}


def style(text: str, *codes: str) -> str:
    if not use_color() or not codes:
        return text
    return "".join(codes) + text + RESET


class Spinner:
    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = [".", "..", "...", "...."]

    def __enter__(self) -> "Spinner":
        if not sys.stdout.isatty():
            print(f"{style('>', CYAN, BOLD)} {self.message}")
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if sys.stdout.isatty():
            print("\r" + " " * 100 + "\r", end="")

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            frame = self._frames[index % len(self._frames)]
            print(
                f"\r{style('>', CYAN, BOLD)} {self.message} {style(frame, CYAN)}",
                end="",
                flush=True,
            )
            index += 1
            time.sleep(0.4)


def print_section_title(title: str) -> None:
    line = style("=" * 60, BLUE)
    print(line)
    print(style(title, BOLD, CYAN))
    print(line)


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
    print(f"{style('Run ID:', BOLD)}        {run_id}")
    print(f"{style('Run Dir:', BOLD)}       {run_dir}")
    print(
        f"{style('Return Code:', BOLD)}   {style(str(return_code), GREEN if return_code == 0 else RED, BOLD)}"
    )
    print()

    print(style("Result", BOLD, MAGENTA))
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
        print(f"{style('•', CYAN)} {label:<12} {value}")
    print()

    print(style("Plan", BOLD, MAGENTA))
    print(f"{style('•', CYAN)} hypothesis     {planner['hypothesis']}")
    if planner["changes"]:
        for index, change in enumerate(planner["changes"], start=1):
            print(
                f"{style('•', CYAN)} change {index}       {change['field']} {style('->', DIM)} {style(str(change['new_value']), YELLOW, BOLD)}"
            )
            print(f"  {style('reason', DIM)}         {change['reason']}")
    else:
        print(f"{style('•', CYAN)} changes        none")
    print(f"{style('•', CYAN)} expected       {planner['expected_effect']}")
    print()

    print(style("Artifacts", BOLD, MAGENTA))
    print(f"{style('•', CYAN)} config_snapshot  {config_snapshot_path}")
    print(f"{style('•', CYAN)} stdout           {stdout_log_path}")
    print(f"{style('•', CYAN)} stderr           {stderr_log_path}")
    print(f"{style('•', CYAN)} summary          {summary_path}")
    print(f"{style('•', CYAN)} planner_output   {planner_output_path}")
    print(f"{style('•', CYAN)} next_config      {next_config_path}")
    print(f"{style('•', CYAN)} analysis         {analysis_path}")
    print()


def print_experiment_overview(round_summaries: list[dict[str, Any]]) -> None:
    print_section_title("Experiment Overview")
    best_round_index: int | None = None
    best_reward: float | None = None

    for entry in round_summaries:
        reward = entry["summary"].get("mean_reward")
        status = entry["status"]
        status_color = (
            GREEN
            if status == "improved"
            else YELLOW
            if status == "unchanged"
            else RED
            if status == "worse"
            else CYAN
        )
        print(
            f"{style('•', CYAN)} Round {entry['round']}: reward {style(str(reward), BOLD)} {style('->', DIM)} {style(status, status_color, BOLD)}"
        )
        if isinstance(reward, (int, float)) and (
            best_reward is None or reward > best_reward
        ):
            best_reward = float(reward)
            best_round_index = int(entry["round"])

    if best_round_index is not None:
        print(
            f"{style('Best Round:', BOLD)} {style(str(best_round_index), GREEN, BOLD)}"
        )
    else:
        print(f"{style('Best Round:', BOLD)} unavailable")


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
    print()
    print(style("Interrupted by user. Saving interruption artifacts...", RED, BOLD))

    if run_dir is None:
        print(style("No active run directory yet; nothing to persist.", YELLOW, BOLD))
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

    print(
        style(
            f"Saved interrupt status: {run_dir / 'interrupt_status.yaml'}", YELLOW, BOLD
        )
    )
    print(
        style(
            f"Saved interrupt analysis: {run_dir / 'interrupt_analysis.json'}",
            YELLOW,
            BOLD,
        )
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

    print_section_title("Go2 Auto Trainer")
    print(f"{style('Task', BOLD)}        Isaac-Velocity-Flat-Unitree-Go2-v0")
    print(f"{style('Rounds', BOLD)}      {args.rounds}")
    print(f"{style('Config', BOLD)}      {config_path}")
    print(
        f"{style('Headless', BOLD)}    {current_config.get('training', {}).get('headless')}"
    )
    print(f"{style('LLM', BOLD)}         {llm_client.__class__.__name__}")
    print()

    try:
        for round_index in range(args.rounds):
            print(
                f"{style('>', CYAN, BOLD)} Preparing round {style(str(round_index), YELLOW, BOLD)}"
            )
            config = build_training_config(current_config)
            with Spinner(f"Running training for round {round_index}"):
                result = runner.run(config)

            active_run_dir = result.run_dir
            print(style(f"✓ Training finished for round {round_index}", GREEN, BOLD))
            analysis = build_analysis(result.summary, previous_summary)
            analysis_path = result.run_dir / "analysis.md"
            save_analysis(analysis_path, analysis)

            planner_goal = current_config.get("planner", {}).get(
                "goal",
                "Improve training quality using only allowed training configuration changes.",
            )
            with Spinner(f"Generating planner suggestion for round {round_index}"):
                planner = plan_next_experiment(
                    goal=planner_goal,
                    current_config=current_config,
                    summary=result.summary,
                    llm_client=llm_client,
                    prompt_path=planner_prompt_path,
                )

            planner_output_path = result.run_dir / "planner_output.json"
            save_planner_output(planner_output_path, planner)

            with Spinner(f"Applying safe patch for round {round_index}"):
                next_config = apply_plan_to_config(current_config, planner)
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
