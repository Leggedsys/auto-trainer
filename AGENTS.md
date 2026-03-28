# AGENTS

This repository is a minimal external controller for Go2 training experiments.
Agents should keep changes small, explicit, and aligned with the repository's safety model.

## Mission And Scope

- Purpose: propose and apply tightly constrained configuration changes across training rounds.
- Main flow: run training, parse logs, generate an LLM plan, validate it, patch config, save artifacts.
- Core entry points: `src/main.py`, `src/runner.py`, `src/parser.py`, `src/planner.py`, `src/patcher.py`, `src/llm.py`.
- Prompt asset: `prompts/planner.txt`.
- Base config: `configs/base.yaml`.
- External artifacts live outside this repo under `target.artifact_root`.

## Hard Repository Rules

- Do not modify Isaac Lab source code from this repository.
- Do not change robot, task, observation, or physics definitions unless a human explicitly asks and the repository policy changes.
- Keep the LLM interface constrained to JSON text input/output.
- Prefer editing config, prompt text, validation, parsing, reporting, and orchestration logic rather than widening project scope.
- Preserve the external-project model: this repo controls runs, but artifacts are written outside the repo.

## Existing Local Rules Review

- Existing `AGENTS.md` already restricts the LLM role to experiment planning and safe config changes.
- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.
- If any of those files are added later, merge their guidance into this document rather than contradicting them.

## Environment And Setup

- Python requirement: `>=3.10` from `pyproject.toml`.
- Package install: `pip install -e .`
- Runtime dependencies declared today: `PyYAML`, `openai`, `rich`.
- Default external Isaac Lab root expected by configs: `/root/IsaacLab`.
- Default launcher: `/root/IsaacLab/isaaclab.sh`.
- Default training script: `/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`.

## Build, Lint, And Test Commands

This project currently has packaging metadata, but no dedicated lint config and no checked-in test suite.
Use the commands below as the practical command set for agents.

### Install

```bash
pip install -e .
```

### Build / Package Validation

```bash
python -m pip install -e .
python -m compileall src
```

- `pip install -e .` verifies packaging metadata is valid enough for editable install.
- `python -m compileall src` is the safest lightweight syntax check for all Python modules.

### Lint / Static Checks

There is no configured linter in-repo today. If you need a lightweight correctness check, use:

```bash
python -m compileall src
```

If the user asks for linting and local tools are available, prefer non-destructive checks such as:

```bash
ruff check src
python -m mypy src
```

Only run those when the tool is actually installed or the user asks you to add/configure it.

### Test Commands

Pytest is available in the environment, but the repository currently has no checked-in tests.
Observed result at analysis time: `python -m pytest` collected `0` tests.

```bash
python -m pytest
```

### Single-Test Commands

When tests are added, use standard pytest node selection:

```bash
python -m pytest tests/test_file.py
python -m pytest tests/test_file.py -k test_name
python -m pytest tests/test_file.py::test_name
python -m pytest tests/test_file.py::TestClass::test_name
```

### Run The Program

```bash
python src/main.py --rounds 1
python src/main.py --rounds 1 --headless
python src/main.py --config configs/base.yaml --rounds 2
```

### Initialize External Target Metadata

```bash
python src/main.py --init --project-root /root/IsaacLab --target-id isaaclab_go2 --artifact-root /root/auto-trainer-artifacts/isaaclab_go2
```

### Direct Shell Wrapper

```bash
scripts/train_go2.sh --isaaclab-root /root/IsaacLab --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num-envs 256 --max-iterations 50 --headless
```

## Architecture Notes For Agents

- `src/main.py` orchestrates CLI parsing, init mode, round loop, reporting, retries, and interrupt handling.
- `src/runner.py` creates run directories, snapshots config, launches the shell wrapper, and writes logs.
- `src/parser.py` extracts summary metrics from stdout logs using regex.
- `src/planner.py` builds the LLM prompt, parses returned JSON, and validates planner shape.
- `src/patcher.py` is the main safety gate for allowed fields and numeric bounds.
- `src/analyst.py` writes human-readable round analysis.
- `src/utils.py` handles YAML IO and converts raw config into `TrainingConfig`.
- `src/models.py` uses dataclasses for structured runtime data.

## Code Style Guidelines

The codebase is small and consistent. Match its current style unless the user asks for a broader refactor.

### Imports

- Use `from __future__ import annotations` at the top of Python files.
- Group imports in this order: standard library, third-party, local modules.
- Prefer direct imports of local modules like `from utils import save_yaml` over deep package plumbing.
- Avoid wildcard imports.
- Import only what is used.

### Formatting

- Follow PEP 8 style and the repository's existing layout.
- Use 4-space indentation.
- Keep functions separated by a single blank line block structure as in current files.
- Use double quotes consistently; current codebase standard is double-quoted strings.
- Keep line wrapping readable; the codebase uses parenthesized multiline expressions instead of backslashes.
- Preserve concise doc-free functions unless a block is truly non-obvious.

### Types

- Add type hints for function parameters and return values.
- Use built-in generics like `dict[str, Any]` and `list[dict[str, Any]]`.
- Use `Path` for filesystem paths.
- Use `Protocol` for interfaces where multiple implementations exist, as in `LLMClient`.
- Use `dataclass(slots=True)` for simple structured runtime records.
- Prefer explicit optional types like `Path | None` over untyped `None` flows.

### Naming

- Use `snake_case` for functions, variables, and module-level helpers.
- Use `PascalCase` for classes and dataclasses.
- Use descriptive names that reflect training semantics: `current_config`, `previous_summary`, `artifact_root`.
- Use uppercase names for module-level constants such as regexes, allowed-field sets, and bounds tables.
- Keep CLI option names aligned with config field names when practical, such as `--max-iterations` mapping to `training.max_iterations`.

### Data And Configuration Handling

- Treat YAML config as the source of runtime configuration.
- Preserve key ordering when writing YAML by keeping `sort_keys=False`.
- When updating nested config values, validate them before writing.
- Keep planner output JSON-serializable and human-inspectable.
- Prefer explicit path joins via `Path` instead of string concatenation.

### Error Handling

- Raise `ValueError` for invalid planner payloads, illegal fields, and bad config transitions.
- Raise `RuntimeError` for missing backend configuration or failed external LLM requests.
- Preserve exception chaining with `raise ... from exc` when translating exceptions.
- Fail safely: reject invalid plans rather than partially applying them.
- When possible, write status artifacts that explain failure states, as done for patch rejection and interrupts.
- Do not swallow exceptions silently unless there is a clear fallback path.

### CLI And UX Patterns

- Keep CLI parsing in `argparse`.
- Add new CLI options only when they map to a clear runtime need.
- Continue using `rich` for human-facing console output in the main flow.
- Prefer small helper functions for repeated UI/reporting blocks.
- Keep terminal output informative but compact.

### Subprocess And External Execution

- Use `subprocess.run(..., check=False)` when capturing external training failures as artifacts rather than crashing immediately.
- Pass explicit `cwd` and copied environment variables for reproducible subprocess behavior.
- Write stdout and stderr to files for later inspection.
- Validate external paths before execution when possible.

### LLM And Planning Constraints

- Preserve the contract: planner input is prompt text, output is JSON text.
- Keep planner changes tightly whitelisted.
- Maintain the retry-on-invalid-plan behavior unless there is a strong reason to change it.
- Any widening of allowed planner fields must be reflected in prompt constraints, planner validation, and patcher validation together.
- Prefer deterministic, low-ambiguity prompt wording.

### Testing Guidance For Future Changes

- For parser changes, add focused tests around regex extraction and ANSI stripping.
- For patcher changes, add tests covering allowed fields, denied fields, type errors, and numeric bounds.
- For planner changes, add tests for invalid JSON, invalid shape, and oversized change lists.
- For CLI changes, prefer isolated tests around helper functions before end-to-end subprocess tests.

## Change Boundaries Agents Should Respect

- Safe targets for edits: `src/*.py`, `prompts/planner.txt`, `configs/base.yaml`, docs.
- Higher-risk areas: shell wrapper behavior in `scripts/train_go2.sh`, because it affects external execution.
- Do not introduce hidden side effects in init mode or artifact path handling.
- Do not hardcode secrets or API keys.
- Do not move artifacts back into the repository unless explicitly requested.

## Practical Workflow For Agents

- Read `README.md`, `configs/base.yaml`, and the relevant `src/` modules before editing.
- Prefer minimal patches that keep the current architecture intact.
- After code edits, run at least `python -m compileall src`.
- Run `python -m pytest` if tests exist or if you added tests.
- If no tests exist, say so explicitly in your handoff.
- When behavior changes affect safety validation, describe the exact boundary that changed.

## Memory Protocol

- Treat `plans/memory_policy.yaml` as the canonical memory protocol for planning artifacts.
- Treat `plans/PROJECT_MEMORY.md` as compressed long-term memory and `plans/NEXT_ACTIONS.md` as compressed working memory.
- Treat `plans/worklog/inbox.md` as the raw-log buffer for unstable, process-heavy, or exploratory notes.
- Do not create date-based planning files, iteration logs, roadmap variants, or duplicate summaries unless a human explicitly asks.
- Prefer overwriting and compressing existing planning memory rather than appending new narrative history.
- Only write planning memory that changes future decisions, capability boundaries, implementation priority, or irreversible constraints.
- Delete superseded planning information instead of preserving it as process history.
- If a new planning structure is needed, prefer extending `plans/memory_policy.yaml` over adding another markdown file.
- If information can be cheaply re-derived from the current repository and existing memory, do not store it in planning memory.
- When updating project memory, keep the result concise and decision-oriented rather than explanatory or chronological.

## Worklog And Context Protocol

- Before substantial planning or implementation, read memory according to `plans/memory_policy.yaml` rather than loading all planning files by default.
- Select one of two context views before LLM-heavy work: `project_view` for repo-wide direction and `task_view` for the current concrete task.
- Use `project_view` when changing architecture, roadmap, capability boundaries, or long-term priorities.
- Use `task_view` when solving a specific implementation task, validation issue, bug, or local refactor.
- Do not inject the full raw worklog into context unless unresolved task-local information is needed.
- Write unstable intermediate notes to `plans/worklog/inbox.md` first; only compress them into long-term or working memory if they survive the retention rules.
- When the worklog becomes large or threads are resolved, compress it and delete stale entries instead of preserving a process log.

## Current Test Status Snapshot

- No `tests/` directory was present during analysis.
- No `pytest.ini`, `tox.ini`, `Makefile`, or `package.json` was present.
- `python -m pytest` currently reports that zero tests are collected.
- Agents should not assume a richer test harness exists until it is added.
