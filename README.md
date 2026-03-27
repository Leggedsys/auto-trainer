# Go2 Auto Trainer

Minimal external controller for running Isaac Lab Go2 training experiments without modifying the Isaac Lab repository.

## Scope

This project is intentionally narrow.

- supports one task: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- runs outside Isaac Lab
- calls Isaac Lab only through shell/subprocess
- runs a 3-round experiment loop
- writes experiment artifacts outside the trainer repository
- saves logs, summaries, planner output, and next-round configs

It is not a general platform.

## Requirements

- Isaac Lab is installed at `/root/IsaacLab`
- launcher exists at `/root/IsaacLab/isaaclab.sh`
- RSL-RL training script exists at `/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`
- Python 3.10+

## Project Layout

```text
go2-auto-trainer/
  README.md
  pyproject.toml
  configs/
    base.yaml
  prompts/
    planner.txt
  scripts/
    train_go2.sh
  src/
    main.py
    runner.py
    parser.py
    planner.py
    patcher.py
    analyst.py
    llm.py
    models.py
    utils.py
```

Artifacts are stored outside this repository under the configured target artifact root.

## Install

From the project root:

```bash
pip install -e .
```

## How It Works

The loop is fixed to 3 rounds:

1. round 0 uses `configs/base.yaml`
2. training logs are parsed into `summary.json`
3. planner proposes up to 3 config changes
4. patcher safely applies allowed changes into `next_config.yaml`
5. next round trains from that generated config

The trainer repository stores code only. Run directories are written to the configured external artifact root for the target.

## How To Run

Optional first step: initialize the target project path, target id, and external artifact root.

```bash
python src/main.py --init \
  --project-root /root/IsaacLab \
  --target-id isaaclab_go2 \
  --artifact-root /root/auto-trainer-artifacts/isaaclab_go2
```

This updates `configs/base.yaml` and writes a small `target_info.yaml` file under the artifact root.
The project root is selected explicitly so the trainer knows which external project it is attached to.
The init step also performs a few minimal path checks and records them in `target_info.yaml`.

Preferred: use DeepSeek with an API key:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
python src/main.py --rounds 3
```

DeepSeek is called through its OpenAI-compatible chat completion API.

Fallback: use a mocked planner response through an environment variable:

```bash
export GO2_AUTO_TRAINER_LLM_RESPONSE='{"hypothesis":"Increase rollout scale slightly.","changes":[{"field":"training.num_envs","new_value":512,"reason":"Try higher sample throughput."},{"field":"training.max_iterations","new_value":75,"reason":"Give training a bit more horizon."}],"expected_effect":"May improve reward and stability."}'
python src/main.py --rounds 3
```

This will automatically run 3 rounds in sequence.

Useful CLI options:

```bash
python src/main.py --init --project-root /root/IsaacLab --target-id isaaclab_go2 --artifact-root /root/auto-trainer-artifacts/isaaclab_go2
python src/main.py --config configs/base.yaml --rounds 2
python src/main.py --rounds 1 --headless
python src/main.py --rounds 2 --num-envs 512 --max-iterations 100
```

## Output Per Round

Each round creates a new directory under `<artifact_root>/runs/`:

```text
<artifact_root>/runs/<run_id>/
  config_snapshot.yaml
  stdout.log
  stderr.log
  summary.json
  planner_output.json
  next_config.yaml
  analysis.md
```

Important files:

- `summary.json`: parsed training metrics
- `planner_output.json`: planner suggestion in JSON
- `next_config.yaml`: config used for the next round
- `analysis.md`: human-readable comparison and notes

## Target Configuration

`configs/base.yaml` now contains a target section:

```yaml
target:
  id: isaaclab_go2
  project_root: /root/IsaacLab
  artifact_root: /root/auto-trainer-artifacts/isaaclab_go2
```

- `target.id` identifies the external project/task pairing
- `target.project_root` identifies the external project location
- `target.artifact_root` is where runs and other experiment artifacts are written

This follows the same general idea as Isaac Lab external project usage: keep the trainer code separate from the experiment outputs, and attach the trainer to an explicit external project root.

## Allowed Planner Changes

The patcher only allows these fields:

- `training.num_envs`
- `training.max_iterations`
- `training.learning_rate`
- `training.entropy_coef`
- `training.clip_param`
- `training.num_mini_batches`
- `reward.*`
- `command.*`

It rejects changes to task identity, robot settings, observation settings, physics settings, code paths, and Isaac Lab paths.

Training parameter safety bounds are also enforced for planner-generated patches:

- `training.num_envs`: 1 to 1024
- `training.max_iterations`: 1 to 200
- `training.learning_rate`: 1e-6 to 1e-2
- `training.entropy_coef`: 0.0 to 0.1
- `training.clip_param`: 0.05 to 0.4
- `training.num_mini_batches`: 1 to 64

Manual CLI overrides such as `--num-envs` and `--max-iterations` are still allowed.

## Interrupt Behavior

If you stop the run with `Ctrl+C`, the program will:

- print an interrupt message in the terminal
- save `interrupt_status.yaml` in the active run directory
- ask the LLM for an interruption analysis and save it as `interrupt_analysis.json`
- keep all completed runs intact

## Isaac Lab Path Notes

This project stays outside Isaac Lab and does not modify Isaac Lab source code.

Default paths:

- Isaac Lab root: `/root/IsaacLab`
- launcher: `/root/IsaacLab/isaaclab.sh`
- training script: `/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`

If your Isaac Lab installation is elsewhere, update `configs/base.yaml`.

## Direct Shell Usage

You can run the wrapper directly for a single training launch:

```bash
scripts/train_go2.sh \
  --isaaclab-root /root/IsaacLab \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num-envs 256 \
  --max-iterations 50 \
  --headless
```

## TODO / Future Work

The current project is intentionally minimal. To become a more complete automatic trainer, the following work is still needed:

- checkpoint management
  - record checkpoint paths for every run
  - maintain a best-checkpoint table
  - support continuing from the best checkpoint instead of always starting a fresh run
- one-time repository initialization
  - inspect the target Isaac Lab task once at startup
  - discover log locations, checkpoint locations, and important config entry points
- signal selection
  - define which training outputs matter most for decision making
  - separate useful long-horizon trends from noisy per-iteration values
- context compression
  - summarize past rounds before sending context to the LLM
  - keep only the most relevant config deltas, best results, and failure reasons
- recovery and resume
  - resume from the last complete run after interruption
  - decide whether to restart from scratch or continue from a stored checkpoint
- experiment indexing
  - maintain a compact table linking run -> config -> summary -> planner output -> checkpoint
  - make it easy to inspect the best run and the latest resumable state
