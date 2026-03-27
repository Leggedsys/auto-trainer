# Go2 Auto Trainer

Minimal external controller for running 3 rounds of Isaac Lab Go2 training experiments without modifying the Isaac Lab repository.

## Scope

This project is intentionally narrow.

- supports one task: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- runs outside Isaac Lab
- calls Isaac Lab only through shell/subprocess
- runs a 3-round experiment loop
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
  runs/
```

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

## How To Run

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
python src/main.py --config configs/base.yaml --rounds 2
python src/main.py --rounds 1 --headless
python src/main.py --rounds 2 --num-envs 512 --max-iterations 100
```

## Output Per Round

Each round creates a new directory under `runs/`:

```text
runs/<run_id>/
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
