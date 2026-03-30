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

It should now be treated as the project onboarding step, not just a path initializer. In addition to runtime paths, the project also carries a field registry at `configs/field_registry.yaml` describing the current planner-editable fields, their roles, couplings, bounds, and validation sources.

The init output `target_info.yaml` now also includes a field-registry summary so you can quickly inspect how many fields are planner-enabled, which layers are currently integrated, and which artifacts validate them.

The init step now also creates a first capability package under `<artifact_root>/capability_package/`, including:

- `project_identity.yaml`
- `capability_status.yaml`
- `task_policy.yaml`
- `onboarding_report.yaml`

This is still the first 1B milestone: it summarizes current field wiring status, but it does not yet auto-discover and auto-wire arbitrary new tasks.

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

The planner and patcher currently allow these fields:

- `training.num_envs`
- `training.max_iterations`
- `training.learning_rate`
- `training.entropy_coef`
- `training.clip_param`
- `training.num_mini_batches`
- `command.lin_vel_x`
- `command.lin_vel_y`
- `command.ang_vel_z`
- `command.heading`
- `reward.track_lin_vel_xy_exp.weight`
- `reward.track_ang_vel_z_exp.weight`
- `reward.feet_air_time.weight`
- `reward.flat_orientation_l2.weight`

It rejects changes to task identity, robot settings, observation settings, physics settings, code paths, and Isaac Lab paths.

Training parameter safety bounds are also enforced for planner-generated patches:

- `training.num_envs`: 1 to 1024
- `training.max_iterations`: 1 to 200
- `training.learning_rate`: 1e-6 to 1e-2
- `training.entropy_coef`: 0.0 to 0.1
- `training.clip_param`: 0.05 to 0.4
- `training.num_mini_batches`: 1 to 64
- `command.lin_vel_x`: [-1.5, 1.5]
- `command.lin_vel_y`: [-1.5, 1.5]
- `command.ang_vel_z`: [-2.0, 2.0]
- `command.heading`: [-pi, pi]
- `reward.track_lin_vel_xy_exp.weight`: 0.0 to 5.0
- `reward.track_ang_vel_z_exp.weight`: 0.0 to 5.0
- `reward.feet_air_time.weight`: 0.0 to 2.0
- `reward.flat_orientation_l2.weight`: -10.0 to 0.0

Manual CLI overrides such as `--num-envs` and `--max-iterations` are still allowed.

## Trainer Override Scaffold

The repository now includes `trainer_override_mode` and `trainer_overrides` in `configs/base.yaml` and in the runtime `TrainingConfig` model.

Current modes:

- `none`: record requested overrides only; do not inject them
- `hydra`: translate supported keys into Hydra agent overrides for the Isaac Lab RSL-RL script

Current supported trainer override fields:

- `learning_rate`: `1e-6` to `1e-2`
- `entropy_coef`: `0.0` to `0.1`
- `clip_param`: `0.05` to `0.4`
- `num_mini_batches`: `1` to `64`

Today this is still a controlled scaffold:

- it is recorded into `effective_input.yaml`
- a separate `trainer_override_resolution.yaml` records whether the chosen mode produced effective CLI overrides
- a separate `trainer_override_verification.yaml` checks whether the dumped Isaac Lab `agent.yaml` actually contains the requested override values
- it is not planner-editable

Based on the checked Isaac Lab training entrypoint, Hydra-based agent overrides are feasible because the script preserves unknown CLI args for Hydra and applies them to `agent.*` config keys before training.

At the repository wrapper layer, `scripts/train_go2.sh` now preserves unknown extra arguments and forwards them to the Isaac Lab training script, which enables Hydra override passthrough without modifying Isaac Lab source code.

When Isaac Lab writes `params/agent.yaml` for a run, this repository now verifies the requested trainer overrides against that dumped agent config and records the result as `trainer_override_verification.yaml`.

This keeps future trainer override work explicit without pretending unsupported values already affect training.

The first trainer-parameter field family is now also reopened to the planner at the `training.*` layer, because these fields have a closed loop across planner validation, patch validation, Hydra mapping, passthrough, and `agent.yaml` verification.

The first command and reward field families are now also reopened to the planner because they now have the same closed loop across planner validation, patch validation, Hydra env overrides, passthrough, and final `env.yaml` verification.

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
  - extend onboarding from summary-only checks to capability wiring checks
  - inspect the target Isaac Lab task once at startup
  - discover log locations, checkpoint locations, important config entry points, and field wiring status
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
