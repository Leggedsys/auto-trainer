#!/usr/bin/env bash

set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
LAUNCHER="${LAUNCHER:-/root/IsaacLab/isaaclab.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py}"
TASK="${TASK:-Isaac-Velocity-Flat-Unitree-Go2-v0}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
HEADLESS="${HEADLESS:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --isaaclab-root)
      ISAACLAB_ROOT="$2"
      shift 2
      ;;
    --launcher)
      LAUNCHER="$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$LAUNCHER" ]]; then
  printf 'Isaac Lab launcher not executable: %s\n' "$LAUNCHER" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  printf 'Training script not found: %s\n' "$TRAIN_SCRIPT" >&2
  exit 1
fi

CMD=(
  "$LAUNCHER"
  -p
  "$TRAIN_SCRIPT"
  --task
  "$TASK"
  --num_envs
  "$NUM_ENVS"
  --max_iterations
  "$MAX_ITERATIONS"
)

if [[ "$HEADLESS" == "1" ]]; then
  CMD+=(--headless)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"
