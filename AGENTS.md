# AGENTS

This repository uses a minimal LLM interface for experiment planning.

## Purpose

The LLM is only used to propose the next round of configuration changes for Go2 training.

It is not allowed to:

- modify Python code
- modify shell scripts
- modify Isaac Lab source code
- change robot, task, observation, or physics definitions

## Current Interface

Primary code paths:

- `src/llm.py`
- `src/planner.py`
- `prompts/planner.txt`

Target/project attachment paths:

- `configs/base.yaml` under `target.project_root`
- `configs/base.yaml` under `target.artifact_root`
- `target_info.yaml` inside the external artifact root

Interface contract:

- input: prompt text
- output: JSON text

Expected planner JSON shape:

```json
{
  "hypothesis": "string",
  "changes": [
    {
      "field": "training.num_envs",
      "new_value": 512,
      "reason": "string"
    }
  ],
  "expected_effect": "string"
}
```

## Backends

Supported backends today:

- DeepSeek via `DEEPSEEK_API_KEY`
- mock response via `GO2_AUTO_TRAINER_LLM_RESPONSE`

Selection rule:

- if `DEEPSEEK_API_KEY` exists, use DeepSeek
- otherwise use the mock response environment variable

## Safety Model

The LLM does not directly execute changes.

Its output is constrained in multiple layers:

1. prompt-level whitelist and bounds
2. planner JSON validation
3. patcher field whitelist
4. patcher numeric bounds
5. retry on invalid patch proposal

If the LLM keeps producing invalid values, the run stops safely and writes a patch rejection artifact.

## Initialization Model

The trainer is attached to an external project through init metadata.

The init command records:

- target id
- external project root
- external artifact root
- minimal discovered path checks for launcher and training script

## Interrupt Analysis

On user interrupt, the system also uses the LLM to produce a small interruption analysis and saves it in the active run directory.

## Future Direction

Possible future work for the LLM interface:

- checkpoint-aware planning
- compressed multi-round context summaries
- failure-aware replanning
- run history indexing
