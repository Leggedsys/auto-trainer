# templete 字段识别实验输入

## 目标样本

- project_root: `/root/legged/templete`
- task_id: `Template-Autotrain-v0`
- task_type: manager-based RL env

## 建议输入文件

### 1. task 注册入口

- `/root/legged/templete/source/autotrain/autotrain/tasks/manager_based/autotrain/__init__.py`

作用：

- 识别 task id
- 识别 env cfg entry point
- 识别 agent cfg entry point

### 2. env cfg

- `/root/legged/templete/source/autotrain/autotrain/tasks/manager_based/autotrain/autotrain_env_cfg.py`

作用：

- 识别 reward 候选字段
- 识别 observation/action/event/termination 结构
- 判断 command 字段族是否存在

### 3. agent cfg

- `/root/legged/templete/source/autotrain/autotrain/tasks/manager_based/autotrain/agents/rsl_rl_ppo_cfg.py`

作用：

- 识别 trainer 候选字段
- 识别默认 trainer 超参数

### 4. 历史 env 产物

- `/root/legged/templete/logs/rsl_rl/cartpole_direct/2026-03-27_19-43-56/params/env.yaml`

作用：

- 补充最终配置视角
- 帮助确认 env 层字段在最终配置中的实际路径

### 5. 历史 agent 产物

- `/root/legged/templete/logs/rsl_rl/cartpole_direct/2026-03-27_19-43-56/params/agent.yaml`

作用：

- 补充最终配置视角
- 帮助确认 trainer 字段在最终配置中的实际路径

## 当前预期识别结果

### 预期存在的 trainer 候选

- `training.max_iterations`
- `training.learning_rate`
- `training.entropy_coef`
- `training.clip_param`
- `training.num_mini_batches`

### 预期存在的 reward 候选

- `reward.alive.weight`
- `reward.terminating.weight`
- `reward.pole_pos.weight`
- `reward.cart_vel.weight`
- `reward.pole_vel.weight`

### 预期不存在的字段族

- `command.*`

这里“不存在”应被识别为 `not_applicable`，而不是“接入失败”。

## 字段识别 prompt 草案

```text
You are analyzing an Isaac Lab task project for auto-trainer onboarding.

Your job is to identify candidate planner-editable fields from the provided source files and config dumps.

Return JSON only.

Rules:
- Identify only fields that are explicitly defined in the provided files.
- Prefer trainer hyperparameters, reward weights, and command ranges.
- If a field family is absent, do not invent fields from that family.
- Do not infer hidden fields that are not visible in the provided code or dumped configs.
- Use the exact schema below.

Return this exact JSON shape:
{
  "project": {
    "task_id": "string",
    "task_type": "string"
  },
  "candidate_fields": [
    {
      "field": "string",
      "layer": "training|reward|command|other",
      "kind": "scalar|range|boolean|enum|unknown",
      "source_file": "string",
      "source_symbol": "string",
      "default_value": "any JSON value",
      "hydra_path": "string or null",
      "candidate_for_planner": true,
      "reason": "string"
    }
  ],
  "absent_field_families": ["string"]
}
```

## 实验目标

- 验证 LLM 是否能从关键源码面中识别出 trainer / reward 候选字段
- 验证它是否能正确判断 command 字段族在该项目中不存在
- 验证输出是否足够转成第一版 field registry
