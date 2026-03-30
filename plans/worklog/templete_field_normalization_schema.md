# templete 字段归一化 schema

## 目标

将 raw discovery 结果转成 auto-trainer 平台可消费的 normalized candidate list。

## 固定输出 schema

```json
{
  "project": {
    "task_id": "string",
    "task_type": "string"
  },
  "normalized_fields": [
    {
      "source_field": "string",
      "normalized_field": "string",
      "normalized_layer": "training|reward|command|other",
      "kind": "scalar|range|boolean|enum|unknown",
      "source_file": "string",
      "hydra_path": "string or null",
      "default_value": "any JSON value",
      "validation_source": "agent.yaml|env.yaml|effective_input.yaml|null",
      "candidate_for_planner": true,
      "normalization_reason": "string"
    }
  ],
  "not_applicable_families": ["string"],
  "unmapped_source_fields": [
    {
      "source_field": "string",
      "reason": "string"
    }
  ]
}
```

## 说明

- `source_field` 表示原始识别结果中的字段名
- `normalized_field` 表示 auto-trainer 平台字段名
- `not_applicable_families` 用于表达项目本身不存在的字段族
- `unmapped_source_fields` 用于表达识别到了但当前不应稳定映射的字段

## 约束

- 不要发明源码中不存在的字段
- 不要强行把每个 source field 都映射成 planner 字段
- 如果一个字段族不存在，应放入 `not_applicable_families`
- 如果一个 source field 过于底层、风险高或平台尚未定义映射，应放入 `unmapped_source_fields`

## templete 样本的预期结果

### 应归一化的 trainer 字段

- `max_iterations` -> `training.max_iterations`
- `algorithm.learning_rate` -> `training.learning_rate`
- `algorithm.entropy_coef` -> `training.entropy_coef`
- `algorithm.clip_param` -> `training.clip_param`
- `algorithm.num_mini_batches` -> `training.num_mini_batches`

### 应归一化的 reward 字段

- `rewards.alive.weight` -> `reward.alive.weight`
- `rewards.terminating.weight` -> `reward.terminating.weight`
- `rewards.pole_pos.weight` -> `reward.pole_pos.weight`
- `rewards.cart_vel.weight` -> `reward.cart_vel.weight`
- `rewards.pole_vel.weight` -> `reward.pole_vel.weight`

### 应标为 not_applicable 的字段族

- `command`

### 可能先不映射的平台外 trainer 字段

- `algorithm.schedule`
- `algorithm.gamma`
- `algorithm.lam`
- `algorithm.desired_kl`
- `algorithm.max_grad_norm`
- `algorithm.value_loss_coef`
- `algorithm.use_clipped_value_loss`
- `algorithm.num_learning_epochs`

这些字段可以保留在 `unmapped_source_fields`，等待后续 planner gating 或更高阶段决定是否纳入平台字段。
