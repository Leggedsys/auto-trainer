# templete 字段归一化草案

## 目的

本文件用于把第一次字段识别实验得到的 source-level candidate fields 归一化为 auto-trainer 平台字段。

当前阶段先做 normalization，不直接做 planner 开放决策。

## raw discovery 中的主要字段

### trainer 类

- `max_iterations`
- `algorithm.learning_rate`
- `algorithm.entropy_coef`
- `algorithm.clip_param`
- `algorithm.num_mini_batches`
- 以及其他更宽的 PPO 候选项

### reward 类

- `rewards.alive.weight`
- `rewards.terminating.weight`
- `rewards.pole_pos.weight`
- `rewards.cart_vel.weight`
- `rewards.pole_vel.weight`

### absent family

- `command_ranges`

## 建议的 normalization 规则

### trainer

- `max_iterations` -> `training.max_iterations`
  - hydra: `agent.max_iterations`
- `algorithm.learning_rate` -> `training.learning_rate`
  - hydra: `agent.algorithm.learning_rate`
- `algorithm.entropy_coef` -> `training.entropy_coef`
  - hydra: `agent.algorithm.entropy_coef`
- `algorithm.clip_param` -> `training.clip_param`
  - hydra: `agent.algorithm.clip_param`
- `algorithm.num_mini_batches` -> `training.num_mini_batches`
  - hydra: `agent.algorithm.num_mini_batches`

### reward

- `rewards.alive.weight` -> `reward.alive.weight`
  - hydra: `env.rewards.alive.weight`
- `rewards.terminating.weight` -> `reward.terminating.weight`
  - hydra: `env.rewards.terminating.weight`
- `rewards.pole_pos.weight` -> `reward.pole_pos.weight`
  - hydra: `env.rewards.pole_pos.weight`
- `rewards.cart_vel.weight` -> `reward.cart_vel.weight`
  - hydra: `env.rewards.cart_vel.weight`
- `rewards.pole_vel.weight` -> `reward.pole_vel.weight`
  - hydra: `env.rewards.pole_vel.weight`

### not applicable

- `command_ranges` -> field family status = `not_applicable`

## 当前判断

这个样本说明：

- raw discovery 可以识别出足够多的 source fields
- normalization 可以把其中一部分稳定映射到现有平台字段模式
- planner gating 目前不应直接放开所有 trainer/reward 字段，而应在 normalization 后再单独判断

## 下一步

- 把这份 normalization 草案转成固定 schema
- 再让 LLM 或规则层输出正式 normalized candidate list
