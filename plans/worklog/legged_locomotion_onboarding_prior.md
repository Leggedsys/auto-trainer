# legged locomotion onboarding 先验

## 范围

当前 1B 只面向 Isaac Lab 中的腿足 locomotion task，尤其是：

- manager-based locomotion velocity tasks
- 四足与腿足机器人

暂不覆盖：

- direct RL
- multi-agent RL
- cartpole / manipulation / factory 等非腿足任务

## 稳定字段族先验

### trainer 核心字段

优先假设存在：

- `training.max_iterations`
- `training.learning_rate`
- `training.entropy_coef`
- `training.clip_param`
- `training.num_mini_batches`

这些通常分别映射到：

- `agent.max_iterations`
- `agent.algorithm.learning_rate`
- `agent.algorithm.entropy_coef`
- `agent.algorithm.clip_param`
- `agent.algorithm.num_mini_batches`

### command 核心字段

优先假设存在：

- `command.lin_vel_x`
- `command.lin_vel_y`
- `command.ang_vel_z`
- `command.heading`

这些通常映射到：

- `env.commands.base_velocity.ranges.lin_vel_x`
- `env.commands.base_velocity.ranges.lin_vel_y`
- `env.commands.base_velocity.ranges.ang_vel_z`
- `env.commands.base_velocity.ranges.heading`

### reward 共性骨架

优先关注的共性 reward 字段：

- `reward.track_lin_vel_xy_exp.weight`
- `reward.track_ang_vel_z_exp.weight`
- `reward.feet_air_time.weight`
- `reward.flat_orientation_l2.weight`
- `reward.dof_acc_l2.weight`
- `reward.action_rate_l2.weight`
- `reward.dof_torques_l2.weight`

这些字段在不同腿足任务中常见，但不要求全部存在。

## normalization 原则

### trainer

- 规则优先
- 如果原始字段是 `algorithm.learning_rate` 这类形式，直接归一化到 `training.*`

### command

- 规则优先
- 如果源码中存在 `base_velocity` command range，直接归一化到 `command.*`

### reward

- 共性 reward 规则优先
- 若发现 task-specific reward（例如特定机器人专用项），保留在特例集合中，等待后续 registry 与 planner gating 判断

## not applicable 规则

在腿足 locomotion onboarding 中：

- 若没识别到稳定 command 结构，可标记 `command` 为 `missing`，而不是直接推定 `not_applicable`
- 对于 reward 共性骨架中不存在的单个字段，应视为“该字段不存在”，不代表 reward 字段族整体不存在

## 当前用途

本先验用于：

- 约束 1B 当前范围
- 指导 normalization 规则设计
- 避免把 Go2 特例误写成通用腿足任务结构
