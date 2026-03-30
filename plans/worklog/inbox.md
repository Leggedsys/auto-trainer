# 工作记录缓冲区

用途：

- 作为原始工作记录入口
- 允许记录过程性、探索性、未压缩的信息
- 作为压缩进 `plans/PROJECT_MEMORY.md` 和 `plans/NEXT_ACTIONS.md` 的输入缓冲层

使用规则：

- 先记录，再压缩
- 不把本文件当长期记忆
- 达到压缩阈值后，必须清理、分流、覆盖压缩
- 只保留少量尚未定型但仍有价值的内容

建议记录格式：

```text
## 条目
- 主题：
- 类型：idea / decision / risk / task / observation / discard
- 内容：
- 是否进入长期记忆：yes / no / maybe
- 是否进入当前行动：yes / no / maybe
```

当前状态：

- 缓冲区为空，可直接写入新的原始工作记录。

## 条目
- 主题：M1 最小闭环收紧
- 类型：decision
- 内容：下一步先统一 planner、patcher 和 prompt 的允许字段到当前真实生效最小集，只保留 `training.num_envs` 与 `training.max_iterations`，并补一个单独的 `effective_input.yaml` artifact 来记录实际启动输入。
- 是否进入长期记忆：maybe
- 是否进入当前行动：no

## 条目
- 主题：M1 最小测试补齐
- 类型：observation
- 内容：已增加最小测试覆盖两件关键事情：其一是 planner 与 patcher 的允许字段保持一致且仅保留 M1 最小字段集；其二是 runner 会落盘 `effective_input.yaml`，记录实际 command、cwd 和训练输入。当前 pytest 共 3 项，全部通过。
- 是否进入长期记忆：no
- 是否进入当前行动：maybe

## 条目
- 主题：trainer override 骨架
- 类型：decision
- 内容：先建立 `trainer_overrides` 数据结构和 artifact 可见性，但暂不把它接进训练命令，也不开放给 planner。这样可以先把未来注入点显式建出来，同时避免伪装成“已经生效”的能力。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：Isaac Lab trainer override 可行路径
- 类型：observation
- 内容：检查 `/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`、`cli_args.py` 和 Hydra 工具后，确认当前训练脚本会把未知 CLI 参数留给 Hydra，并允许用 `agent.*` 覆盖 agent 配置，因此 `learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches` 这类 trainer 参数存在无须修改 Isaac Lab 主仓库源码的 Hydra 注入路径。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：Hydra override 透传打通
- 类型：decision
- 内容：本仓库的 `scripts/train_go2.sh` 已改为保留未知附加参数并原样透传给 Isaac Lab `train.py`，因此在 `trainer_override_mode=hydra` 下，受支持的 trainer overrides 已具备真实命令注入路径，无需修改 Isaac Lab 主仓库源码。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：首批 trainer 字段受控生效
- 类型：decision
- 内容：先把 `learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches` 作为首批受控 trainer override 字段，加入边界校验和 artifact 生效标记，但仍暂不开放给 planner，先保持人工/配置驱动。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：trainer override 真实生效验证
- 类型：decision
- 内容：不只验证 Hydra 参数进了 command，还要在 Isaac Lab 生成的 `params/agent.yaml` 上核对请求值是否真正写入最终 agent 配置，并把结果落盘为 `trainer_override_verification.yaml`。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：trainer 字段族整体重新开放
- 类型：decision
- 内容：既然 `learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches` 这组 trainer 字段已经具备 planner 校验、patcher 校验、Hydra 注入、wrapper passthrough 与 `agent.yaml` 验证的闭环，就按字段族整体重新开放给 planner，而不再按单字段细碎门控。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：command 覆盖优先走 env Hydra 路径
- 类型：decision
- 内容：Go2 velocity task 的 command 配置位于 `env.commands.base_velocity.ranges.*`，而 Isaac Lab 已支持 `env.*` Hydra 覆盖，因此 command 字段族应优先复用 Hydra env override，而不是先改 Isaac Lab 源码。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：reward 第一批字段选择
- 类型：decision
- 内容：reward 第一批只开放 weight 标量，不开放 params 和结构项；优先字段为 `track_lin_vel_xy_exp.weight`、`track_ang_vel_z_exp.weight`、`feet_air_time.weight`、`flat_orientation_l2.weight`，因为它们配置位置明确、可用 Hydra 覆盖、可通过 `env.yaml` 最终核验。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：项目字段注册表
- 类型：decision
- 内容：项目初始化阶段应生成项目专属 `field_registry.yaml`，用结构化语义补足 planner 对字段的理解，避免仅靠命名猜字段作用。planner prompt 后续应读取这份 registry，而不是只看 config 和字段名。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：初始化升级为 onboarding
- 类型：decision
- 内容：`--init` 不应只初始化路径，还应校验 field registry，并把当前接入字段层、planner 可用字段数和验证来源摘要写入 `target_info.yaml`，使初始化结果同时具备“运行入口”和“能力入口”信息。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：1B onboarding 设计方向
- 类型：decision
- 内容：1B 不应一开始追求“任意新项目全自动接线”，而应先建立可信 onboarding pipeline：项目入口发现、候选字段识别、字段注册表生成、接入状态判断、能力缺口报告，并最终用一个新项目验收。与此同时，需要重新设计 controller、target project、capability metadata 与 artifact root 的分离关系，避免把当前 Go2 布局硬编码成通用结构。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：字段识别优先于预设模板
- 类型：decision
- 内容：1B 不应继续用手工预设字段名去生成 registry，而应先把包含字段定义的关键源码整体提供给 LLM，再用固定 schema 输出候选字段结构。`/root/legged/templete` 将作为第一个外部样本验证这条路径。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：legged locomotion normalization 规则层
- 类型：decision
- 内容：对于当前收缩后的 1B，normalization 不再继续完全交给 LLM，而改为规则优先。raw discovery 由 LLM 给出 source fields，随后由 legged locomotion 专用规则层将其归一化到 `training.*`、`command.*`、`reward.*` 平台字段。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：onboarding 内部 normalization 接口
- 类型：decision
- 内容：1B 下一步不直接把 normalization 规则埋进 init 主流程，而是先通过 `onboarding.py` 提供 raw discovery 读取、normalized candidate list 生成和保存接口，给 future field discovery 接入预留稳定边界。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes

## 条目
- 主题：discovery 字段族约束与 normalized candidate list
- 类型：decision
- 内容：raw discovery 中的 `absent_field_families` 不应自由发散，只允许当前平台关心的少数字段族；同时 normalized 结果应开始落入 capability package，形成 `normalized_candidate_list.yaml`，为后续 capability status 和 onboarding report 更新做准备。
- 是否进入长期记忆：yes
- 是否进入当前行动：yes
