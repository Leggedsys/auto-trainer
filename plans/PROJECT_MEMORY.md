# 项目开发记忆

说明：

- 本文件用于管理 `auto-trainer` 项目自身的开发规划记忆。
- 它服务于本项目开发过程，不是未来产品运行时的训练记忆层。
- 随着项目成熟，这套记忆可能迁移到仓库外部，因此内容应保持高压缩、低耦合。

## 1. 项目终局

本项目的终局不是安全调几个训练参数，而是把系统演进为一个面向 Isaac Lab 外部项目的自治强化学习员工。

终局能力定义：

- 输入一段简短 prompt
- 自动生成或修改 Isaac Lab 外部训练项目
- 自动验证、启动训练、分析结果并继续迭代
- 在达到目标、预算耗尽或人工中止时安全停机
- 全过程可观察、可暂停、可回滚、可审计

典型工作场景：

```text
用户：帮我使用这个机器狗模型训练一个后空翻动作
auto-trainer：
    生成项目中......
    项目预计耗时20小时，开始训练......（输入P暂停，输入D开启GUI观察当前结果）
```

## 2. 不可违背的方向约束

- 优先生成和控制 Isaac Lab 外部项目，不直接改 Isaac Lab 源码。
- 权限必须分层开放，不能一步放开高风险代码生成。
- 没有真实注入路径和验证链路的能力，不允许进入可编辑面。
- 任何自动改动都必须有 artifact、回滚点和生效证明。
- 记忆系统本身必须压缩，不能退化成日志堆积。

## 3. 当前总体路线

路线压缩为三阶段：

1. 可信自动调参器
2. 外部项目生成器
3. 自治实验代理

更细的能力释放顺序为：

1. 可信观测
2. 可信训练编排
3. 可信训练器超参控制
4. 可信环境参数覆盖
5. 可信外部项目模板生成
6. 可信受限代码生成
7. 可信结构变体生成
8. 高自治实验研究

## 4. 当前阶段判断

当前阶段应拆成两个子阶段理解：

- **阶段 1A：可信自动调参器内核**
  - 面向当前已接入项目
  - 重点是“允许修改 = 实际生效 = 可验证”
- **阶段 1B：可信项目 onboarding**
  - 面向新项目
  - 重点是把字段发现、字段接入和字段知识建模自动化进 `--init`

当前判断：

- 阶段 1A 已非常接近闭合
- 阶段 1B 仍未闭合，是下一段核心工作

## 5. 当前最高优先级

当前最高优先级已经从“补 trainer / env 闭环”切换为：

- 保持 1A 已闭合能力稳定
- 开始推进 1B，也就是新项目自动接入 init

阶段 1A 现在的闭合定义：

- 对当前已接入项目，开放字段具备 planner、patcher、resolver、passthrough、最终配置核验和 artifact 闭环
- planner 具备字段语义理解，而不是只靠字段命名猜测

阶段 1B 现在的目标：

- 让 `--init` 不只是摘要当前接入状态
- 而是真正开始承担字段发现、字段接入、字段知识建模和缺口报告

## 6. 关键决策记忆

### 决策 A：终局前推

项目目标已经明确前推到“简短 prompt -> 生成 Isaac Lab 外部训练项目 -> 自动训练迭代 -> 达标后停机”，不再把自己只定义为受限调参器。

### 决策 B：分层开放优先于一次性放权

即使终局包含自动代码生成和自治训练，也必须按验证能力逐层开放权限。

### 决策 C：记忆不是日志

规划与记忆系统不得持续堆叠日期文件、矩阵文件、里程碑文件和重复总结。长期记忆只保留高价值、低可再推导的信息。

### 决策 D：当前先修真实生效链路

近期不追求增加更多可改字段，先修正“允许修改”和“真实生效”不一致的问题。

### 决策 E：M1 期间只保留真实生效最小字段集

在 trainer override 和 env override 尚未打通前，planner、prompt 和 patcher 只保留 `training.num_envs` 与 `training.max_iterations` 作为自动可改字段；实际启动输入单独落盘为 artifact，避免“看起来改了但训练没吃到”的假闭环。

### 决策 F：trainer override 先建骨架再开放权限

`trainer_overrides` 可以先进入配置模型和 artifact，可用于表达未来 trainer 参数注入点；但在真实命令注入和验证路径完成前，它既不进入 planner 白名单，也不伪装成已经生效的训练能力。

### 决策 G：trainer override 优先走 Hydra 注入路径

经检查 Isaac Lab 当前的 RSL-RL 训练入口，未知 CLI 参数会保留给 Hydra，且 `agent.*` 覆盖可以进入 agent 配置。因此 trainer override 的第一优先真实注入路径应是 Hydra agent override，而不是修改 Isaac Lab 主仓库源码。

### 决策 H：本仓库 wrapper 已具备 Hydra passthrough 能力

`scripts/train_go2.sh` 现在会保留未知附加参数并原样透传给 Isaac Lab 训练脚本，因此在 `trainer_override_mode=hydra` 下，受支持 trainer 参数已经具备从本仓库进入 Isaac Lab Hydra 配置层的真实命令注入路径。

### 决策 I：首批 trainer override 字段先受控生效，不立即交给 planner

`learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches` 已作为首批支持字段进入 override 校验与 artifact 生效标记，但当前阶段仍保持人工或配置驱动，不直接重新开放给 planner。

### 决策 J：真实生效以 agent.yaml 核验为准

对于 Hydra 模式下的 trainer override，不能只以命令参数已透传作为“已生效”依据；应以 Isaac Lab 训练过程中落盘的 `params/agent.yaml` 为核验源，并把比对结果写成独立 artifact。

### 决策 K：trainer 字段族按闭环整体开放

只要一个字段族已经具备统一 schema、统一注入路径、统一最终验证、统一 artifact 和回滚基础，就按字段族整体开放，而不按字段数量做细碎门控。当前 `learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches` 已满足这一条件，因此已整体重新开放给 planner。

### 决策 L：command 字段族优先复用 env Hydra 覆盖

Go2 velocity task 的 command 入口位于 `env.commands.base_velocity.ranges.*`，且 Isaac Lab 已支持 `env.*` Hydra 覆盖。因此 `command.*` 字段族的首选真实注入路径应是 env Hydra override，而不是修改 Isaac Lab 主仓库源码。

### 决策 M：command 字段族已完成真实闭环验证

`lin_vel_x`、`lin_vel_y`、`ang_vel_z`、`heading` 已具备 schema、Hydra 注入、wrapper passthrough、`env.yaml` 最终配置核验与 artifact 闭环，因此 command 字段族已达到和 trainer 字段族相同级别的真实可验证状态。

### 决策 N：reward 第一批只开放 weight 标量

reward 第一批只开放 `track_lin_vel_xy_exp.weight`、`track_ang_vel_z_exp.weight`、`feet_air_time.weight`、`flat_orientation_l2.weight` 这类 weight 标量，不先开放 params、reward 项启停或 reward 结构变化。这样可以保持注入与验证简单、风险边界清晰。

### 决策 O：reward 第一批字段已完成真实闭环验证

首批 reward weight 字段已经具备 schema、Hydra 注入、wrapper passthrough、`env.yaml` 最终配置核验与 artifact 闭环，因此 reward 第一批字段已达到真实可验证状态。

### 决策 P：command 与 reward 第一批字段已重新开放给 planner

既然 command 字段族和 reward 第一批字段都已经完成真实闭环验证，就不再人为保留在“仅人工配置”阶段，而是和 trainer 字段族一样重新开放给 planner。

### 决策 Q：项目初始化应生成字段语义注册表

随着开放字段增多，planner 不能再只依赖字段名猜语义。项目初始化阶段应生成项目专属 `field_registry.yaml`，其中包含字段角色、作用方向、耦合关系、边界和验证来源。planner prompt 应优先消费这份结构化字段语义，而不是只看原始配置文本。

### 决策 R：初始化命令升级为项目 onboarding

`--init` 不再只是路径初始化。它还应校验 field registry，并把当前接入字段层、planner 可用字段数、验证来源等能力摘要写入 `target_info.yaml`，使初始化结果同时具备运行入口与能力入口信息。

### 决策 S：第一阶段拆分为 1A / 1B

第一阶段不再作为单块目标理解：

- 1A 是“可信自动调参器内核”，面向已接入项目，当前已基本闭合
- 1B 是“可信项目 onboarding”，面向新项目自动接入，当前尚未闭合

后续开发与评估应明确区分这两部分，避免把已经闭合的内核能力和仍未自动化的接入能力混为一谈。

### 决策 T：1B 的目标是可信 onboarding，不是一次性全自动接线

1B 的目标不是在第一步就实现“拿任意新项目后全自动完成所有字段接线”，而是建立可信 onboarding pipeline：

- 发现项目入口
- 识别候选字段
- 生成或更新字段语义注册表
- 判断字段接入状态
- 报告能力缺口
- 在必要时进入半自动接入流程

也就是说，1B 首先要做的是“可信发现与可信诊断”，然后才逐步推进到更高自动化。

进一步约束：

- 1B 当前先面向单 task onboarding，不先追求多 task 联合项目建模。
- 1B 的目标是把“给定 task 推到极限”的接入流程自动化，而不是一开始就支持开放式研究代理。
- 字段注册之外，还需要注册 task objective policy，用于定义主指标、次指标、plateau 判据、停止条件和不可接受行为。
- 字段是否算“已接入”必须有状态判定，至少区分：known、mapped、resolvable、verifiable、planner_enabled。
- onboarding 的最小验收应包含真实两轮实验：第一轮跑通，第二轮至少一项注册字段被 LLM 修改并通过最终配置核验。
- 1B 在第一步不要求自动生成所有 resolver 代码，但必须能可靠指出缺口并给出清晰的接入状态。

### 决策 U：capability metadata 应逐步从 controller 配置中解耦

随着多项目接入，`field_registry.yaml` 这类能力元数据不应长期绑定在 controller repo 的静态 `configs/` 下。更合理的长期方向是将其视为 target-specific capability metadata，并逐步与 controller 通用逻辑解耦。

### 决策 V：1B 必须以新项目验收为准

1B 的完成标准不能只在当前 Go2 项目上自洽，而必须拿一个新的项目进行 onboarding 验收。是否闭合，应以“新项目接入是否能得到可信能力摘要与清晰缺口报告”来判断。

### 决策 AK：1B 当前范围先收缩到腿足 locomotion task

为降低复杂度并提高规则稳定性，1B 当前不面向任意 Isaac Lab RL 项目，而先明确限定在腿足 locomotion task，尤其是 manager-based locomotion velocity 一类任务。也就是说，1B 的第一版目标不是“通用 RL onboarding”，而是“legged locomotion onboarding”。

### 决策 AL：legged locomotion 的字段族具有稳定共性

在 Isaac Lab 的 manager-based locomotion velocity 样本中，当前已确认以下共性：

- trainer 字段族高度稳定，核心通常包括 `max_iterations`、`learning_rate`、`entropy_coef`、`clip_param`、`num_mini_batches`
- command 字段族高度稳定，核心通常围绕 `base_velocity` 的 `lin_vel_x`、`lin_vel_y`、`ang_vel_z`、`heading`
- reward 字段族存在较稳定的共性骨架，如 `track_lin_vel_xy_exp`、`track_ang_vel_z_exp`、`feet_air_time`、`flat_orientation_l2`、`dof_acc_l2`、`action_rate_l2`，但也允许 task-specific 特例层存在

因此 1B 应采用：

- trainer：规则优先
- command：规则优先
- reward：共性规则 + LLM 识别特例

### 决策 AM：当前不扩到 direct RL 与多智能体结构

由于 DirectRLEnv、DirectMARLEnv 与 manager-based locomotion 在字段定义位置、reward 表达方式和接入策略上差异较大，1B 当前不扩到这些结构。等 legged locomotion onboarding 稳定后，再扩展到其他 RL/task 结构。

### 决策 AN：legged locomotion normalization 采用规则优先

在当前收缩后的 1B 范围内，字段归一化不再继续完全交给 LLM。raw discovery 由 LLM 识别 source-level candidate fields，而 normalization 由 legged locomotion 专用规则层负责，把 source field 稳定映射到 `training.*`、`command.*`、`reward.*` 平台字段，并把暂不纳入的平台外字段留在 unmapped 集合中。

### 决策 AO：onboarding 流程先显式拆出 raw discovery 与 normalization 接口

在 1B 早期，`--init` 不应直接内嵌全部发现与归一化逻辑，而应先通过独立 onboarding 接口明确分层：

- raw discovery 输入
- normalized candidate list 输出

这样后续接 LLM 字段识别时，边界更稳定，也更容易替换 discovery 实现。

### 决策 AP：1B 已具备最小可运行的 discovery -> normalization 原型

1B 现在应视为已经具备最小可运行原型：

- field discovery prompt 构造
- LLM raw discovery 输出解析
- normalization 规则层归一化
- normalized candidate list 保存接口

这还不是完整 onboarding，但已经不再停留在手工实验阶段。

### 决策 AQ：discovery 的字段族类别必须受限

在 1B 当前阶段，`absent_field_families` 不能完全自由生成，而应限制在平台当前关心的少数字段族集合中。否则 raw discovery 会引入过多无关类别，影响 normalization 和 capability package 的稳定性。

### 决策 AR：normalized candidate list 应成为 capability package 的正式中间产物

随着 discovery -> normalization 原型打通，`normalized_candidate_list.yaml` 应视为 capability package 的正式中间产物。它位于 raw discovery 与 capability status 之间，承接字段归一化后的稳定结果，为后续状态判断与 onboarding report 更新提供输入。

### 决策 AS：capability_status 应优先基于 normalized candidate list 计算

随着 normalized candidate list 落入 capability package，`capability_status.yaml` 不应继续只依赖静态 field registry 汇总，而应优先基于 normalized candidate list 判断哪些字段当前真的已进入 onboarding 识别链路、哪些字段族处于 `not_applicable` 状态。

### 决策 AT：--init 已开始具备受限版自动 discovery 能力

在当前 1B 范围内，`--init` 可以在受限条件下自动触发 discovery -> normalization：当 task 被判定为 legged locomotion，且 target 中显式提供 `env_cfg_path`、`agent_cfg_path`、`task_registry_path` 等关键入口时，init 会生成 `raw_discovery.yaml` 与 `normalized_candidate_list.yaml`，并用其更新 capability package 状态。这仍是受限版自动接入，不是通用任意项目接入。

### 决策 W：1B 需要新的项目结构分离约束

为支持可信 onboarding，项目结构应逐步满足以下约束：

- tool core 与具体 target project 彻底分离，保持 controller 洁净。
- workspace path 必须显式锁定，并作为 capability package、artifact root、onboarding 输出的统一容器。
- Isaac Lab 路径不应长期散落在 task 配置中，后续应逐步收敛到更稳定的工具安装或环境层配置。
- capability package 应视为 target-specific 资产，而不是 controller repo 的普通静态配置文件。
- 多项目能力当前先按 task 划分项目单元，而不是先做更复杂的跨 task 组合抽象。

### 决策 X：版本控制应面向 capability package，而不只是代码 diff

在 1B 阶段，多项目支持的关键资产不只是源码，而是 capability package：字段注册、task objective policy、接入状态、验证结果和 onboarding 输出。后续项目版本控制应优先支持这些语义资产，而不是只依赖源码级 diff。

### 决策 Y：1B 第一版 capability package 采用显式文件模型

1B 的第一版不应把所有 onboarding 信息塞进一个大文件，而应采用清晰的 capability package 结构。建议最小文件模型如下：

- `project_identity.yaml`
  - target_id
  - project_root
  - task_id
  - workspace_root
  - artifact_root
  - isaaclab_root
- `field_registry.yaml`
  - 字段语义注册表
- `task_policy.yaml`
  - 主指标、次指标、plateau 判据、停止条件、不可接受行为
- `capability_status.yaml`
  - 每个字段和字段族的接入状态
- `onboarding_report.yaml`
  - 一次 init/onboarding 的摘要结果与缺口报告

这几个文件共同构成一个 task-specific capability package。

### 决策 Z：字段接入状态需要结构化状态模型

`capability_status.yaml` 不应只写“已接入/未接入”，而应至少支持以下状态维度：

- `known`: 已被识别并进入 registry
- `mapped`: 已有明确 config/hydra 路径
- `resolvable`: 已有 resolver 逻辑
- `verifiable`: 已有最终配置核验路径
- `planner_enabled`: 已开放给 planner
- `evidence`: 当前状态的证据来源，如 `agent.yaml`、`env.yaml`、artifact 名称、测试名

这样 onboarding 才能输出真正可操作的缺口，而不是模糊描述。

### 决策 AA：onboarding report 应面向缺口与行动，而不是面向描述

`onboarding_report.yaml` 的重点不应是长篇说明，而应是：

- 当前 task 识别结果
- capability package 位置
- 已闭环字段族
- 未闭环字段族
- 缺失的 resolver / verification / planner wiring
- 是否达到“两轮验收”的最小条件
- 下一步建议动作

它应当是一个面向工程行动的结构化报告，而不是 narrative 文档。

### 决策 AB：1B 第一版先落 capability package 最小文件

1B 的第一版实现不必等待全部自动接入能力完成。当前应先让 `--init` 真实产出 capability package 的最小文件：`project_identity.yaml` 与 `capability_status.yaml`，把“当前已接入什么、接到哪一步”先结构化落地。

### 决策 AC：1B 第一版 capability package 继续补 task_policy 与 onboarding_report

在最小文件落地后，`--init` 还应继续产出第一版 `task_policy.yaml` 与 `onboarding_report.yaml`。即使当前仍是半静态结构，也要先把 task 目标约束和 onboarding 缺口报告显式化，避免 capability package 只剩字段列表与状态计数。

### 决策 AD：1B 的 field registry 应来自字段识别，而不是预设字段模板

1B 中的 `field_registry.yaml` 不应长期通过手工预设字段名生成。更合理的路径是：先定位“包含字段定义的关键源码面”，再把这些源码交给 LLM 按固定 schema 输出候选字段结构，最后再生成 registry。也就是说，1B 的核心是 discover fields，而不是 preset fields。

### 决策 AE：字段识别优先使用 LLM + 固定输出 schema

字段识别阶段当前优先采用：

- 精选输入源码面
- 固定 prompt
- 固定 JSON/YAML 输出 schema

先让 LLM 识别 trainer、reward、command 等候选字段，再进入后续 registry 生成与接入状态判断。这样更贴近新项目 onboarding 的真实需求，也避免过早把当前 Go2 字段模板硬编码成通用方案。

### 决策 AF：第一版字段识别 schema 应尽量小而硬

第一版字段识别不应追求一次输出过多解释信息，而应优先采用固定、最小且可程序化的 schema。每个候选字段至少输出：

- `field`
- `layer`
- `kind`
- `source_file`
- `source_symbol`
- `default_value`
- `hydra_path`
- `candidate_for_planner`
- `reason`

目标是先保证可解析、可转 registry、可人工复核，而不是一次生成完整 capability package。

### 决策 AG：第一版字段识别输入只喂关键源码面

字段识别时不应把整个项目代码库塞给 LLM，而应只提供“包含字段定义的关键源码面”。当前优先输入面包括：

- task 注册入口
- env cfg 文件
- agent cfg 文件
- 若已有历史运行，则补 `env.yaml` 与 `agent.yaml`

这样既能保持上下文可控，也足以让 LLM识别大部分 trainer/reward/command 候选字段。

### 决策 AH：字段识别 pipeline 拆成三段

1B 的字段识别不应视为一步完成，而应拆成三个阶段：

- `raw discovery`
  - 从关键源码面中识别 source-level candidate fields
- `normalization`
  - 把 source field 映射成 auto-trainer 平台字段、标准 layer 和完整 hydra path
- `planner gating`
  - 决定哪些 normalized fields 真正进入 planner 开放面

这样可以把“识别字段”和“开放字段”分离，避免第一次识别就被迫承担过多策略判断。

### 决策 AI：normalization 需要单独的最小 schema

raw discovery 输出后，normalization 阶段至少要补齐以下字段：

- `source_field`
- `normalized_field`
- `normalized_layer`
- `hydra_path`
- `kind`
- `default_value`
- `validation_source`
- `candidate_for_planner`
- `normalization_reason`

其中：

- `source_field` 反映项目源码中的原始字段命名
- `normalized_field` 反映 auto-trainer 平台视角下的标准命名

这样后续 registry 与 capability status 才能基于稳定字段名工作。

进一步约束：

- normalization 输出应继续保持固定 JSON/YAML 结构，不能回到自由文本总结。
- normalization 阶段允许丢弃无法稳定映射的平台字段，而不是强行归一化所有 source fields。
- normalization 阶段必须支持 `not_applicable` 字段族输出，以表达某些项目天生不具备该字段面。
- planner gating 只能消费 normalized fields，不能直接消费 raw discovery 结果。

### 决策 AJ：templete 样本首先验证 normalization，而不是直接开放

对 `/root/legged/templete` 这类外部样本，第一次实验的重点不应是立刻开放 planner 字段，而应先验证：

- LLM 能否识别 source fields
- source fields 能否被稳定归一化
- 不存在的字段族能否被标成 `not_applicable`

只有这一步稳定后，再进入 planner gating。

## 7. 记忆维护原则

- 只保留会改变未来决策的信息。
- 能从当前代码、主路线和约束中重新推出的信息，不进入长期记忆。
- 用覆盖式压缩替代追加式记录。
- 重要历史只保留少量“关键决策”，不保留过程性讨论痕迹。
- 如果一条信息不能显著提高未来决策质量，就应删除。

## 8. 与执行面的关系

本文件只保存稳定记忆，不保存细碎任务列表。

本文件也不保存原始工作过程。原始过程记录应先进入 `plans/worklog/inbox.md`，经压缩后再进入本文件。

具体近期动作放在：

- `plans/NEXT_ACTIONS.md`
- `plans/memory_policy.yaml`

原始记录缓冲区在：

- `plans/worklog/inbox.md`

其中：

- `PROJECT_MEMORY.md` 负责长期稳定记忆
- `NEXT_ACTIONS.md` 负责当前执行状态
- `memory_policy.yaml` 负责约束记忆结构与更新协议
