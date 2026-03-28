# 记忆技能草案

## 目的

本文件定义一个适合 OpenCode 使用方式的“项目开发记忆技能”草案。

目标不是实现程序级强制注入，而是把当前仓库中的开发记忆协议抽象成一个可重复执行的工作流，方便未来：

- 作为人工使用规范
- 作为可移植的 skill 提示模板
- 作为未来自定义 code agent 框架的前置规格

## 技能名称建议

- `project-memory-manager`

可选别名：

- `planning-memory`
- `dev-memory-manager`

## 技能作用范围

该技能只服务于 `auto-trainer` 项目本身的开发规划与实现。

它不负责：

- RL 训练运行时记忆
- 未来产品内部的训练 agent 状态管理
- 训练任务本身的经验回放或实验数据库

## 技能输入

技能启动时需要先判断当前工作属于哪一类：

### 1. project_view

适用于：

- 项目路线讨论
- 架构调整
- 权限边界变化
- 里程碑顺序变化
- 长期规划压缩

应先读取：

- `plans/PROJECT_MEMORY.md`
- `plans/NEXT_ACTIONS.md`
- `plans/memory_policy.yaml`

默认不读取：

- `plans/worklog/inbox.md`

### 2. task_view

适用于：

- 具体开发任务
- 某个 bug
- 某项验证失败
- 某个局部实现收敛

应先读取：

- `plans/NEXT_ACTIONS.md`
- `plans/memory_policy.yaml`

按需读取：

- `plans/worklog/inbox.md`
- `plans/PROJECT_MEMORY.md`

原则：

- 只取完成当前任务所需的最小上下文
- 不默认加载全部记忆文件

## 技能执行流程

### 步骤 1：选择视角

先判断当前问题是：

- 项目整体视角
- 具体任务收敛视角

### 步骤 2：按视角读取最小记忆

- `project_view` 读取稳定记忆
- `task_view` 读取当前行动和必要的局部原始记录

### 步骤 3：开始工作

执行规划、分析、编码或验证。

### 步骤 4：记录原始过程

若产生以下内容，先写入 `plans/worklog/inbox.md`：

- 临时判断
- 候选决策
- 未定型风险
- 过程性发现
- 尚不确定是否值得长期保留的信息

### 步骤 5：收尾时压缩

工作结束后按规则分流：

- 改变长期决策的内容 -> `plans/PROJECT_MEMORY.md`
- 改变当前优先级和下一步的内容 -> `plans/NEXT_ACTIONS.md`
- 纯过程性噪音 -> 从 `plans/worklog/inbox.md` 删除

## 技能内部决策规则

写入长期记忆前，至少检查三个问题：

1. 这条信息之后是否难以从代码和现状重新推出？
2. 它是否会改变未来决策？
3. 它是否已经是该想法最稳定、最高层的表达？

至少满足两个“是”，才考虑进入长期记忆。

## 技能防膨胀规则

- 不新增日期型规划文件
- 不为同一事实保留多个展开视图
- 不把 worklog 当长期记忆
- 不保留过程性叙述作为历史
- 优先覆盖更新而不是追加叙事

## 与当前仓库文件的映射

- 长期开发记忆：`plans/PROJECT_MEMORY.md`
- 当前开发行动：`plans/NEXT_ACTIONS.md`
- 原始过程缓冲区：`plans/worklog/inbox.md`
- 记忆协议：`plans/memory_policy.yaml`

## 未来演进方向

如果未来要把这个 skill 进一步落实，可以按三个阶段演进：

1. 作为人工遵循的提示模板
2. 作为 OpenCode 可调用的 skill 文案
3. 作为自定义 code agent 框架中的上下文选择与压缩模块

## 当前结论

在现阶段，最合适的做法不是追求程序级强制，而是把“如何先读记忆、如何选视角、如何写 worklog、如何压缩”固化成一套 skill 草案。

这能在不依赖底层框架定制的前提下，把项目开发记忆管理先做对。
