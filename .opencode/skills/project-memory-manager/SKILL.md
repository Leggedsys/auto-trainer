---
name: project-memory-manager
description: 使用项目开发记忆协议先选择整体视角或任务视角，再按需读取记忆与压缩规则，避免上下文膨胀
compatibility: opencode
metadata:
  audience: agents
  scope: auto-trainer-development
  workflow: memory-protocol
---

## 用途

用于管理 `auto-trainer` 项目开发过程中的记忆读取、原始记录和压缩更新。

不用于未来产品运行时的 RL 训练记忆。

## 什么时候使用

在以下场景优先使用：

- 开始复杂开发任务前
- 做架构、路线、边界、优先级讨论前
- 进行容易导致上下文膨胀的实现或排障前
- 整理和压缩 `plans/` 记忆前

## 快速决策树

### 如果你在做项目级判断

例如：

- 路线讨论
- 架构调整
- 能力边界变化
- 优先级变化

那么：

1. 使用 `project_view`
2. 先读 `views.md`
3. 再按说明读取 `plans/memory_policy.yaml`、`plans/PROJECT_MEMORY.md`、`plans/NEXT_ACTIONS.md`
4. 默认不要读 `plans/worklog/inbox.md`

### 如果你在做具体任务收敛

例如：

- 开发功能
- 修 bug
- 排查验证失败
- 推进局部实现

那么：

1. 使用 `task_view`
2. 先读 `views.md`
3. 再按说明读取 `plans/memory_policy.yaml`、`plans/NEXT_ACTIONS.md`
4. 只有需要局部未决信息时才读 `plans/worklog/inbox.md`
5. 只有触及长期方向时才补读 `plans/PROJECT_MEMORY.md`

## 按需阅读索引

不要默认全读，只按需要读取：

- `views.md`
  - 视角选择规则
  - 每种视角该先读哪些记忆文件

- `memory-files.md`
  - `plans/` 中各记忆文件的职责边界

- `compression-rules.md`
  - worklog 怎么写
  - 结束后怎么压缩、删除、分流

## 最小工作流

1. 先选 `project_view` 或 `task_view`
2. 只读必要的 skill 附件和 `plans/` 文件
3. 工作过程先写 `plans/worklog/inbox.md`
4. 收尾时把稳定结论压缩进长期记忆或当前行动

## 强约束

- 不要默认读取所有 `plans/*` 文件
- 不要默认读取全部 skill 附件
- 不要把 worklog 当长期记忆
- 不要把过程噪音写入稳定记忆
- 只保留会改变未来决策的信息

## 目标结果

- 项目级问题使用高层稳定上下文
- 具体任务使用局部收敛上下文
- 原始过程有缓冲区可写
- 长期记忆保持短、小、稳
