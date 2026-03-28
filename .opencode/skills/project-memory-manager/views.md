# 视角选择

本文件用于说明什么时候应该使用 `project_view`，什么时候应该使用 `task_view`。

## 1. project_view

适用于：

- 项目整体路线讨论
- 架构设计或架构调整
- 能力边界变化
- 优先级变化
- 长期规划压缩
- 重要记忆更新

先读：

- `plans/memory_policy.yaml`
- `plans/PROJECT_MEMORY.md`
- `plans/NEXT_ACTIONS.md`

默认不读：

- `plans/worklog/inbox.md`

理由：

- 这一视角需要稳定、高层、压缩后的项目开发记忆
- 不应把过程噪音带进项目级判断

## 2. task_view

适用于：

- 具体编码任务
- bug 修复
- 验证失败排查
- 某个局部实现收敛
- 当前步骤的具体执行

先读：

- `plans/memory_policy.yaml`
- `plans/NEXT_ACTIONS.md`

按需读：

- `plans/worklog/inbox.md`
- `plans/PROJECT_MEMORY.md`

理由：

- 这一视角关注当前任务局部收敛
- 只需要最小的任务相关上下文
- 项目级长期记忆只在确实触及方向、边界、约束时再补充

## 3. 选择规则

如果问题主要在问：

- 这个项目下一步该往哪走
- 边界该不该改
- 路线是不是要调整

用 `project_view`。

如果问题主要在问：

- 这个具体功能怎么做
- 这个 bug 怎么修
- 这个局部实现接下来怎么收敛

用 `task_view`。

如果不确定：

- 先用 `task_view`
- 只有在发现问题触及长期方向时，再补读 `PROJECT_MEMORY.md`

## 4. 防上下文爆炸原则

- 不要因为文件存在就全部读取
- 先选视角，再最小读取
- `project_view` 防止任务细节淹没路线判断
- `task_view` 防止路线材料淹没具体实现
