# 统一 Ray 拓扑实施与验收报告

**状态：实施进行中，尚未达到 SC-001–010 的完整交付条件。**

用户已确认当前只有一个物理节点。M1–M3 的真实多节点验收继续受阻；单节点测试、CPU 探针和实现完成均不替代真实训练验收。统一六个 CLI 命令、冻结计划、精确资源登记、原生 vLLM 借用 PG、M3 双 launcher、独立 CPU DCP 核验与逐节点观测已有实现。

20:55 的[最终资源检查](../../outputs/ray-topology-acceptance/resource-inventory/cc43fd07dc434890a22050be6dea8f0b.json)确认外部 Ray 仍有一个存活节点，八个 GPU 进程均不属于本次运行，本次 GPU 进程残留为零。

## 正常训练矩阵

下表只选训练层或资源清单层的最新记录；同名 CPU 契约的 passed 不计入训练通过。所有失败和阻塞记录仍保留在追加索引中。

| Case | 最新记录状态 | 证据层级 | 证据 |
|---|---|---|---|
| m0-4k | failed | training | [0ccb0c559ed7](../../outputs/ray-topology-acceptance/m0-4k/0ccb0c559ed74f1da7689635217d08e8/result.json) |
| m0-128k | blocked | inventory | [9df3d74b54a6](../../outputs/ray-topology-acceptance/m0-128k/9df3d74b54a644d9862796bc54e0a483/result.json) |
| m1-11-4k | blocked | inventory | [19ec09eb54ff](../../outputs/ray-topology-acceptance/m1-11-4k/19ec09eb54ff4b2ebc18540be3b63f63/result.json) |
| m1-11-128k | blocked | inventory | [58c208d8b347](../../outputs/ray-topology-acceptance/m1-11-128k/58c208d8b34749d7b8314144d2277735/result.json) |
| m1-12-4k | blocked | inventory | [f6fca04a7293](../../outputs/ray-topology-acceptance/m1-12-4k/f6fca04a72934b3fb82e14bcd9ffc7f0/result.json) |
| m1-21-4k | blocked | inventory | [4ff6951700ed](../../outputs/ray-topology-acceptance/m1-21-4k/4ff6951700ed49559c6770369d77dea9/result.json) |
| m1-22-4k | blocked | inventory | [a7a17a56d300](../../outputs/ray-topology-acceptance/m1-22-4k/a7a17a56d3004238a6f19058a36efff4/result.json) |
| m1-22-128k | blocked | inventory | [3c63e3788558](../../outputs/ray-topology-acceptance/m1-22-128k/3c63e378855846069b210435cea561b5/result.json) |
| m2-dp1-4k | blocked | inventory | [2fba2f77b058](../../outputs/ray-topology-acceptance/m2-dp1-4k/2fba2f77b05843d2bc60db519e39509f/result.json) |
| m2-dp2-4k | blocked | inventory | [fc8df71c9397](../../outputs/ray-topology-acceptance/m2-dp2-4k/fc8df71c9397461d827a542ec5d99223/result.json) |
| m2-dp2-128k | blocked | inventory | [b4a557d485a3](../../outputs/ray-topology-acceptance/m2-dp2-128k/b4a557d485a34bf295624e6826d60c9b/result.json) |
| m3-4k | blocked | inventory | [5b556478c3a5](../../outputs/ray-topology-acceptance/m3-4k/5b556478c3a540f0b9469c998781d561/result.json) |
| m3-128k | blocked | inventory | [4981ccb93bd5](../../outputs/ray-topology-acceptance/m3-128k/4981ccb93bd543b5990c83e65a832445/result.json) |

M0-4K 最新重试已结束，运行目录为 [98186a02…](../../outputs/ray-topology-acceptance/native-m0/98186a02a69541f389dd6c7039a1ca42/status.json)。启动时八卡空闲，随后外部 CI 再次占用训练卡；门禁在模型初始化前拒绝继续。实际分配和传输探针通过，但训练没有产生样本或更新。六个清理阶段全部通过，独立观察的八项资源均为 released，[资源清理](../../outputs/ray-topology-acceptance/native-m0/98186a02a69541f389dd6c7039a1ca42/resource-cleanup.json)和 [driver supervisor](../../outputs/ray-topology-acceptance/native-m0/98186a02a69541f389dd6c7039a1ca42/driver-request.cleanup.json)均确认完成；外部任务未被终止。继续验证修复需要能覆盖初始化和训练的八卡空闲时段。M0-128K 须等 M0-4K 通过；旧/新入口真实等价对照也尚未通过。

## 已完成的验证及实际修复

- [完整回归](../../outputs/ray-topology-acceptance/execution-contracts/1556031ca4eb4c45b27284cc8722d6e7/result.json)：366 passed、1 deselected；后续原生初始化、训练握手、状态、地址固定等[定向回归](../../outputs/ray-topology-acceptance/execution-contracts/ea02cd3639e54482b7071454b87fdbfb/result.json)：124 passed。两组选取有重叠，不相加。
- 真实八个 CPU rank 的 Store 数据探针回归通过：[日志](../../outputs/ray-topology-acceptance/plan-data-probe-regression-final.log)。
- 单节点八 rank CPU Gloo [正常通信](../../outputs/ray-topology-acceptance/collective-probe/e292f5715268491f83f0e0059767ea79/result.json)与[rank-7 故障传播/清理](../../outputs/ray-topology-acceptance/collective-probe/6ba57f023c114dbc92671f178105cd8e/result.json)通过。它们不加载模型、不使用 GPU，不证明跨节点行为。
- 真实 M0 无模型分配、部分分配失败回滚和 TCP Store 写/读/删除探针通过；这些不等于训练通过。
- 原生运行暴露的旧预算字段、初始化失败后的 Store 关闭、Ray 结构化异常序列化问题已修复。`auto` Ray 地址在首次解析后固定，避免后续检查连接临时测试集群。
- 前次原生训练已加载四个 vLLM workers 和四个 TorchTitan ranks，但握手因 `spmd_types` 使用 `dp_shard` 而非 `fsdp` 失败。现按原生后端选择 mesh 名称，并在两道 gate 等待期间立即传播角色失败；CPU 回归通过，最新真实重试在更早的资源门禁失败，尚未证明这两项修复的真实训练闭环。旧[失败记录](../../outputs/ray-topology-acceptance/m0-4k/6cd54a78c40941edb36efa728823e506/result.json)及[资源清理](../../outputs/ray-topology-acceptance/native-m0/e5c001bd0eb74a2ab1e7aee53e5c0b28/resource-cleanup.json)保留。
- Ruff 和 `git diff --check` 通过。[vLLM 类型检查对照](../../outputs/ray-topology-acceptance/type-check/comparison.json)：基线 17 项、当前 11 项、无新增；正式 mypy 仍未通过。CPU DSpark 数学/保存选择为 1 passed、3 skipped，不能算 GPU 数学回归通过。

## FR / SC 追踪

| 要求 | 当前实现与未完成验证 |
|---|---|
| FR-001–003 | 身份、冻结计划、v3 配置与 CPU 预览契约已实现；全参与节点 preview 前后独立快照仍须补齐。 |
| FR-004 | 旧入口转入同一 controller，显式参数迁移及冲突测试通过；旧/新真实等价对照未完成。 |
| FR-005–008 | 精确资源登记、全角色 gate、所有权及有限等待已实现；M0 实际分配/部分回滚通过，多节点缺证据。 |
| FR-009、021 | M2 TP8 和 M3 共享原生训练世界已有接缝/握手测试；三节点执行、跨节点 Gloo 与故障验证 blocked。 |
| FR-010–014 | 计划驱动样本/reader、预算、ACK/删除和训练 commit 分离已实现并有 CPU 契约；128K 实际压力未完成。 |
| FR-015–016 | fail-fast、取消、watchdog、subreaper 与精确回收有轻量真实进程探针；真实训练故障矩阵未完成。 |
| FR-017 | CPU DCP 范围、初始参数哈希、optimizer/cursor、全 rank commit 及释放核验已实现；原生训练完整闭环待通过。 |
| FR-018 | 实际事件源、独立采样和只读 status 已接通；正常/背压/失败现场交叉核验待补齐。 |
| FR-019–020 | 保持原生 vLLM/TorchTitan 数学路径，运行前重验源码/依赖/模型/输入身份；全矩阵数学与保存证据未完成。 |

SC-001 有 CPU 配置/预览证据但全参与节点快照未齐；SC-002/003/005/010 缺必需真实训练；SC-004/006/007 有轻量证据但缺指定训练故障和 128K 压力证据；SC-008 缺全部现场指标回放。SC-009 要求区分 blocked/failed/not_run，本报告保留证据层级，不以 CPU 契约或历史单机成功代替本轮训练验收。

[任务清单](tasks.md)为 66/89 完成，将实现与真实验收分开标记；[实施记录](implementation-baseline.md)保留复现、修复和验证细节；[验收总索引](../../outputs/ray-topology-acceptance/results.json)保留所有历史尝试。本次运行与观察进程均已结束。M0 待持续空闲的八卡时段恢复；多节点部分需真实资源后按 [quickstart](quickstart.md) 顺序推进 4K、128K 和隔离故障。
