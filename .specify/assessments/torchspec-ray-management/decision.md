# Decision: 选择性改进 Ray 管理层

- **Slug**: torchspec-ray-management
- **Decided**: 2026-09-20
- **Verdict**: go
- **Artifacts reviewed**: intake.md、research.md、torchspec-evidence.md、problem.md、concept.md
- **Verdict scope**: 方案 B 值得进入规格阶段；本次不实施源码改造。整体迁移到上游训练栈不推荐。

## Scorecard

| Criterion | Rating | Justification |
|---|---|---|
| Problem validity | strong | 单机固定 4+4、集群强制一个 consumer 节点是源码明确的扩展边界。 |
| Evidence strength | strong | 双方固定版本源码与本地实际 H800 短程结果可核查；没有把未做的吞吐对照作为依据。 |
| Value vs. inaction | adequate | 增加训练节点和减少入口分化有价值；近期仅做原布局调试时收益较小。 |
| Feasibility / appetite | adequate | 方案 B 保留后端和训练语义，可先做现有布局等价迁移；多机验证仍有成本。 |
| Strategic fit | adequate | 符合本项目 DSpark/TorchTitan 训练目标；未假设存在新的集群规模承诺或正式 constitution。 |
| Risk posture | adequate | 已识别 PG 双重预留、rank 映射、128K 内存与恢复一致性风险，方案将其分开验收。 |

## Verdict & Rationale

**go：借鉴 TorchSpec 的拓扑与分组管理方式，进入当前项目管理层改造的规格阶段。**

TorchSpec 的管理入口更通用，可以独立声明角色规模并组织多个推理引擎和训练 rank。但 vLLM MP 后端、fractional actor 分配、数量背压、best-effort 恢复都不能视为对本地原生 Ray DP、TorchTitan 和强内存契约的全面升级。决策依据是可维护性与扩展能力，不是未经测试的性能收益。

整体迁移方案 C 当前不采用。现有单机/两机入口仍可用于调试，管理层改造不构成明天调试的前置条件。

## Handoff to Spec Kit Specify

- **Problem**: 扩大推理/训练布局能力，避免多个入口分别维护拓扑、资源 ownership 与失败清理规则。
- **Chosen approach**: concept B；统一计划和生命周期，保留 vLLM 原生 Ray 与每节点 torchrun/TorchTitan 执行。
- **In scope**: 明确受支持拓扑、资源所有权、组接口、每节点预算；先保留旧布局行为，再验证一种多 consumer 布局。
- **Out of scope**: 替换训练算法、直接拷贝上游 mp executor、自动弹性 world-size、GPU 轮换、同时引入 RDMA、未经样本提交协议的 actor 自动重启。
- **Success metrics**: 原有 4K/128K 正确性基线持续通过；实际 GPU/rank 与计划一致；所有样本归属和 ACK 正确；在途字节受控；失败有界退出且无本 run 残留。
- **Carried-forward open questions**: 下一阶段节点规模、vLLM 外部 PG 兼容方式、跨节点训练预算、长期恢复/评估需求，以及性能验收阈值。
- **Concrete requested change points**: 见 [comparison.md](comparison.md)，它是本次用户要求的改进建议，不是已经实施或批准排期的任务列表。

下一阶段输入可使用此 handoff 加 comparison 的 P0 条目生成正式规格；`go` 仅表示值得规格化，不表示完成了改造或部署。
