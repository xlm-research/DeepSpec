# Specification Quality Checklist: 统一 Ray 拓扑与生命周期，扩展多节点 draft 训练

**Purpose**: 检查规格的完整性、边界与可验证性，作为澄清和技术方案阶段的输入。

**Created**: 2026-09-20

**Last Reviewed**: 2026-09-20；按最终M2/M3澄清及analyze修订重新审阅。

**Feature**: [spec.md](../spec.md)

**Review Ownership**: 此检查表由specify / clarify工作流维护；本次按用户授权的analyze问题修订同步复核范围、验收编号与相关质量勾选项。

**Marker Semantics**: `[x]` 仅表示规格质量准则已通过审阅，不表示实现完成或运行验收通过。

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- **结果**：16/16项规格质量准则已复核；规格保留Draft状态，plan/tasks已生成并修订。该结果只评价规格质量，实施仍从未勾选任务开始。
- **技术约束解释**：Ray、vLLM、TorchTitan、字节预算是用户明确要求保留的现有边界；FR-019/020 记录这些约束。规格没有规定代码结构、类/API、具体 actor 粒度或强制 PG 组织方式，后续技术方案负责选择实现。
- **读者与术语**：以训练任务维护者的用户场景表达价值，Scope 和 Key Entities 定义 TP/DP、训练参与者、读完与训练提交等必要术语；无需先阅读实现代码。
- **首版范围**：M0/M1保持兼容。M2为两个推理节点各8 GPU及一个训练节点8 GPU，共24 GPU；推理TP8×DP2、每副本两端各4卡，训练TP4×DP2，并有推理DP1的4/4/8 GPU对照。M3为一个推理节点4 GPU和两个训练节点各4 GPU，共12 GPU；推理TP4×DP1、跨训练节点TP4×DP2，必须独立验收。A1是用户已确认的范围，A7的实际资源可用性仍待核验；更大M2推理DP仅在两端额度足够时接受声明，须另取实测证据。新增拓扑、恢复、弹性和传输优化不在本次交付范围。
- **可验证性**：SC-001–010覆盖计划校验、真实训练、资源隔离、样本/更新一致性、容量、故障退出、诊断和证据范围。SC-002规定M0的4K/128K、M1四种DP组合的4K及DP1/DP1、DP2/DP2的128K回归；SC-003验收M2，SC-010独立验收M3的4K/128K及跨训练节点故障，不能互相替代。
- **依赖覆盖**：FR-020/021与A2/A3/A7明确环境、模型/数据、网络/存储和两种三节点布局的资源依赖；没有把资源就绪当作已知事实。运行证据采集T079/T080已前置到首次真实验收之前。
- **状态与所有权覆盖**：FR-001/005/006/007/016 与 US1/US5/US6 覆盖运行身份、资源归属、终态及不可达节点回收不确定性。
- **数据与提交覆盖**：FR-010–014/017/019/021与US2/US3/US4、SC-002/003/005/006/010区分固定样本计划、特征读取、容量退还和完整训练提交；通用计数从冻结计划推导，12样本/3更新只是验收实例。
- **验收声明边界**：“Feature meets measurable outcomes” 在本表表示规格中的场景/要求足以定义和核查这些结果；本次没有执行模型训练、多节点故障注入或性能对照。
- **Hooks**：项目不存在 `.specify/extensions.yml`，无 before_specify / after_specify hooks；未创建或切换 Git 分支。
