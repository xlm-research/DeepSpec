# Problem: 保持训练语义的同时扩大集群布局能力

- **Slug**: torchspec-ray-management
- **Created**: 2026-09-20
- **Inputs**: [intake.md](intake.md)、[research.md](research.md)、[torchspec-evidence.md](torchspec-evidence.md)

## Problem Statement

当前 Ray streaming 训练已在目标 H800 上完成短程正确性验证，但资源和训练拓扑固定在单机 4+4 GPU、或两机分别承载 producer/consumer 的形态。后续增加 draft 训练节点、改变推理与训练资源比例时，用户需要跨入口修改配置和生命周期逻辑，容易让 GPU 归属、样本归属及内存预算失去一致性。证据：research L1–L5。

## Affected Users & Current Behavior

项目调试者、训练任务维护者是直接使用者。当前可以继续完成已有布局的调试；并无证据说明现有训练必须立即重写。影响主要出现在扩大布局、长期运营和故障后继续训练的需求上。故障恢复目前明确不支持，不能用“已有完整 checkpoint”推导“已有流式恢复”。证据：research L7–L9。

## Goals

1. 用户能明确知道哪些节点/GPU 分别属于推理、训练和数据服务，声明的布局在模型加载前即可判断是否支持。
2. 单机与多机行为采用一致的启动、就绪、失败、退出规则；资源不足和初始化失败可诊断并有界退出。
3. 扩展训练节点时保留 DSpark loss、TorchTitan 并行/优化器及固定样本计划语义。
4. 长序列训练的资源上限覆盖实际在途特征，不能以更通用调度为由弱化字节预算和读者确认。
5. 借助可比较的等待/计算/传输指标决定后续性能工作，而不是根据框架接口数量判断速度。

## Non-Goals

- 本评估不实现运行代码，也不承诺某一规模的吞吐提升。
- 不迁移到上游 DSpark trainer，不扩大已经验证的 TP/CP/PP 支持矩阵。
- 不把在线弹性 world-size、GPU 复用轮换、RDMA 调优作为首轮成功条件。
- 自动故障恢复不是“打开 Ray actor restart”即可完成的目标，需要独立的数据提交语义。

## Baseline & Success Metrics

| 项目 | 当前基线 | 后续改造验收要求（建议） |
|---|---|---|
| H800 正确性 | 4K/128K 各 12 样本、3 更新、48 rank 读取校验，独立 checkpoint 核验通过 | 原有布局和原有语义继续通过；不得跳过失败样本而报告成功 |
| 训练拓扑 | 集群入口 consumer_nodes=1，TP4 / DP1 或 2 | 支持的多 consumer 节点布局须有真实多机验证；不支持的布局在分配 GPU 前失败 |
| GPU 归属 | 显式 PG、namespace 和进程归属检查 | 实际设备集合与计划一致、角色不重叠、无重复资源预留 |
| 特征容量 | window + bytes，删除成功后退还 | admitted bytes 不越界，归属读者未读完不得删除；完整更新组可推进 |
| 退出 | 有 deadline、清理和残留验证 | startup/运行/退出故障均在配置超时内结束；仅回收本 run 资源 |
| 性能 | 无两项目同条件对照 | 对照至少记录 feature tokens/s、optimizer step time、训练等待占比、resident/host/GPU 峰值，注明模型与数据相同 |
| 恢复 | 流式入口 fail-fast，无特征重放 | 若实施恢复，必须证明 checkpoint 后样本重放与 optimizer 提交边界一致，另设门槛 |

前三项已有明确源码和运行证据；多机扩展与性能目标属于待验证要求，不是本次新增测试结果。

## Cost of Inaction & Open Questions

保持现状对明天的单机调试可接受；如果之后仍只使用现有两种布局，重构收益有限。如果要增加 draft 节点，则现有硬编码会成为直接限制。

下一阶段具体节点数、卡型、最长序列、长期保存频率与恢复时间目标尚未指定。它们影响后续规格和验收规模，不阻碍本次比较与分阶段建议。
