---
status: accepted
date: 2026-09-14
---

# TorchTitan 接入优先保持 DSpark 的训练语义

引入 TorchTitan 组件时，以当前 DeepSpec 工作树中的 DSpark 训练语义为兼容基线：每个 microbatch 先计算跨数据分片的 loss 加权均值，再对梯度累积窗口中的 microbatch 等权平均。用户已确认优先与 DSpark 保持一致；当各 microbatch 的有效权重和不同时，改用 TorchTitan 默认的整步有效 token 平均会改变训练目标，因此该改变须作为独立的算法决策讨论。

## 适配约束

- 保留 CE、概率分布 L1、confidence BCE 的定义、系数、有效位置 mask、位置衰减权重及 confidence target 的 detach 语义。分母按实际 draft 监督位置的权重求和。
- 保留每个 microbatch 的全局分母及一次 optimizer step 的 GAS 平均。若采用本地 TorchTitan 的 FSDP 梯度求和约定，dense 参数路径移除旧 loss 中抵消梯度平均的倍率；若保留当前梯度平均约定，则保留该倍率。采用匹配的一套 loss 与归约规则。
- 保留 anchor 采样、teacher feature 与 label 的对齐、冻结参数集合、优化器精度及学习率调度语义。适配组件时对照参数更新结果。
- 用户已同意将 target 缓存分区边界调整到完整 optimizer step 后，保持样本顺序、GAS 和每次 update 的 microbatch 分组，允许分区大小和数量随之变化。正常阶段切换不保存半步累计梯度，也不为提前卸载而缩短 GAS、提前更新或额外丢弃样本；保留当前每 epoch 洗牌后按完整 global batch 截断的取样规则。完整训练状态卸载与恢复遵循 [ADR-0003](0003-unload-draft-between-training-phases.md)。
- TP 副本不增加样本分母。MoE 专家参数按实际 dispatcher 与 sparse mesh 的归约规则单独推导缩放，保证其对同一训练目标的梯度贡献；旧 pure-EP 的补偿倍率仅适用于旧通信约定。

在梯度求和约定下，对 rank r、microbatch m 和启用的 loss 项 k，目标适配式为：

```text
D[m,k] = sum(该 microbatch 所有独立监督分片上的有效权重)
local_loss[r,m] = sum_k alpha[k] * S[r,m,k] / (D[m,k] + epsilon) / GAS
```

其中 S 是本地加权 loss 分子。各 loss 项沿用当前 epsilon 和零分母处理；此式描述 dense 参数的目标缩放，专家通信路径需另行验证。

## 验收依据

实施前固化当前工作树的可复现基线，固定 teacher 特征、初始权重、输入、anchors、GAS 和并行拓扑，对照各 loss 项、关键参数梯度及一次 optimizer update。必须包含各 microbatch 有效分母不同的案例；所选路径还需覆盖阶段内多 microbatch 累积、分区对齐前后的 update 分组、连续训练与阶段卸载恢复后的更新对照，以及启用 MoE 时的专家梯度。具体模型、拓扑和数值容差在验收方案中确定。

训练职责的最新归属见 [ADR-0004](0004-separate-feature-delivery-and-draft-training.md)：TorchTitan 拥有 draft 训练循环和训练组件配置，DeepSpec 调度两端、决定训练多少批数据并控制资源卸载；该决定替代了 [ADR-0002](0002-retain-dspark-loop-for-torchtitan-components.md) 中 DeepSpec 持有训练循环的约定。本 ADR 的 DSpark 数学语义继续作为兼容基线，本次先用 Qwen3.8 DSpark dense draft 验证。首阶段包含 SelectiveAC、阶段卸载恢复，以及单机八卡真实 draft 尺寸与 128K 输入验收；后续覆盖 dense 并行组合和 GLM-5.3-Flash DSpark MoE EP，具体矩阵见 [设计草案](../../doc/qwen38_torchtitan_component_design.md)。用户要求在保持上述训练语义的前提下尽量提高性能。
