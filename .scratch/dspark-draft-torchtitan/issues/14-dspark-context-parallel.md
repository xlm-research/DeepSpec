# 14: 支持 DSpark 专用 CP 阶段训练

**What to build:** 训练操作者能够保持既有 target features 不变，将其交给 DSpark 专用 context parallel draft 训练，在长上下文分片下保留混合 attention 语义及完整阶段恢复能力。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 09：通过 TorchTitan 组件支持复制 DP 与纯 FSDP2。

**Status:** ready-for-agent

- [ ] 完成八卡候选 E：DP shard4 × CP2、TP1、PP1；允许沿用 DSpark 模型专用 attention 通信实现，保留 DeepSpec 外层循环与 draft 专用组件边界。
- [ ] 从 producer CP1 完整 features 以及既有 producer 分片重组后的 features 构造相同 CP 消费视图，验证 token 顺序、mask、位置、teacher hidden features、样本索引和恢复游标一致。
- [ ] 保持 teacher context 与 draft query 混合 attention、边界/padding 和 anchor 语义，CP 各独立监督分片共同组成正确的每 microbatch 全局分母。
- [ ] 为 CP 与 FSDP 实际归约规则匹配梯度缩放，保留 GAS 等权平均、冻结集合及参数/通信精度；各 rank 参数更新对应同一完整训练目标。
- [ ] 通过真实模型短序列 FP32/BF16 对照完整序列参考，覆盖不等有效分母、GAS 不小于 2 及至少两次 updates，比较各 loss、全部可训练梯度、clip norm 和 optimizer/scheduler。
- [ ] 完成 DCP 提交、所有 draft GPU 状态卸载、相同 CP 拓扑恢复与下一次 update；验证新进程恢复、RNG/样本连续、多阶段资源释放及所有消费者结束前缓存不被删除。
- [ ] 记录实际八卡配置、数据重分发成本、显存和 draft 总耗时；保留 native ring CP 与外层 model compile 的已有限制，不宣称本票消除了该限制。

