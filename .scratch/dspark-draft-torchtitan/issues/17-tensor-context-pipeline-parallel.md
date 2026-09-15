# 17: 支持 TP、CP、PP 联合训练

**What to build:** 训练开发者能够在一个合法的 Qwen draft 配置中联合使用 TP、CP 和 PP，完成从固定监督读取到完整更新、持久化和阶段恢复的路径。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 15：支持 TP 与 CP 联合训练；16：在现有训练循环中支持两阶段 PP 1F1B。

**Status:** ready-for-agent

- [ ] 完成八卡候选 H：DP replicate1、shard1、TP2、CP2、PP2，采用已建立的双输入布局、DSpark CP 和 PP 1F1B；说明真实模型的 stage 划分。
- [ ] 在 TP/CP/PP 交界处保持 teacher context、draft query、必要监督和可微 context 梯度的正确布局；每个阶段和 shard 消费同一逻辑样本的对应视图。
- [ ] loss 分母仅计独立监督位置，PP stages 与 TP 副本不重复计数，实际归约补偿及 GAS 平均恰好应用一次，pipeline 切块保持逻辑 microbatch 归属和 RNG。
- [ ] 使用与候选 H 逻辑 microbatch 匹配的真实 Qwen 短序列参考，覆盖不等有效分母、GAS 不小于 2 和至少两次 updates，比较全部 loss、可训练梯度、clip norm、master weights、Adam/scheduler。
- [ ] 完整 update 后排空 pipeline 与所有通信，同步提交完整 DCP 并释放所有 stage 的 draft GPU 状态；同拓扑恢复下一次 update，覆盖新进程、RNG/数据游标一致和多阶段无残留增长。
- [ ] 既有 producer 完整或分片 features 可通过索引/reader 供联合布局消费，缓存等待全部消费者完成后清理，target 环境、生产配置和监督内容保持不变。
- [ ] 记录实际八卡证据、双输入 stage/layout 限制、配置及 draft 分项/总耗时；不从本票通过推断外层 compile 或任意 SP/loss parallel 组合已通过。

