# 10: 支持 HSDP 阶段训练

**What to build:** 训练操作者能够选择 HSDP，在同一次 draft 训练中组合参数分片和数据复制，并按固定混合拓扑卸载、持久化恢复及继续更新。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 09：通过 TorchTitan 组件支持复制 DP 与纯 FSDP2。

**Status:** ready-for-agent

- [ ] 通过保留训练入口完成八卡候选 C：DP replicate2 × shard4，TP/CP/PP/EP 均为 1；使用已建立的 draft TorchTitan 组件路径，target 继续采用独立固定配置。
- [ ] HSDP 的分片与复制归约共同实现同一 DSpark 梯度目标；microbatch 分母、GAS 和归约补偿各应用一次，不因多个 mesh 维度重复计算监督量。
- [ ] 真实 Qwen 小规模 FP32/BF16 对照包含不等有效分母、GAS 不小于 2 和至少两次 updates，验证全部可训练梯度、clip norm、master weights、Adam moments/计数及 scheduler。
- [ ] 验证 SAC 与混合归约下的梯度累积，保留冻结集合、参数和通信精度；比较时参考配置保有相同逻辑 microbatch/update 归属。
- [ ] 从完整 update 同步提交 DCP，卸载全部 draft GPU 状态，按相同 HSDP 拓扑恢复下一次 update；覆盖新进程恢复、RNG/游标连续以及多阶段残留显存检查。
- [ ] 交付可复现运行配置和实际八卡证据，说明合法 degree 与现有环境限制，阶段统计采用 draft 分项与总耗时口径。

