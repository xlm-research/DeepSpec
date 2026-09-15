# 15: 支持 TP 与 CP 联合训练

**What to build:** 训练开发者能够同时使用 TP 和 CP，通过同一份固定 target 监督完成双输入 DSpark 训练，并在联合拓扑下保存、卸载和继续恢复训练。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 11：适配 Qwen 双输入 TP 训练；14：支持 DSpark 专用 CP 阶段训练。

**Status:** ready-for-agent

- [ ] 完成八卡候选 F：DP replicate1、shard1、TP4、CP2、PP1，使用已验收的 TP 与 DSpark CP 路径。
- [ ] feature reader 对 TP peers 及 CP 分片生成正确双输入视图；attention、residual/norm、Markov/confidence 和输出 head 之间的布局转换保留样本、token 和监督对应关系。
- [ ] 分母只覆盖独立监督位置，TP 复制不重复计数，CP 与 FSDP 的实际梯度归约和 GAS 缩放匹配；不以跨拓扑 global batch 相等替代逻辑 microbatch 等价。
- [ ] 真实 Qwen FP32/BF16 短序列对照与候选 F 匹配的参考，包含不等有效分母、GAS 不小于 2 及至少两次 updates，比较各 loss、全部梯度、clip norm、master weights、Adam/scheduler。
- [ ] 联合布局完成完整 update、同步 DCP 提交、全部 GPU draft 状态卸载、相同拓扑恢复及下一次 update；覆盖新进程恢复、RNG/索引游标连续和多阶段显存释放。
- [ ] 保持 target 配置、生产布局及原始缓存不变，验证缓存等待全部相关消费者完成后才清理；交付实际 rank 数、可复现配置和 draft 分项/总耗时。
- [ ] 明确本票验证的 SAC、SP、loss parallel 和 compile 开关，不将独立轴通过推断为任意组合通过，保留 native CP 的 compile 限制。

