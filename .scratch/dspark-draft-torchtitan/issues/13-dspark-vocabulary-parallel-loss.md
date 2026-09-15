# 13: 支持完整 DSpark 词表并行 loss

**What to build:** 训练开发者能够在 TP 下使用完整 DSpark 词表并行监督，无需为训练 loss 收集完整词表 logits，即可保留 CE、分布 L1、acceptance/confidence 和相应梯度。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 11：适配 Qwen 双输入 TP 训练。

**Status:** ready-for-agent

- [ ] 在候选 D 的真实 Qwen 模型上首先单独启用 loss parallel，SP 关闭，以相同输入、权重和逻辑 microbatch 的完整 logits 路径作为数值参考。
- [ ] 跨词表 shards 保留全词表归一化，正确计算 CE、概率分布 L1、acceptance/confidence target、confidence BCE 及相关 head 语义；保留系数、mask、位置衰减、epsilon、零分母处理与 confidence target detach。
- [ ] 训练 loss 与相关监督计算不依赖收集完整 draft 或 target 词表 logits；只支持 CE 或通过 full-logit gather 过渡不能视为本票完成。
- [ ] 保留冻结 LM head 对 draft hidden 的梯度、全部 trainable head 梯度以及各词表 shard 的参数更新；TP 副本不增加样本分母，各 microbatch 分母与 GAS 补偿恰好应用一次。
- [ ] 真实模型 FP32/BF16 对照包含不等有效分母、边界/零分母、启用和禁用可选 loss 项、GAS 不小于 2 及至少两次 updates；比较全部 loss、梯度、clip norm、master weights、Adam/scheduler。
- [ ] 完成完整 update、DCP 提交、全部 draft GPU 状态卸载、同拓扑恢复及下一次 update，覆盖新进程恢复和 RNG/数据游标连续。
- [ ] 记录实际八卡证据、通信/峰值显存与 draft 分项及总耗时，说明对完整词表收集的实际消除情况；SP 与 loss parallel 联合使用由票 18 验收。

