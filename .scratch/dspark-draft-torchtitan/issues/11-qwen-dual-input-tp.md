# 11: 适配 Qwen 双输入 TP 训练

**What to build:** 训练开发者能够在新的 draft 组件路径中使用 Qwen 双输入 TP，让 teacher context、draft query 和输出 heads 共同完成等价训练，并在固定 TP 拓扑下恢复后继续更新。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 09：通过 TorchTitan 组件支持复制 DP 与纯 FSDP2。

**Status:** ready-for-agent

- [ ] 通过既有训练入口完成候选 D 的新组件 TP 路径：DP shard2 × TP4、CP1、PP1；与首阶段现有 TP 路径使用相同逻辑 microbatch 配方对照。
- [ ] 适配 teacher context 与 draft query 双输入布局、attention、残差、norm、Markov/confidence 以及输出 head；特征 reader 向 TP peers 提供正确样本视图，TP 副本不增加 loss 分母。
- [ ] 冻结 LM head 保持冻结，同时向可训练 draft hidden 回传正确梯度；参数、通信精度、FP32 master optimizer、anchors 和各项 DSpark loss 语义不变。
- [ ] 按真实 Qwen 的 24 个 Q heads 和 4 个 KV heads 验证合法 TP1/2/4，非法 head/degree 组合在昂贵训练启动前被识别；不宣称 TP8 或自动 KV 复制已支持。
- [ ] 真实模型 FP32/BF16 短序列对照覆盖不等有效分母、GAS 不小于 2 和至少两次 updates，比较各项 loss、全部可训练梯度、clip norm 和完整 optimizer/scheduler 更新。
- [ ] 完成 DCP 提交、全部 draft GPU 状态卸载、相同 TP 拓扑重建及下一次 update；覆盖新进程恢复、RNG/样本连续与多阶段显存释放，SAC 在重建前后保持一致。
- [ ] 记录每种实际验证的 degree、rank 数与可复现配置；本票允许沿用完整 logits 的基准 loss 路径，完整词表并行 loss 由票 13 验收。

