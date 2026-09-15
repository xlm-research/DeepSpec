# 14: 支持完整 DSpark 词表并行 loss

**What to build:** 操作者能在 TP 阶段训练中启用完整词表并行监督，保持 DSpark 更新并避免为 loss 收集完整词表 logits。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** in-progress

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 在候选 D 的真实 Qwen 上覆盖全词表归一化下 CE、概率分布 L1、acceptance/confidence 与相关 head 语义，包括 detached confidence target、mask/位置衰减和零分母。
- [ ] 与完整 logits 参考比较各 loss、全部可训练梯度、clip norm、master/Adam/scheduler，覆盖不等有效分母、GAS ≥ 2、FP32/BF16 和至少两个 updates。
- [ ] 冻结 LM head 仍向 hidden 传播正确梯度；TP 副本不增加样本分母，vocab 归约与监督/数据归约分别正确。
- [ ] 通过真实阶段入口完成 DCP、全体 worker 退出、同 TP/FSDP 拓扑恢复下一 update，RNG、样本及完整状态保持连续。
- [ ] 以执行 trace 或通信/内存证据证明训练 loss 不收集完整词表 logits；临时 gather 的正确性实现可用于对照，但不构成本票完成。
- [ ] 记录实际八卡配置、内存和分项耗时；本票验收 loss parallel 单独启用，SP 联合配置由票 15 验证。

覆盖母规格 User Stories：33、40。

