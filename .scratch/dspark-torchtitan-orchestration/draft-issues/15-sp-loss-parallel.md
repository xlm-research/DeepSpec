# 15: 联合启用 SP 与词表并行 loss

**What to build:** 操作者能同时启用 SP 和完整 DSpark loss parallel，得到与分别开启或关闭时一致的阶段更新和恢复结果。

**Blocked by:** 13：支持 TP 下的 Sequence Parallel；14：支持完整 DSpark 词表并行 loss。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 在候选 D 上比较两者关闭、仅 SP、仅 loss parallel、两者开启四种配置，固定同一 logical microbatch 和训练配方。
- [ ] 联合布局保持双输入与 heads 的梯度路径，CE/L1/confidence 语义完整，且 loss 不重新引入完整 logits 收集。
- [ ] 真实 Qwen 对照不等分母、GAS ≥ 2、至少两个 updates 的 loss、全部梯度、clip norm 与 master/Adam/scheduler。
- [ ] 联合配置完成 DCP、全部 worker 退出与同拓扑新进程恢复，验证下一 update、RNG/游标和资源释放。
- [ ] 只补测联合变化及实际回归风险，沿用未受影响的单能力证据，记录开关适用范围及实际八卡结果。

覆盖母规格 User Stories：32、33、40。

