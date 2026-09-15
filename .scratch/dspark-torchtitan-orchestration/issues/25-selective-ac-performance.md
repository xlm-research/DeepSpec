# 25: 测量并调优 SelectiveAC 保存策略

**What to build:** 性能工程师能在固定八卡 Qwen 工作负载上选择有实际收益的 SAC 保存策略，降低 draft 总成本并保留阶段恢复正确性。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 以票 10 的完整流程为基线，固定模型、features、逻辑 microbatch/GAS、精度、optimizer/LR 和 target 配置，仅比较 SAC 保存策略这一类因素。
- [ ] 记录候选的训练、保存、退出、启动恢复及总 wall time、峰值显存和重复测量波动，包含实际重计算、初始化/编译成本，排除 target 生产与等待。
- [ ] 拟采用策略通过真实模型不等分母、GAS ≥ 2、至少两次更新的 loss/全梯度/master/Adam/scheduler 对照，并验证重计算 RNG。
- [ ] 策略经完整 DCP、所有 worker 退出、同拓扑新进程恢复下一 update 与多阶段资源验证，保存失败保护保持有效。
- [ ] 仅采用有数值和稳定总耗时收益的策略；无收益时保留基线并交付测量结论，明确适用模型与拓扑范围。

覆盖母规格 User Stories：43、44、46。

