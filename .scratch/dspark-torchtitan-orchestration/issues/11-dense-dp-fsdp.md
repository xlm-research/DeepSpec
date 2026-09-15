# 11: 支持八卡复制 DP 与纯 FSDP2

**What to build:** 首阶段完成后，操作者能通过同一 TorchTitan 阶段入口分别选择八卡复制 DP 或纯 FSDP2，并在各自拓扑内保存退出和续训。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** in-progress

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 交付候选 A（DP replicate8）和 B（DP shard8），其他轴为 1；使用原生配置与 draft 适配，DeepSpec 仍只编排阶段。
- [ ] 每种配置从固定 features 跑通真实 Qwen 短序列至少两个 updates；各自使用匹配 logical microbatch/GAS 的参考，不将不同 DP 分组当成同一训练目标。
- [ ] 验证不等分母、GAS ≥ 2、精度、冻结集合、loss 及全部梯度/master/Adam/scheduler；复制与分片归约各采用正确且唯一的补偿。
- [ ] 每行都完成完整 DCP、全部 worker 退出、同拓扑新进程恢复下一 update，覆盖资源释放、RNG/游标连续及保存失败不推进。
- [ ] 记录实际八卡配置、运行与数值证据、分项性能和限制；本票不自动宣称两行均支持 128K。

覆盖母规格 User Stories：31、40。
