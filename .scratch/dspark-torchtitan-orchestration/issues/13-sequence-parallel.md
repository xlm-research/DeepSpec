# 13: 支持 TP 下的 Sequence Parallel

**What to build:** 操作者能在已通过首验的 TP 布局中单独开启 SP，减少激活复制并保持完整阶段训练和恢复。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** in-progress

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 在候选 D（DP shard2 × TP4）开启 SP，先使用真实 Qwen 短序列，覆盖双输入、norm/残差与相关 heads 的布局。
- [ ] 对照 SP 关闭的匹配参考，保持样本、逻辑 microbatch、GAS、分母、RNG 和优化配方；验证 FP32/BF16、各 loss、全部梯度及至少两个完整 updates。
- [ ] 完成统一阶段入口的 DCP、所有 worker 退出和同拓扑新进程恢复，下一 update、scheduler/RNG/消费游标连续。
- [ ] 验证 target 生产配置与监督内容不变，记录真实激活显存、数据布局成本及实际八卡证据。
- [ ] 本票验收 SP 单独启用，SP 与完整 loss parallel 的联合行为由票 15 验证。

覆盖母规格 User Stories：32、40。

