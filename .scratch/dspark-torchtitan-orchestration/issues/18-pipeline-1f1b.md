# 18: 支持 TorchTitan 两阶段 PP 1F1B

**What to build:** 操作者能在 TorchTitan 内部使用两阶段 1F1B 流水线训练真实 DSpark，DeepSpec 调度完整特征分区并在保存退出后继续下一阶段。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 交付候选 G：DP shard4 × PP2，TP1/CP1；真实 Qwen 五个 draft layers 采用合法两阶段划分，不要求等分或承诺 PP8。
- [ ] 流水线传播 draft query、可微 teacher-derived context、相应梯度和必要监督输入，不能只自动切 decoder layers 而遗漏反向路径。
- [ ] TorchTitan 拥有 schedule、GAS、跨参数梯度裁剪、optimizer/scheduler、进度和 checkpoint；物理流水线 microbatch 保持原 logical microbatch 的样本、分母、GAS 与 RNG，PP stages 不增加分母。
- [ ] 真实 Qwen 短序列对照非 PP 的匹配参考，覆盖不等分母、多 microbatches、全部梯度/master/Adam/scheduler 和至少两个 updates。
- [ ] 完整 update 后排空流水线与在途通信，再提交所有 stage 的完整 DCP，退出全部 worker；同 PP 拓扑的新进程恢复下一 update。
- [ ] 验证所有 stage 的资源释放与保存失败阻止交接，记录实际八卡、多阶段和分项耗时证据。

覆盖母规格 User Stories：35、36、40。

