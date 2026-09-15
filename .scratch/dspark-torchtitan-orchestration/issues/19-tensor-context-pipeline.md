# 19: 联合启用 TP、CP 与 PP

**What to build:** 操作者能在一个合法的 TP/CP/PP 八卡布局中完成真实 DSpark 更新、所有 stage 的保存退出和新进程恢复。

**Blocked by:** 17：联合启用 TP 与 CP；18：支持 TorchTitan 两阶段 PP 1F1B。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 交付候选 H：TP2 × CP2 × PP2，DP replicate/shard 均为 1；联合配置明确双输入的 stage/layout、监督所有权及可微 context 的反向路径。
- [ ] 固定 producer features、逻辑 microbatch 和训练配方，对照匹配参考，验证分母、GAS 与 TP/CP/PP 归约没有重复。
- [ ] 通过真实 Qwen 短序列至少两个 updates，对照各 loss、全部 stage 的梯度、clip norm、master/Adam/scheduler，覆盖不等有效分母。
- [ ] 排空 pipeline 和通信后同步提交所有状态，全体 worker 退出并释放资源，同拓扑新进程恢复下一 update 且 RNG/样本连续。
- [ ] 覆盖联合保存失败与多阶段资源释放，记录实际八卡和运行限制；单能力证据不代替本联合布局验收。

覆盖母规格 User Stories：35、36、40。

