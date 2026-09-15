# 09: 支持 Qwen 双输入 TP4 阶段训练

**What to build:** 新训练入口能在固定 target 特征上运行 Qwen 双输入 TP，并完成 DP shard2 × TP4 的短序列更新、保存退出和恢复，为八卡 128K 首验提供前置能力。

**Blocked by:** 04：提交完整 DCP，退出并跨进程续训。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 从小规模真实模型验证 TP1/TP2，再运行八卡 DP shard2 × TP4、CP1/PP1；保留真实 Qwen 的 GQA 约束，不承诺 TP8。
- [ ] 布局覆盖 teacher context、draft query、attention、残差/norm、Markov/confidence 与输出 head；TP peers 的特征视图来自同一消费计划，不增加独立样本分母。
- [ ] 以匹配逻辑 microbatch 的参考比较各 loss、全部可训练梯度、clip norm、master/Adam/scheduler，覆盖不等分母、GAS ≥ 2 和至少两个 updates；并行缩放与实际归约一致。
- [ ] 通过统一阶段入口提交全部 TP/FSDP 状态，所有 worker 退出后以相同拓扑新进程恢复下一 update；验证 RNG、样本和 optimizer 连续、资源释放及保存失败不推进。
- [ ] producer CP1 与既有分片特征重组均保持输入语义；改变 draft TP 不改变 vLLM 生产配置或缓存所有权。
- [ ] 本票基础 loss 允许正确的完整 logits 路径；完整 vocabulary-parallel loss 另由票 14 交付，不能把基础 TP 记作 loss parallel。

覆盖母规格 User Stories：8、29、32、40。

