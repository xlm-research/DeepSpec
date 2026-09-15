# 22: 支持 GLM native EP8 阶段训练

**What to build:** 操作者能在八卡 native expert parallel 下训练真实 GLM DSpark，并完整保存、退出及恢复专家与 dense 状态。

**Blocked by:** 21：通过 TorchTitan 完成 GLM DSpark 基础阶段。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 从小规模真实 GLM 配置交付首个候选：DP shard8、EP8，其他并行轴和 expert TP 均为 1，使用 native dispatcher；专家数满足合法性要求。
- [ ] dense rank 域提供 sparse 视图，EP 不再额外乘入 world size；保持 routed/shared experts、router、dispatch/combine 和 token 路由的既定行为。
- [ ] 按实际 dispatcher 与 sparse mesh 推导专家梯度缩放，并与 dense 参数归约分别验证；保留 microbatch 全局分母及 GAS 等权平均，不照搬旧 pure-EP 倍率。
- [ ] 对照票 21 匹配语义的 GLM 基线，覆盖各 loss、所有专家/dense 梯度、clip norm、master/Adam/scheduler、不等分母、GAS ≥ 2 和至少两个 updates。
- [ ] 完整 DCP 包含全部 expert shards、dense/frozen 参数及恢复状态；所有 worker 退出，专家通信和预取资源释放，同拓扑新进程恢复下一 update，保存失败不推进 target。
- [ ] 记录实际八卡、多阶段资源、SAC 兼容性与性能分项；native dispatcher 通过不表示 DeepEP 或其 SAC 组合已通过。

覆盖母规格 User Stories：8、37、38、40。

