# 16: 支持 DSpark 专用 CP 阶段训练

**What to build:** 操作者能将固定 target features 交给 DSpark context parallel 训练，在上下文分片下保留混合 attention 和完整阶段恢复。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 交付候选 E：DP shard4 × CP2，TP1/PP1；允许延续 DSpark 专用 attention 通信，通过 TorchTitan 拥有的训练入口运行。
- [ ] 从 producer CP1 完整 features 以及既有 producer shards 重组得到正确 CP 视图，保持 token 顺序、mask、位置、teacher context、draft query、anchors 和消费游标。
- [ ] CP 独立监督分片共同形成每 microbatch 分母，匹配实际 FSDP/CP 归约并保留 GAS 等权平均，不重复计入 TP/其他副本。
- [ ] 真实 Qwen 短序列 FP32/BF16 对照完整序列参考，覆盖边界/padding、不等分母、GAS ≥ 2、全部梯度和至少两个 updates。
- [ ] 通过同一阶段入口完成 DCP、全部 worker 退出、同拓扑新进程恢复和下一 update；相关消费者结束且提交前不得清理缓存。
- [ ] 记录实际八卡、数据重分发与完整阶段成本；保留 native ring CP 与外层 model compile 的既有限制，不宣称消除未验证兼容性问题。

覆盖母规格 User Stories：29、30、34、40。

