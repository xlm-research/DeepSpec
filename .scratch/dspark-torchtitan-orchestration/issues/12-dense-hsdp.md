# 12: 支持 HSDP 阶段训练

**What to build:** 操作者能选择复制与参数分片组合的 HSDP，并通过真实阶段流程恢复后继续等价更新。

**Blocked by:** 11：支持八卡复制 DP 与纯 FSDP2。

**Status:** completed

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [x] 交付候选 C：DP replicate2 × DP shard4，TP/CP/PP 均为 1；明确原生 mesh 与训练状态的归属，不重复初始化或归约。
- [x] 真实 Qwen 短序列 FP32/BF16 对照匹配参考，覆盖不等 microbatch 分母、GAS ≥ 2 和至少两个 updates，核对 loss、全部梯度、clip norm 与完整优化状态。
- [x] 通过实际阶段入口提交 HSDP 完整状态，全部 worker 退出并释放资源，同拓扑新进程恢复后样本/RNG/scheduler 连续且下一 update 等价。
- [x] 验证保存失败仍阻止交接，HSDP 配置不改变固定 target producer；恢复仅支持同一任务的相同拓扑。
- [x] 记录实际八卡证据、性能分项和资源限制，复用 DP/FSDP 已验证的公共契约。

覆盖母规格 User Stories：31、40。


## 实施与验收（2026-09-15）

- 配方：`qwen38_27b_hsdp`；八卡 replicate2 × shard4。
- 真实 Qwen DSpark 五层、24Q/4KV 的小规模 FP32/BF16，两个 updates、GAS2；完整输出、梯度和优化状态均匹配独立参考。
- 两种精度均完成连续、保存退出、同拓扑新进程恢复及实际 DCP 提交失败测试。恢复后的完整 checkpoint 与下一 update 精确一致；所有本任务 worker 均已退出。
- 证据：`outputs/dspark_dense_12_19_20260915_v2/hsdp/` 各阶段 `complete.json`。其中部分阶段与外部 GPU 作业共存，运行器单独记录其占用；不据此宣称吞吐收益。
- 详见 `doc/benchmarks/dspark_native_dense_12_19.md`。
