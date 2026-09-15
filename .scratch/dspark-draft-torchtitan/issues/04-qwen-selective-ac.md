# 04: Qwen 接入 TorchTitan SelectiveAC

**What to build:** 训练开发者能够在现有 Qwen3.8 DSpark 与 FSDP2 路径中选择 TorchTitan SelectiveAC，观察重计算带来的显存与计算取舍，同时保持当前训练更新语义。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 01：固化 DSpark 数值基线与最小训练验证入口。

**Status:** completed

- [x] 以现有环境及固定 TorchTitan 参考版本完成实际组件兼容性核对；发现需要更换环境基线时带具体冲突返回设计讨论，不通过重装源码编译 vLLM 或修改 target 绕过。
- [x] 在 draft 专用 activation checkpoint 入口接入 SelectiveAC，随后沿用现有 FSDP2 入口；首次构建使用的适配方式可供后续阶段重建复用。
- [x] 每个 block 使用一套 activation checkpoint 策略，避免叠加 HF 内置 checkpoint 与 TorchTitan 包装；首次正确性对照关闭外层 model compile，并保持重计算所需 RNG 行为。
- [x] 保留当前 FSDP 梯度平均约定及匹配的 DSpark 归约补偿，microbatch 分母、GAS 平均和补偿各应用一次，不改变 loss、冻结集合及精度。
- [x] 通过真实 Qwen 小规模 FP32/BF16 对照，覆盖两个 rank 的 FSDP、GAS 不小于 2、不等分母及至少两次更新；比较各 loss、全部可训练参数梯度、clip norm、master weights 和 Adam/scheduler 状态。
- [x] 在单机八卡 DP shard2 × TP4、CP1 的短序列候选上验证真实更新，记录实际 rank 数、耗时与峰值显存；完整阶段卸载和真实尺寸 128K 由后续票验收。
- [x] 启用该 draft 策略时现有 target 生产配置、运行环境与监督内容保持一致，不通过让 target 使用相同 AC 设置解决角色隔离问题。


实施提交：`4d9949a`。验收记录：`doc/benchmarks/dspark_selective_ac.md`。
