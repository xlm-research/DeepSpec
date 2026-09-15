# 05: 启用 SelectiveAC 并保持阶段续训等价

**What to build:** 训练操作者能够在新的 TorchTitan Qwen FSDP2 阶段流程中启用 SelectiveAC，保持更新与跨进程恢复结果。

**Blocked by:** 04：提交完整 DCP，退出并跨进程续训。

**Status:** completed

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [x] 通过原生 AC 配置对真实 DSpark blocks 应用 SelectiveAC；每个 block 采用一套 checkpoint 策略，不叠加 HF 与 TorchTitan checkpoint 包装。
- [x] 先关闭外层 model compile，在两卡 FSDP2 下比较 AC 关闭与开启，覆盖 FP32/BF16、GAS ≥ 2、不等有效分母和至少两个 updates。
- [x] 比较 loss、全部可训练梯度、clip norm、master/Adam/scheduler 和重计算 RNG；保留票 01 的匹配训练语义与容差。
- [x] 从同一 DeepSpec 阶段入口完成 DCP、全部 worker 退出、新进程同拓扑恢复及下一 update；重建后 AC 配置和随机序列保持一致。
- [x] 记录实际显存和重计算成本，给出合法配置与运行证据；本票只证明基础 AC 行为，性能策略调优由后续票分别验证。

覆盖母规格 User Stories：10、19、40。


验收证据：`doc/benchmarks/dspark_native_draft.md` 的 SelectiveAC 章节；`selective-ac-state-comparison-final.log` 四个 rank/dtype 逐位通过，`checkpoint-selective-ac-test.log` FP32/BF16 全过程 1149.779 秒通过，无 skip。
