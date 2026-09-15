# 03: 通过 TorchTitan 完成两卡 FSDP2 更新

**What to build:** 同一新训练入口能够使用两卡 FSDP2 消费固定监督，得到与 DSpark 基线等价的完整更新。

**Blocked by:** 02：通过 TorchTitan 完成 Qwen 单卡真实更新。

**Status:** completed

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [x] 通过原生并行配置接入 draft 专用 FSDP2，使用两个实际 GPU/rank、短序列真实 Qwen 模型和至少两个 updates；通用并行适配以规格选定的 spmd_types 方向为准。
- [x] 根据实际梯度平均或求和约定匹配 loss 缩放；每 microbatch 分母、GAS 等权平均和归约补偿各应用一次，不直接沿用旧倍率。
- [x] 覆盖 GAS ≥ 2、不等有效分母、冻结 head 的反向梯度及参数/通信精度；不通过修改 logical microbatch 或 local/global batch 配方换取等价。
- [x] FP32/BF16 比较各 loss、全部可训练梯度、clip norm、master weights、Adam/scheduler 和两次 updates；保存实际软件版本、拓扑与容差依据。
- [x] 训练进程能报告完整 update 进度并正常退出；不改变 target 生产配置，也不以只完成 forward/backward 或梯度存在性证明训练正确。

覆盖母规格 User Stories：8、27、31、41。


实现与验收：2026-09-14 已通过真实单卡/双卡 FP32、BF16 两次更新对照，GAS=2，无 skip；见 [原生 draft 数值证据](../../../doc/benchmarks/dspark_native_draft.md)。完整阶段恢复另由票 04 验收。
