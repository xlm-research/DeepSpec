# 27: 减少 DCP 保存与恢复的重复工作

**What to build:** 操作者能减少阶段 DCP 数据路径和重复初始化的开销，同时保留独立可恢复的完整提交。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 依据票 10 的分项计时，聚焦 DCP 写入/读取及会被恢复状态覆盖的重复模型初始化或权重读取；本票不混入 compile 或分区大小调优。
- [ ] 仍构建必要模型/optimizer 对象并完整恢复 frozen 参数、master/Adam、scheduler、RNG 和数据进度；不得以旧进程的 GPU 状态或外置 frozen 引用替代独立 DCP。
- [ ] 每阶段同步提交成功后才退出交接；若使用异步 API，实际等待/staging 成本仍计入保存，并保持最近两份保留和按需 HF 行为。
- [ ] 真实模型连续与跨进程恢复比较完整更新，改变初始化随机消耗不改变下一 anchors；真实保存/提交失败保留上一有效提交并阻止阶段推进。
- [ ] 比较固定工作量的全流程总耗时、分项、I/O、CPU/GPU 内存与波动，包含启动与通信重建，排除 target；只采用正确且有稳定收益的候选。

覆盖母规格 User Stories：17、18、19、43、44、46。

