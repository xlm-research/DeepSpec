# 26: 测量并调优 FSDP reshard 与预取

**What to build:** 性能工程师能根据端到端计时选择 FSDP reshard 和预取设置，同时保持更新、保存退出与恢复语义。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 以票 10 的固定配方为基线，分别评估 child reshard、forward/backward 预取及深度，每轮能够区分变动因素与实际收益。
- [ ] 保持样本、anchors、逻辑 microbatch/GAS、训练目标、精度与优化配方，不通过改变 target 推理获得收益。
- [ ] 真实模型比较不等有效分母及至少两个 updates 的 loss、全部梯度、clip norm、master/Adam/scheduler；拟采用设置验证完整阶段恢复。
- [ ] 多阶段检查预取输入、参数与通信资源随全体 worker 退出释放，保存失败不推进 target，下一进程不依赖旧 GPU 状态。
- [ ] 报告训练/保存/退出/启动恢复的分项与总 wall time、显存及波动，包含初始化/编译而排除 target 时间；无稳定收益可保留基线。

覆盖母规格 User Stories：43、44、46。

