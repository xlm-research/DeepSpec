# 28: 验证 compile 在阶段重启后的实际收益

**What to build:** 操作者能在合法 draft 布局中判断 compile 及跨进程缓存是否降低总耗时，而不是只改善热身后的 step 速度。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 仅选择已合法的模型/拓扑候选，固定票 10 的工作量与训练配方；明确采用的 compile 范围，不默认启用与 native ring CP 不兼容的外层编译。
- [ ] 通过真实训练进程分别测量首次编译、后续阶段重新启动的缓存命中与加载、通信重建和执行成本，不能用理论缓存支持代替观察结果。
- [ ] 拟采用设置通过真实模型 loss、全部梯度和 master/Adam/scheduler 对照，含不等分母、GAS ≥ 2、至少两个 updates 与随机操作。
- [ ] 完成完整 DCP、全部 worker 退出、新进程同拓扑恢复下一 update，样本/RNG/进度连续，缓存不持有旧训练 GPU 状态。
- [ ] 报告各阶段总 wall time 与分项、显存和重复测量波动，排除 target 生产/等待；无稳定全流程收益或不兼容时保留基线并记录原因。

覆盖母规格 User Stories：43、44、45、46。

