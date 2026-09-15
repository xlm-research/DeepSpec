# 01: 复用数值基线，打通固定特征回放

**What to build:** 训练开发者能用固定 target features 驱动现有真实 Qwen DSpark 训练并得到可供新入口比较的完整更新结果；必要的局部预重构先在这一条可运行路径完成。

**Blocked by:** None (can start immediately)。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 核对旧票 01–05 的实际基线、fixture 和验收记录，记录当前工作树差异及现有解释器、PyTorch/CUDA、源码编译 vLLM 和 TorchTitan 参考版本；未受影响的证据直接复用，不重做已完成工作。
- [ ] 仅在现有阶段或固定特征入口确实不能复用时进行最小局部预重构；预重构前后沿用相同生产训练循环，以真实模型更新结果证明行为不变。
- [ ] 基线包含固定初始权重、样本顺序、tokens/masks/features、anchors/RNG、逻辑 microbatch/GAS、冻结集合、精度和拓扑，能够由新入口读取同一份输入并比较结果。
- [ ] 复用或补齐小规模真实 Qwen FP32/BF16、至少两个完整 updates 的对照，包含不等有效分母、GAS ≥ 2、零分母和边界情况；多卡用例记录实际 rank 数。
- [ ] 对照结果覆盖各 loss、全部可训练参数梯度、clip norm、FP32 master weights、Adam/scheduler 和消费位置；按既有基线误差确定容差，不能把历史通过、skip 或 toy 当作新架构通过。

覆盖母规格 User Stories：3、41、42、47、48、65。

