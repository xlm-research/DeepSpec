# 23: 扩展至 GLM 288 个 routed experts

**What to build:** 操作者能将已验证的 native EP 配方扩展到真实 288 个 routed experts，并得到实际规模的训练与阶段恢复证据。

**Blocked by:** 22：支持 GLM native EP8 阶段训练。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 从票 22 的小规模真实模型逐级扩大专家规模，覆盖 288 个 routed experts，明确其余模型尺寸、序列长度及实际 GPU/内存需求，不仅检查整除条件。
- [ ] 使用已验证的 native dispatcher 和 EP 拓扑，保留 routed/shared/router 语义及完整训练配方；本票不同时扩展新的 expert FSDP 拓扑。
- [ ] 沿用小规模严格更新等价证据，新增实际规模完整 update、DCP 提交、全部 worker 退出、同拓扑恢复及下一 update 验证。
- [ ] 校验全部专家与 dense/frozen/optimizer 状态完整、RNG/消费位置连续、多阶段资源释放和保存失败保护。
- [ ] 记录实际 rank 数、模型/上下文、分项性能与资源限制；不能用缩减专家数量或修改训练目标标记规定规模通过。

覆盖母规格 User Stories：37、38、39、40。

