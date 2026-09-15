# 21: 通过 TorchTitan 完成 GLM DSpark 基础阶段

**What to build:** MoE 训练开发者能在 TorchTitan 中使用小规模真实 GLM-5.3-Flash DSpark 模型完成无专家分片的训练和阶段恢复，为 EP 提供可比较基线。

**Blocked by:** 20：交付 dense 支持矩阵与配置校验。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 仅接通小规模真实 GLM DSpark、EP1 的固定 features 阶段流程，使用现有 GLM 模型语义与 TorchTitan 原生配置/训练组件；不能以 Qwen 或 toy MoE 代替。
- [ ] 保留 routed/shared experts、router、原有训练项、teacher feature/label 对齐及 DSpark CE/L1/confidence 目标，不修改 GLM target 或 vLLM 环境。
- [ ] 固定初始权重、features、RNG、GAS、冻结与精度，对照匹配 GLM 参考的各 loss、router/shared/routed 梯度、clip norm、master/Adam/scheduler，包含不等分母和至少两个 updates。
- [ ] 通过统一 DeepSpec 阶段入口提交完整 GLM 状态，全部 worker 退出并释放资源，新进程同拓扑恢复后 RNG/样本与下一 update 连续。
- [ ] 将旧 GLM swap/journal 证据用于适配参考，给出新 TorchTitan 入口自己的数值、恢复与失败保护证据；本票不宣称 native EP 已支持。

覆盖母规格 User Stories：1、2、37、38、41、51。

