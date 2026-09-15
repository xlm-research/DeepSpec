# 02: 通过 TorchTitan 完成 Qwen 单卡真实更新

**What to build:** DeepSpec 能启动独立 TorchTitan 进程，消费固定特征完成小规模 Qwen3.8 DSpark 训练；模型、训练配置、loss 和优化更新由 TorchTitan 拥有。

**Blocked by:** 01：复用数值基线，打通固定特征回放。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 只接通短序列、单卡、无外层 compile/AC 的最小真实模型路径；DeepSpec 传入训练配方与固定特征，TorchTitan 执行至少两个 optimizer updates 并返回可观察的进度与训练结果。
- [ ] 以原生 Config 和组件为来源接入 DSpark 模型、双输入、特征 reader、loss、FP32 master optimizer 及 scheduler；DeepSpec 不再通过旧 trainer 执行该新路径的 backward/update，也不复制同义训练默认值。
- [ ] 保持每 microbatch 全局加权均值再按 GAS 等权平均，保留 CE/L1/confidence、mask/位置衰减、epsilon/零分母、detached confidence target、anchors 和 feature/label 对齐。
- [ ] 固定冻结集合与参数/通信精度；冻结 LM head 仍把梯度传给可训练 hidden。FP32 和 BF16 结果与票 01 的匹配参考比较全部更新及优化状态。
- [ ] 使用已有环境与 draft 专用入口完成增量接入；固定 target 配置和 vLLM 启动来源不随 draft 配方变化，旧权重初始化不冒充旧训练任务完整迁移。
- [ ] 本票交付新任务的完整训练更新；完整阶段 DCP 与再次启动续训由票 04 交付，不宣称当前已具备阶段恢复。

覆盖母规格 User Stories：1、2、4、5、6、7、9、51。

