# 24: 联合启用 EP 与 expert FSDP

**What to build:** 操作者能选择至少一种 EP > 1 且 expert FSDP > 1 的合法布局，在专家分片与状态分片组合下完整续训。

**Blocked by:** 22：支持 GLM native EP8 阶段训练。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 在小规模真实 GLM 模型上确定并实现至少一种合法联合拓扑，明确 dense rank 域与 sparse mesh 映射；EP 不作为额外 world-size 乘数，继续使用 native dispatcher。
- [ ] 对照匹配基线验证 routed/shared experts、router、专家与 dense 归约、每 microbatch 分母及 GAS，包含不等分母和至少两个完整 updates。
- [ ] 核对全部专家/dense 梯度、clip norm、FP32 master/Adam/scheduler；不能因专家或 optimizer 分片改变精度、冻结集合或训练目标。
- [ ] 通过实际阶段入口提交全部 expert/dense shards 和恢复状态，退出所有 worker，同拓扑新进程恢复下一 update，验证 RNG/样本与通信资源释放。
- [ ] 给出具体可复现拓扑、规模、性能和限制；本票与 288 experts 扩展可独立验收，不自动宣称两者的任意联合规模已验证。

覆盖母规格 User Stories：38、39、40。

