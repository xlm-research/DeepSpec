# 20: 交付 dense 支持矩阵与配置校验

**What to build:** 操作者能依据有真实更新及恢复证据的 dense 矩阵选择配置，并在昂贵训练启动前识别非法或未支持组合。

**Blocked by:** 12：支持 HSDP 阶段训练；15：联合启用 SP 与词表并行 loss；19：联合启用 TP、CP 与 PP。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 汇总候选 A（replicate8）、B（shard8）、C（replicate2 × shard4）、D（shard2 × TP4）、E（shard4 × CP2）、F（TP4 × CP2）、G（shard4 × PP2）、H（TP2 × CP2 × PP2）及 D 的 SP/loss parallel 组合；未写轴为 1。
- [ ] 每个宣称支持的配置有真实阶段入口的完整更新、DCP、全部 worker 退出、同拓扑恢复和下一 update 证据；沿用已有证据，仅补测缺口及受改动影响的组合。
- [ ] 配置校验通过实际用户启动入口拒绝非法 world size、head/degree、stage 划分和已知不兼容变换，说明原因，不静默降级后报告原配置通过。
- [ ] 记录各行模型规模、上下文、实际 rank 数、精度/GAS、AC/SP/loss parallel/compile 开关、容差和资源限制；短序列通过不等于 128K 通过，不能把跨行 GAS 变化当作同目标加速。
- [ ] 区分通过、具体失败和未测状态；规定的候选若缺失，必须解决或明确提交范围变更待决定，不能只改状态表就将本票记为完成。
- [ ] 交付可复现选择与启动方式，按既有交付顺序完成 dense 验收后进入 GLM MoE 阶段。

覆盖母规格 User Stories：39、40、47、48。

