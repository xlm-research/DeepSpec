# 30: 完成并行与性能配置的最终联合交付

**What to build:** 维护者获得可复现的 Qwen dense 与 GLM MoE 训练交付，能够从支持矩阵选择实际验证过的并行和性能配置，完成完整阶段续训。

**Blocked by:** 23：扩展至 GLM 288 个 routed experts；24：联合启用 EP 与 expert FSDP；25：测量并调优 SelectiveAC 保存策略；26：测量并调优 FSDP reshard 与预取；27：减少 DCP 保存与恢复的重复工作；28：验证 compile 在阶段重启后的实际收益；29：按完整 update 调整分区大小并比较成本。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 汇总已验收的 dense A–H、SP/loss parallel、GLM native EP、288 experts 和至少一种 EP × expert FSDP 布局，并明确模型规模、上下文、开关及限制；首阶段八卡真实 Qwen 128K 证据仍有效。
- [ ] 对最终拟交付的优化组合验证真实阶段入口的固定-feature 更新、DCP、全体 worker 退出、同拓扑新进程恢复与下一 update；复用未受影响证据，只补测新组合和实际回归风险。
- [ ] 验证最终组合的故障恢复、提交与编排进度核对、RNG/样本/scheduler 连续、缓存生命周期、最近两份 checkpoint 与可用 HF 导出，且 target/vLLM 配置保持既定基线。
- [ ] 报告匹配工作量下训练、保存、退出、启动恢复的分项及总 wall time，包含读取/重分发、通信重建、初始化、实际编译与 HF 导出，排除 target 生产与等待。
- [ ] 交付实际 GPU/rank、软件构建、模型/拓扑、精度/GAS、容差、资源与重复测量证据及可复现启动方式；不以 CPU/toy/skip、未经测量的加速比或所有组合均支持 128K 的推断作为通过。
- [ ] 所有承诺能力由对应票真实交付；缺项须解决或明确提出范围调整，不能只改完成状态。最终只推荐有更新正确性、恢复和性能证据的组合，不要求任意后端/dispatcher/并行轴的笛卡尔积。

覆盖母规格 User Stories：2、3、39、40、43、44、45、46、47、48、65。

