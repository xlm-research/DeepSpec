# 09: 通过 TorchTitan 组件支持复制 DP 与纯 FSDP2

**What to build:** 训练开发者能够在保留的 DSpark 循环中使用 TorchTitan 组件运行复制 DP 与纯 FSDP2，并完成与首阶段一致的持久化、卸载和恢复流程。以 spmd_types 为主要后端建立后续并行能力共用的 draft 适配路径。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 08：完成八卡真实 Qwen 128K 首阶段验收。

**Status:** ready-for-agent

- [ ] 以票 08 的真实 128K 首阶段通过为交付前置；该依赖落实母规格的阶段顺序，不将尚未验收的首阶段标作完成。
- [ ] 使用现有环境及固定 TorchTitan 参考版本，适配集中在 draft 专用入口；DeepSpec 保有训练循环与生命周期，各模型、mesh、同步操作和状态对象有明确管理方，不重复初始化或归约。
- [ ] 完成八卡候选 A（DP replicate8、shard1、TP1、CP1、PP1）和 B（replicate1、shard8、其余轴为 1），先用真实 Qwen 小规模短序列验证，再记录支持限制。
- [ ] 为实际梯度求和或平均约定匹配 DSpark 补偿，保持每逻辑 microbatch 独立全局分母及 GAS 等权平均；各归一化和归约补偿恰好应用一次，冻结集合、参数/通信精度及 FP32 master optimizer 保持基线。
- [ ] 每个候选使用匹配自身逻辑 microbatch 语义的参考，覆盖不等有效分母、GAS 不小于 2、至少两次 updates，并比较各 loss、全部可训练梯度、clip norm、master weights、Adam 及 scheduler。
- [ ] 每个候选均通过训练入口完成完整 update、DCP 提交、完整 GPU 卸载、相同拓扑恢复及下一次 update；验证 RNG/样本进度连续、新进程恢复和无逐阶段残留增长。
- [ ] 记录真实 GPU/rank 数、拓扑、配置、容差依据、阶段耗时和显存；固定 producer 配置，不以不同 DP degree 或 GAS 的直接跨行比较证明相同训练目标或性能优势。

