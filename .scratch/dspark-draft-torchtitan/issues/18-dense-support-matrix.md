# 18: 完成 SP 与 loss parallel 联合验收及 dense 支持矩阵

**What to build:** 训练操作者能够同时启用 SP 与完整词表并行 loss，并依据有实际证据的 dense 并行支持矩阵选择配置；不支持的组合能在昂贵训练启动前被识别。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 10：支持 HSDP 阶段训练；12：支持 TP 下的 Sequence Parallel；13：支持完整 DSpark 词表并行 loss；17：支持 TP、CP、PP 联合训练。

**Status:** ready-for-agent

- [ ] 在候选 D（DP shard2 × TP4、CP1、PP1）上完成 SP 与完整 DSpark loss parallel 同时启用的真实训练，对照两者关闭、仅 SP、仅 loss parallel 的对应结果。
- [ ] 联合配置覆盖 CE、分布 L1、acceptance/confidence、全部可训练梯度和完整 optimizer 更新；包含不等分母、GAS 不小于 2 及至少两次 updates，保留冻结 head 梯度和无完整 logits 收集的 loss 路径。
- [ ] 联合配置完成 DCP 提交、完整 GPU 卸载、相同拓扑恢复及下一次 update，覆盖新进程恢复、RNG/样本连续和多阶段资源释放。
- [ ] 汇总候选 A（replicate8）、B（shard8）、C（replicate2 × shard4）、D（shard2 × TP4）、E（shard4 × CP2）、F（TP4 × CP2）、G（shard4 × PP2）、H（TP2 × CP2 × PP2）的逐行配置及真实更新和恢复证据；未注明的轴为 1。
- [ ] 区分已验证通过、具体失败和未测试状态，记录每行模型规模、上下文、SAC/SP/loss parallel/compile 开关、实际 rank 数、容差和资源限制；各候选先短序列验收，不能宣称所有行都支持 128K。
- [ ] 在用户配置验证中识别非法 world size、head/degree、stage 划分及不支持的变换组合，明确真实 Qwen TP8、默认 PP8 和 native ring CP 加外层 compile 的限制，不能静默降级后报告原配置通过。
- [ ] 沿用已有独立能力证据，新增联合配置及受改动影响的回归验证；不同 DP/GAS 配置分别使用匹配参考，不将跨行吞吐差异直接视为同一更新目标下的收益。
- [ ] 交付可复现的 dense 支持矩阵及启动方式；任何候选未完成必须明确记录原因并处理，不能以修改状态表代替能力交付。本票通过后进入 MoE EP 阶段。

