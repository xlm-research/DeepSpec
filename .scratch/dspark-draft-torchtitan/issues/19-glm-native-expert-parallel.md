# 19: GLM DSpark 接入 native MoE EP

**What to build:** MoE 训练开发者能够通过 GLM-5.3-Flash DSpark 的真实模型类使用 native expert parallel，在专家通信下保持原训练目标，并完整保存、卸载和恢复专家及 dense 训练状态。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 18：完成 SP 与 loss parallel 联合验收及 dense 支持矩阵。

**Status:** ready-for-agent

- [ ] 以票 18 的 dense 阶段验收完成为交付前置，先使用真实 GLM-5.3-Flash DSpark 模型类的小规模配置与 native dispatcher，不能以 Qwen dense 或 toy MoE 通过替代。
- [ ] 完成首个八卡候选：DP shard8、EP8，DP replicate/TP/CP/PP/expert TP 均为 1；dense rank 域提供 sparse 视图，EP 不再额外乘入 world size。
- [ ] routed experts、shared experts、router、dispatch/combine、token 路由及模型原有相关训练项保留基线行为；小规模专家数满足所选 EP degree 的合法性要求。
- [ ] 依据实际 dispatcher 和 sparse mesh 推导专家参数梯度缩放，并与 dense 参数归约分别验证；不照搬旧 pure-EP 补偿，保留每 microbatch 全局分母及 GAS 等权平均。
- [ ] 固定 features、初始化和 RNG，与匹配逻辑 microbatch 的 GLM 参考比较各 loss、router/shared/routed experts 的全部可训练梯度、clip norm、master weights、Adam/scheduler；包含不等有效分母、GAS 不小于 2 及至少两次 updates。
- [ ] 完整 DCP 覆盖全部专家 shards、dense/frozen 参数及完整 optimizer/RNG/样本状态，可独立恢复；提交后释放参数、master/moments、专家通信缓冲和预取输入。
- [ ] 通过相同拓扑的新进程恢复及多阶段循环验证下一次 update 等价、数据/RNG 连续、完整资源释放和保存失败不推进 target 交接。
- [ ] 固定现有 GLM target/vLLM 环境、生产配置及监督内容，记录实际八卡证据、SAC 兼容性、可复现配置和分项耗时；本票不以 native dispatcher 结果宣称 DeepEP 或其 SAC 组合已支持。

