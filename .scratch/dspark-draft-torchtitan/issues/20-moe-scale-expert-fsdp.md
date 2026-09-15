# 20: 扩展 MoE 规模并验证 EP 与 expert FSDP 组合

**What to build:** 训练操作者能够将已验证的 GLM native EP 扩展到更大的专家配置，并选择经过实测的 EP 与 expert FSDP 联合布局，在相同布局的各阶段之间持续恢复训练。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 19：GLM DSpark 接入 native MoE EP。

**Status:** ready-for-agent

- [ ] 从票 19 的小规模真实 GLM 基线逐级扩大专家规模，覆盖真实 288 个 routed experts 的配置并记录其余模型尺寸、序列长度和实际资源需求；不能仅凭专家数整除判定布局适配完成。
- [ ] 至少验证一种 EP 大于 1 且 expert FSDP 大于 1 的合法联合拓扑，明确 dense rank 域与 sparse 视图的映射，EP 不是新增 world-size 乘数；默认延续 native dispatcher。
- [ ] 对每种验收布局验证 routed/shared experts、router、专家归约和 dense 归约共同得到原 DSpark 目标；固定监督样本、逻辑 microbatch、精度和 optimizer 配方，不能通过改变 GAS 权重或参数冻结规避资源问题。
- [ ] 先以可进行严格数值对照的真实模型规模验证各 loss、全部专家/dense 梯度、clip norm、FP32 master/Adam/scheduler 及至少两次 updates，再扩大规模；较大规模结果不能替代小规模等价证据。
- [ ] 每种宣称支持的布局完成真实 update、同步提交完整专家与 dense DCP、完整 GPU 状态卸载、相同拓扑恢复及下一次 update；验证新进程恢复、RNG/游标连续和专家通信缓冲释放。
- [ ] 记录专家 shard 与 optimizer state 的完整恢复证据、多阶段显存、draft 分项及总耗时、实际 GPU/rank 数和可复现运行方式。
- [ ] 交付有实际证据的 MoE 支持矩阵、资源限制和具体不兼容原因；不同运行之间可以选择不同合法布局，同一运行阶段间保持拓扑不变，不隐含任意 dispatcher/TP/CP/PP 交叉支持。

