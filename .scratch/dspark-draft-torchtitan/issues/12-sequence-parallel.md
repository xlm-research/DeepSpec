# 12: 支持 TP 下的 Sequence Parallel

**What to build:** 训练开发者能够在 Qwen TP 训练上单独开启 Sequence Parallel，减少相应激活复制，并在同一 DSpark 训练和阶段恢复契约下继续更新。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 11：适配 Qwen 双输入 TP 训练。

**Status:** ready-for-agent

- [ ] 在八卡候选 D（DP shard2 × TP4、CP1、PP1）上单独启用 SP，并保留关闭 SP 的同拓扑基线；loss parallel 首先关闭，二者组合由票 18 验收。
- [ ] teacher context 与 draft query 的序列布局、残差、norm、Markov/confidence 与输出 head 在各边界正确转换；tokens、mask、hidden features 与样本顺序保持一致。
- [ ] SP 的通信和参数梯度缩放保留每逻辑 microbatch 的 DSpark 分母及一次 GAS 平均，不把 TP peers 当作新增独立监督样本。
- [ ] 通过真实 Qwen FP32/BF16 短序列训练比较启用前后的各 loss、全部可训练梯度、clip norm、master weights 和 Adam/scheduler，覆盖不等有效分母、GAS 不小于 2 及至少两次 updates。
- [ ] 完成启用 SP 的完整 update、DCP 提交、GPU 卸载、相同拓扑和 SP 配置恢复及下一次 update，验证 RNG/样本连续、新进程恢复与多阶段资源释放。
- [ ] 交付实际八卡证据、可复现配置、合法布局限制及分项耗时/显存；target 的生产布局与环境保持固定。

