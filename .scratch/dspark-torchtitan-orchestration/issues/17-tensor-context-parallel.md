# 17: 联合启用 TP 与 CP

**What to build:** 操作者能在 TP4 × CP2 的八卡布局中消费同一监督，完成双输入训练与同拓扑跨阶段续训。

**Blocked by:** 16：支持 DSpark 专用 CP 阶段训练。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 交付候选 F：TP4 × CP2，DP replicate/shard 和 PP 均为 1；复用票 09/10 已验证的 TP 路径与票 16 的 CP 路径。
- [ ] 明确双输入、TP peers、CP 分片及监督 view 的联合布局，producer 配置不变，独立监督分母和 GAS 缩放恰好应用一次。
- [ ] 真实 Qwen 短序列比较匹配参考的各 loss、全部可训练梯度、clip norm、master/Adam/scheduler，包含不等分母和至少两个 updates。
- [ ] 联合布局完成完整 DCP、全体 worker 退出、同拓扑新进程恢复下一 update，样本/RNG/游标连续，多阶段无 draft 资源残留增长。
- [ ] 记录实际八卡结果及 layout 读取/通信成本，不把单独 TP 或 CP 通过当作联合配置证据。

覆盖母规格 User Stories：32、34、40。

