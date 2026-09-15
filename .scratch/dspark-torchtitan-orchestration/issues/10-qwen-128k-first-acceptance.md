# 10: 完成八卡真实 Qwen 128K 首验

**What to build:** 训练操作者获得可复现的单机八卡 Qwen3.8 DSpark 128K 配方，包含 SelectiveAC、真实特征交接、完整保存退出恢复和 draft 全流程性能基线。

**Blocked by:** 05：启用 SelectiveAC 并保持阶段续训等价；07：中断后核对提交与编排进度；08：保留最近两份恢复点并按需导出 HF；09：支持 Qwen 双输入 TP4 阶段训练。

**Status:** in-progress

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 先在真实模型短序列上联合验证 SelectiveAC、FSDP2 与 TP4 的数值和恢复，再以 DP shard2 × TP4、CP1 运行真实 draft 尺寸和 128K 输入；记录实际八卡参与、5 个 draft layers、24 个 Q heads、4 个 KV heads 及其余配置。
- [ ] 通过 DeepSpec 阶段入口完成完整 update、DCP 提交、全部训练 worker 退出、下一阶段新进程恢复及继续 update；真实 vLLM 交接沿用票 06 的生产环境和配置。
- [ ] 验证多阶段样本/RNG/优化状态连续、完整资源交接、保存失败保护、最近两份保留及按需 HF 导出；复用已有未受影响证据，补测本票组合与规模风险。
- [ ] 分别测量 draft 训练、保存、退出卸载、启动恢复及其总 wall time，包含 feature 读取/重分发、通信组重建、初始化和实际编译/HF 导出；排除 target features 生产及等待。
- [ ] 计时覆盖异步 GPU 工作完成并采用多卡关键路径，不把并行 rank 耗时相加；同时记录稳态吞吐、首次/后续阶段成本、峰值/释放后显存及测量波动。
- [ ] 固定工作量与逻辑训练配方比较旧常驻参考和满足退出恢复契约的基础实现；不以 tiny、skip、降低上下文或改变 DSpark 算法代替真实规模通过，也不预设加速比。
- [ ] 归档配置、运行方式、版本、容差依据及通过证据；这是规格要求的首阶段验收门槛，后续 dense 能力交付由此开始。

覆盖母规格 User Stories：11、12、39、40、43、44、45、47。


执行进度（2026-09-14 23:10）：首个 target 分区的20个真实131072-token样本已完成并校验，共161103279860bytes；两个TP4副本均退出。之后另一工作目录的ms-swift训练占用了全部八卡，draft尚未启动（0updates，无DCP）。仅停止本任务编排父进程，保留plan/producer/features；等待独占八卡窗口或替代机器。证据：`128k-first-target-acceptance.json`、`128k-resource-conflict.json`。该目标仍未通过128K训练验收。
