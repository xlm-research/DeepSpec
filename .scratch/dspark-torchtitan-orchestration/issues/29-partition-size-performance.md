# 29: 按完整 update 调整分区大小并比较成本

**What to build:** 操作者能调整 DeepSpec 特征分区数量或大小，在保持原有更新序列的同时比较阶段交接次数带来的成本。

**Blocked by:** 10：完成八卡真实 Qwen 128K 首验。

**Status:** ready-for-agent

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 仅改变完整 optimizer updates 如何划入特征分区；保持样本顺序、epoch 洗牌/截断、logical microbatch、GAS、每次 update 分组、RNG 与全程 scheduler。
- [ ] 通过训练侧准备入口校验合法边界，无法满足完整 update 的配置给出明确错误；不能提前更新、补样本或额外丢样本以满足分区大小。
- [ ] 使用固定工作量和现有 target 配置，比较基线与少量合法分区方案的实际消费顺序、完整更新及跨阶段恢复结果。
- [ ] 每种方案仍在每阶段同步提交完整 DCP 后退出，确认资源释放再交接，保持最近两份保留、按需 HF 与失败恢复/缓存回收契约。
- [ ] 报告训练/保存/退出/启动恢复分项和总 wall time、阶段次数、CPU/GPU 内存与缓存/I/O 成本，排除 target 生产及等待，只采用符合资源约束且有稳定收益的方案。

覆盖母规格 User Stories：15、16、43、44、46、52。

