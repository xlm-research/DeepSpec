# 07: 中断后核对提交与编排进度

**What to build:** 训练或进程失败后，DeepSpec 能从有效提交恢复，识别已经完成的更新，避免消费不完整特征或错误推进阶段。

**Blocked by:** 06：打通真实 vLLM 特征生产与两阶段训练。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 通过真实阶段入口覆盖训练异常、worker 非正常退出、DCP 写入/提交失败和遗留 worker；故障通过实际子进程或文件提交边界注入，不以内部 helper 调用断言代替可观察结果。
- [ ] 覆盖首份 checkpoint 前失败、已有提交后失败和提交成功但 DeepSpec 尚未记录完成就中断；重启选择最近有效提交，已提交 updates 不重复，未提交工作按恢复游标重跑。
- [ ] 恢复前核对训练配方、target/特征身份、计划、分区、拓扑与 checkpoint；不兼容或不完整状态给出具体错误，不静默选择无关恢复点。
- [ ] 缺 shard、错误 tokens/样本、层序/dtype/shape/最终层语义不符及部分 feature 写入不能标记为就绪或被训练消费；失败不得删除仍需恢复的分区和上一有效提交。
- [ ] 验证重启后样本/RNG、loss、完整优化状态与连续训练参考一致，scheduler 不重新 warmup；保存失败和未完成资源释放都阻止下一 target 阶段。
- [ ] 协调停止当前任务后可由新进程重新启动；不要求原地自动重试、不保存半步梯度、不引入跨拓扑恢复。

覆盖母规格 User Stories：24、25、30、50、60、62、63、64。

