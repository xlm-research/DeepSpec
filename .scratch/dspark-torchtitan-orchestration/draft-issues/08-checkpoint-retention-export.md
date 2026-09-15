# 08: 保留最近两份恢复点并按需导出 HF

**What to build:** 训练操作者能滚动保留最近两份完整阶段 checkpoint，并为评估或交付按需获取 HF 权重，保留的里程碑不受清理影响。

**Blocked by:** 04：提交完整 DCP，退出并跨进程续训。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 连续生成至少三份完整阶段 DCP，只有新提交成功后才清理过期普通阶段恢复点，始终保有最新两份；显式里程碑和最终产物单独保留。
- [ ] 通过真实训练与新进程恢复验证两份保留点各自包含完整 frozen/model、master/Adam、scheduler、RNG 和数据进度，不依赖旧进程或外置 frozen 权重。
- [ ] 普通阶段不强制导出 HF/safetensors；评估或最终交付请求能生成被现有消费入口加载的权重，参数与对应 DCP 一致。
- [ ] 真实保存/提交失败不删除上一有效 checkpoint，也不把未提交目录计入保留数量；清理不得改变恢复选择或分区进度。
- [ ] 实际 HF 导出耗时及频率计入保存成本，记录保留策略和可复现的导出/恢复方式。

覆盖母规格 User Stories：20、21、22、23。

