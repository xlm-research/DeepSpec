# 06: 打通真实 vLLM 特征生产与两阶段训练

**What to build:** DeepSpec 能按分区调用训练侧数据准备，使用既有 vLLM 生产特征并交给 TorchTitan，在同一批 GPU 上连续完成两轮生产、训练和保存退出。

**Blocked by:** 04：提交完整 DCP，退出并跨进程续训。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] TorchTitan 定义 tokenizer/chat template、截断、loss mask、特征层需求及合法 update 分组；DeepSpec 调用无需构建 GPU draft 模型的数据准备入口，持久化分区计划并组织 target 请求。
- [ ] 以小规模真实 Qwen 工作负载跑通至少两个分区，验证相同 tokens/mask 进入 vLLM 与 draft；target 使用现有解释器、源码编译产物、模型与推理资源配置，独立于 draft batch/GAS 和并行布局。
- [ ] 特征产物记录 producer 的样本身份、顺序、层与最终归一化 hidden 语义、长度及原始 shards；draft 消费计划单独记录 microbatch/update、读取布局和游标，不把消费拓扑写成 producer 事实。
- [ ] reader 校验样本/tokens、层序、dtype/shape、mask/长度和最终层语义；固定 fixture 覆盖完整 producer features 与已有 shards 重组，保留 token 顺序和监督内容。
- [ ] 完整 target 产物就绪、vLLM worker 退出且资源释放后才启动 TorchTitan；draft 提交且所有 worker 退出后才生产下一分区，交接通过实际进程与 GPU 资源观察验收。
- [ ] 保持 epoch 洗牌、完整 global-batch 截断及每次 update 的 microbatch 分组；分区大小/数量由 DeepSpec 调度，local/global batch 与 GAS 来源于训练配方。
- [ ] 仅在所有相关消费者完成且阶段 checkpoint 已提交后回收 feature 缓存；训练结果和下一消费位置与任务计划一致，并保留 fixed-feature 数值对照及真实生产交接各自的证据。

覆盖母规格 User Stories：15、16、27、28、29、30、52、53、54、55、56、57、59、61、64。

