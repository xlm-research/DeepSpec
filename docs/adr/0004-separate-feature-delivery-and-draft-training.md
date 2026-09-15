---
status: accepted
date: 2026-09-14
---

# DeepSpec 编排 vLLM 推理与 TorchTitan draft 训练

用户已确认：DeepSpec 调度 vLLM 与 TorchTitan，决定训练多少批数据，并控制两端是否卸载资源；vLLM 只负责 target model 推理，TorchTitan 只负责 draft model 训练。Draft 的模型、loss、训练循环、优化器、并行、checkpoint 及训练配置归属 `torchtitan/`，训练组件尽可能直接使用 TorchTitan 已实现的配置和实现。本次先完成 Qwen3.8 DSpark，随后扩展，同一批 GPU 继续分阶段交替使用。

此决定替代 [ADR-0002](0002-retain-dspark-loop-for-torchtitan-components.md) 中 DeepSpec 持有 draft 训练循环和训练状态管理的约定。DeepSpec 拥有两端的外层调度和资源卸载控制权，各端负责执行本端操作并报告完成；TorchTitan 负责 draft 内部训练执行、保存恢复和资源释放。

## 已确认的职责

| 组件 | 职责 |
| --- | --- |
| DeepSpec | 调度 vLLM 和 TorchTitan，决定训练多少批数据、是否卸载两端资源，并衔接 target 特征的生产和消费。 |
| vLLM | 执行 target model 推理，提供目标模型特征，按 DeepSpec 调度释放本端资源。 |
| TorchTitan | 执行 draft model 训练，拥有训练组件、配置、数值更新和训练状态的保存恢复，按 DeepSpec 调度释放本端资源。 |

调度中的“批”已确定指特征分区：DeepSpec 配置分区数量和阶段数据量，TorchTitan 配置 local/global batch size 与 GAS。一个特征分区可包含多个 optimizer update，分区边界对齐完整 update，并保持已确认的样本顺序及每次 update 的分组。这些 batch/GAS 名称表示训练语义，具体字段优先使用 TorchTitan 原生配置体系。

tokenizer、chat template、截断、loss mask 和特征层选择的规则与配置归属 TorchTitan。DeepSpec 调用训练侧的数据准备入口、按阶段组织特征请求，保证 vLLM 和 draft 使用完全一致的 token 序列；vLLM 推理资源参数单独配置。

## 配置与组件复用

训练侧以 TorchTitan 原生配置体系为基础。通用训练、并行、优化器、学习率调度、activation checkpoint、compile、checkpoint、metrics 等尽可能直接复用已有组件及其 Config，模型配方和训练组件参数归属 `torchtitan/`。DeepSpec 拥有编排与训练数据量的控制；两类配置通过阶段请求衔接，不在 DeepSpec 中继续维护同义的训练组件配置和默认值。

本地 [Trainer.Config](../../torchtitan/torchtitan/trainer.py) 已组合这些配置；现有 [Qwen3.8 config registry](../../torchtitan/torchtitan/models/qwen3_8/config_registry.py) 可作为配置组织方式的参考。其通用 Qwen 模型及默认训练目标不等同于 DSpark draft。DSpark 特征消费、anchor/label 对齐、CE/L1/confidence 和阶段恢复所需的适配归训练侧；具体扩展接口在后续设计中确定。

训练数学语义继续遵循 [ADR-0001](0001-preserve-dspark-training-semantics.md)。复用原生组件须保持现有更新与恢复契约，不能由原生默认值隐式改变 microbatch 加权、GAS 或 optimizer 精度。此次明确的是职责和复用原则，具体组件兼容性尚未验证。

## 运行方式

用户已确认继续使用同一批 GPU：vLLM 生产一批特征并释放其 GPU worker，然后 draft 消费这批特征进行训练；完成阶段后交回资源，进入下一次特征生产。生成和训练各自拥有独立入口和配置。

阶段交接继续遵守 [ADR-0003](0003-unload-draft-between-training-phases.md)：DeepSpec 控制阶段切换与卸载请求；TorchTitan 在完整 optimizer step 边界完成训练状态同步持久化，然后退出本轮训练进程。DeepSpec 确认所有训练 worker 退出并释放资源后调度下一阶段；下一 draft 阶段由 DeepSpec 重新启动 TorchTitan 并从 checkpoint 恢复。保存失败不进入下一目标特征生产阶段。vLLM 的资源释放同样由 DeepSpec 调度、本端执行。

## 当前事实

- [Qwen vLLM 配置](../../config/dspark/dspark_qwen3_8_27b_vllm.py) 在同一 `train` 对象中配置 draft 并行和 vLLM engine，根 [训练入口](../../train.py) 实例化 DeepSpec trainer。
- [vLLM worker 启动器](../../deepspec/trainer/qwen3_8_vllm.py) 使用独立 Python 子进程；[Qwen trainer](../../deepspec/trainer/qwen3_8_vllm_trainer.py) 仍负责请求分区、等待生产和推进训练。进程分开尚未解除训练调度依赖。
- 现有交接记录包括 token IDs、loss mask、选定层 hidden states、最终归一化 hidden states 和长度信息。只传一个没有输入与位置关联的 hidden tensor，不能表达现有训练输入契约。
- [DraftFeatureIndex](../../deepspec/data/draft_feature_reader.py) 同时记录源特征及 draft DP、GAS、microbatch/update 归属。后续需要区分可复用的特征描述和某次训练的消费计划。
- 当前 Qwen 阶段结束会保存 DCP 和删除特征缓存，但模型、optimizer 与训练进程跨阶段保留；真正的 draft 卸载/重建参考实现在 [GLM trainer](../../deepspec/trainer/dspark_trainer.py)，不能视为 Qwen 已具备的能力。
- 本地 [TorchTitan Trainer](../../torchtitan/torchtitan/trainer.py) 的 `close()` 关闭 dataloader、CUDA graphs、checkpointer 和 metrics，没有清除 model/optimizer/mesh 引用；[训练 CLI](../../torchtitan/torchtitan/train.py) 在退出时另行销毁 process group。因此原生 `close()` 不能直接承担完整 draft 阶段卸载契约。
- 本地 [DCP checkpointer](../../torchtitan/torchtitan/components/checkpointer/dcp.py) 仍有保存 rank-local training RNG 的 TODO；原生配置和组件复用仍需验证 DSpark 的完整恢复状态。这些结论来自只读源码调查，尚未测量两种进程生命周期方案的开销、显存释放或更新等价。

## 第一轮已确认

1. **Q1 已确认：训练所有权。** 完整 draft 训练职责归 TorchTitan，用户进一步强调尽可能使用其已有配置与组件。
2. **Q2 已确认：运行方式。** 保留同一批 GPU 分阶段交替，生成与训练使用独立入口和配置，沿用 ADR-0003 的持久化与恢复要求。
3. **Q3 已确认：本次覆盖范围。** 本次先 Qwen3.8 DSpark，随后扩展。

## 已确认的后续决定

Q4、Q7 已确认阶段调度和两端资源卸载控制归 DeepSpec；Q5、Q8 采用推荐方案；Q6 已确定本次保证新流程自身的完整断点恢复，旧 DeepSpec 完整训练状态转换不纳入本次范围。

4. **Q4 已确认：阶段编排归 DeepSpec。** 用户明确 DeepSpec 负责调度 vLLM 和 TorchTitan，并决定训练多少批数据。此前建议将阶段编排放到 TorchTitan 的方案不再采用；TorchTitan 拥有 draft 内部训练执行和状态管理。
5. **Q5 已确认：特征请求规则归 TorchTitan。** tokenizer/chat template、截断、loss mask 及特征层选择的规则和配置由训练侧定义。DeepSpec 调用其数据准备入口、按阶段组织请求，保证 vLLM 和 draft 使用完全一致的 token 序列；vLLM 推理资源参数单独配置。
6. **Q6 已确认：保证新流程完整恢复。** 新 TorchTitan 流程必须从本流程已提交的 checkpoint 恢复模型、optimizer、scheduler、RNG 和数据进度，覆盖阶段卸载后的继续训练及任务中断后的重启。旧 DeepSpec 任务的完整训练状态转换不纳入本次范围；仅加载旧权重属于初始化新任务，不作为完整恢复的验收结果。
7. **Q7 已确认：资源卸载由 DeepSpec 控制。** DeepSpec 决定 vLLM 和 TorchTitan 是否卸载资源，各端执行自身释放并报告完成。TorchTitan 执行 draft 保存与恢复，卸载遵守完整 update 和 checkpoint 提交要求。
8. **Q8 已确认：按特征分区调度。** DeepSpec 配置分区数量和阶段数据量，TorchTitan 配置 local/global batch size 与 GAS；一个特征分区可含多个 optimizer update，分区边界满足既定完整 update 要求。

### Q6 确定的恢复范围

| 操作 | 含义 | 当前范围 |
| --- | --- | --- |
| 新流程断点恢复 | 从新 TorchTitan 流程已提交的 checkpoint 恢复完整训练状态，延续其训练进度。 | 已确定必须支持，包括阶段卸载恢复和从已提交 checkpoint 重启。 |
| 旧任务迁移续训 | 把旧 DeepSpec checkpoint 的模型、优化状态、调度器、RNG 和数据进度映射到新流程，延续旧任务。 | 不纳入本次范围。 |
| 旧权重初始化 | 使用旧模型权重建立新的训练任务，优化状态及训练进度按新任务初始化。 | 与完整续训含义不同，不能代替前两种恢复要求。 |

新流程的完整恢复仍需保存 DSpark 所需的 FP32 优化状态、各 rank RNG、样本消费位置与特征分区进度，并与 DeepSpec 的阶段调度状态对应。默认 TorchTitan `train_state` 只有 step 和 token 计数，原生 DCP 的 rank-local RNG 保存仍有 TODO，因此需要使用其状态扩展接口补齐并验证。恢复以连续训练与中断恢复后的后续更新对照验收，同一任务保持既定 draft 拓扑，跨拓扑恢复仍另行设计。

## Q9 已确认：保存后退出，下阶段启动恢复

用户已确认：DeepSpec 要求卸载 draft 时，TorchTitan 完成 checkpoint 保存后退出本轮训练进程，下阶段由 DeepSpec 重新启动并恢复。

该方案复用 TorchTitan 的启动流程，通过进程退出释放其持有的 GPU 资源。每阶段的进程启动、通信组重建和可能的编译开销必须计入既定 draft 性能口径，尚未实测收益。常驻方案需要额外实现并验证完整 GPU 状态销毁和重建，TorchTitan 原生 `close()` 不提供该契约，因此第一版选择进程退出方案。DeepSpec 决定何时卸载，TorchTitan 执行本端保存、退出和恢复。

## 当前决策树

已确定的前提是 DeepSpec 编排两端并控制卸载、TorchTitan 拥有 draft 训练、同一批 GPU 分阶段交替，本次先 Qwen3.8 DSpark。

- **调度数据量：Q8 已确定。** 下一步明确阶段请求、完整 update 对齐及总训练进度与学习率调度的衔接。
- **输入与特征需求：Q5 已确定。** 下一步明确特征记录和消费计划的边界、校验与缓存生命周期。
- **恢复范围：Q6 已确定。** 本次保证新流程自身完整恢复，不要求旧 DeepSpec 训练状态转换；细化新流程的恢复元数据和验收。
- **训练进程生命周期：Q9 已确定。** 按 DeepSpec 调度保存后退出，下阶段重新启动恢复；启动与重建开销计入 draft 性能。

Q1–Q9 的架构选择均已明确。配置归属、阶段请求、特征契约和恢复接口的整合方案见 [当前完整设计](../../doc/deepspec_orchestration_torchtitan_design.md)，该方案等待用户整体核对。尚未实施代码迁移或执行训练。
