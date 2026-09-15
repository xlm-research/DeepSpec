# DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复

日期：2026-09-14。

Status: design-review

2026-09-14 修订：按本次 `/to-spec` 请求，将 [ADR-0004](../../docs/adr/0004-separate-feature-delivery-and-draft-training.md) 已确认的 Q1–Q9 与 [阶段接口设计](../../doc/deepspec_orchestration_torchtitan_design.md) 整合为实施规格。DeepSpec 编排两端，TorchTitan 拥有 draft 训练，每阶段保存后退出、下阶段启动恢复。主要测试边界待用户确认，确认后将本条规格标记为 `ready-for-agent`。

发布位置：用户选定的本地 Markdown issue tracker。

已确认边界：DeepSpec 按特征分区调度并控制两端资源卸载；TorchTitan 定义数据预处理与特征需求的规则和配置，DeepSpec 调用其数据准备入口。本次保证新流程完整断点恢复，旧 DeepSpec 完整训练状态转换不纳入本次范围；Q9 已确认保存后退出、下阶段启动恢复。

## Problem Statement

训练开发者希望将 draft 模型、训练执行及训练配置归属 TorchTitan，尽可能直接复用已有组件，同时让 DeepSpec 专注于 target/draft 编排、特征分区和资源交接。现有训练循环混合了特征生产、DSpark 监督消费、加权 loss、梯度累积、FP32 master optimizer 和阶段交接，需要拆开职责并保持训练语义。

当前 DSpark 对每个 microbatch 分别计算跨独立监督分片的加权 loss 均值，再对梯度累积窗口中的 microbatch 等权平均。参考 TorchTitan 路径则采用整步有效 token 分母，并可能使用不同的 FSDP 梯度归约约定。直接搬用默认 loss、GAS 或梯度缩放会改变训练目标。

当前 Qwen draft 在 target 特征生成阶段仍保留 GPU 训练状态。历史 exact 分区曾允许截断梯度累积窗口，当前 Qwen 阶段路径已要求完整 update 边界并能保存 DCP，但没有完整阶段退出、重新启动恢复的编排协议。Target 特征生产的 rank 布局、请求分组与 draft 配置仍存在耦合，改变 draft 并行方式可能影响 target。

用户要求所有优化限于 draft model。Target model 继续通过现有源码编译的 vLLM 和现有运行环境提供监督，不能通过修改 target、替换环境或改变 DSpark 训练配方取得性能收益。需要衡量的性能是 draft 训练、保存、卸载和恢复的累计耗时，排除 vLLM 推理生成 target features 的时间。

## Solution

DeepSpec 决定特征分区数量和阶段数据量，调度 vLLM 与 TorchTitan 并控制两端资源卸载。TorchTitan 拥有 DSpark 模型、预处理、特征消费、训练循环、optimizer update 和 checkpoint 实现，训练配置及通用组件尽可能直接使用原生能力。首阶段以单机八卡 Qwen3.8 DSpark dense draft 接入 SelectiveAC，实现完整 update 分区、同步 DCP 提交后训练进程退出，以及下阶段重新启动后的同拓扑恢复；先完成小规模数值对照，再验收真实 draft 尺寸和 128K 输入。

固定既有 target 特征生产方式，通过 draft 专用索引和 reader 将生产布局与消费布局解耦。Draft 可以选择自己的并行布局，保持输入特征、样本顺序、逻辑 microbatch 和 DSpark 更新语义。

每个 draft 训练阶段完成完整 optimizer update 后，同步提交完整训练 checkpoint，然后按 DeepSpec 的阶段请求退出全部训练 worker。DeepSpec 确认提交及资源释放后推进下一阶段，再次启动 TorchTitan 恢复模型、optimizer、scheduler、RNG 和数据进度。阶段停点与全程训练进度分别表达，scheduler 不因新阶段重新 warmup。阶段保存使用完整 DCP，HF 权重按评估或最终交付需要导出，滚动保留最近两份阶段 checkpoint。

后续按明确的合法组合覆盖 DP replicate、FSDP2、HSDP、TP、SP、loss parallel、CP、PP，以及 GLM-5.3-Flash DSpark 的 MoE EP。以真实更新等价和阶段恢复正确性为基础，测量并优化 draft 全流程性能；不预先承诺未经测量的加速比。

## User Stories

1. As a draft training developer, I want TorchTitan to own draft training and its native component configs while DeepSpec orchestrates phases, so that the two runtimes have clear responsibilities and DSpark semantics remain intact.
2. As a draft training developer, I want TorchTitan integration to affect only the draft model, so that target inference remains stable.
3. As a training operator, I want to keep my existing environment and source-built vLLM, so that working target inference does not require rebuilding its software stack.
4. As a training researcher, I want each logical microbatch to retain its own globally normalized loss, so that integration preserves the DSpark objective.
5. As a training researcher, I want gradient accumulation to retain equal weighting across microbatches, so that batches with different valid-token counts keep their original influence.
6. As a training researcher, I want CE, distribution L1, and confidence BCE to retain their definitions, so that supervision remains comparable with existing experiments.
7. As a training researcher, I want masks, position decay, anchors, and detached confidence targets to remain unchanged, so that infrastructure changes do not alter the learning algorithm.
8. As a training developer, I want gradient scaling to account for the actual parallel reduction rules, so that dense and expert parameters receive the intended updates.
9. As a training researcher, I want frozen parameters and FP32 master optimizer state to retain their current behavior, so that precision and optimization remain consistent.
10. As a training developer, I want TorchTitan SelectiveAC to work with the selected draft FSDP2 configuration, so that component reuse preserves DSpark updates while exposing memory and computation tradeoffs.
11. As a training operator, I want the first integration validated on one machine with eight GPUs, so that the initial deployment matches my available validation setup.
12. As a training operator, I want real Qwen3.8 draft dimensions and 128K inputs included in first-stage acceptance, so that tiny-model success does not hide production-scale failures.
13. As a training operator, I want the complete draft GPU training state unloaded after each draft phase, so that the next target phase can use the released resources.
14. As a training researcher, I want phase boundaries aligned with complete optimizer updates, so that unloading does not lose partially accumulated gradients.
15. As a training researcher, I want partition resizing to preserve sample order and update membership, so that lifecycle changes preserve the existing training schedule.
16. As a training researcher, I want the existing epoch shuffle and full-global-batch truncation rules preserved, so that boundary alignment does not silently add or discard training examples.
17. As a training operator, I want a complete durable checkpoint committed before each unload, so that recovery does not depend on the previous process remaining alive.
18. As a training researcher, I want model weights, Adam state, scheduler, RNG, and data progress restored together, so that the next update continues the same training trajectory within numerical tolerance.
19. As a training developer, I want rebuild operations and target generation isolated from restored draft RNG state, so that phase transitions do not change anchor or stochastic execution sequences.
20. As a training operator, I want stage checkpoints stored in DCP without a mandatory duplicate HF export, so that persistence avoids unnecessary full-weight gathering and serialization.
21. As a model consumer, I want HF weights exported when evaluation or final delivery requires them, so that removing duplicate stage exports preserves usable model artifacts.
22. As a training operator, I want the latest two complete stage checkpoints retained, so that storage remains bounded while a previous recovery point remains available.
23. As a training operator, I want milestones and final artifacts retained separately, so that rolling cleanup does not remove intentionally preserved results.
24. As a training operator, I want failed saves to stop the task before the next target phase, so that an incomplete checkpoint is never treated as a successful handoff.
25. As a training operator, I want restart to use the latest verified committed checkpoint, so that interrupted writes do not corrupt recovery.
26. As a training operator, I want each draft phase in one run to restore the same topology, so that initial recovery behavior has a clear and testable contract.
27. As a parallel training developer, I want target producer settings independent of draft topology, so that changing draft DP, TP, CP, or PP does not reconfigure vLLM.
28. As a parallel training developer, I want stable feature sample identities and ordering metadata, so that draft consumers can read existing feature files across different layouts.
29. As a parallel training developer, I want draft readers to reconstruct producer shards and distribute the required consumer views, so that the same supervision supports multiple draft parallel configurations.
30. As a training operator, I want feature cleanup to wait for every relevant draft consumer, so that one rank cannot delete data still needed by another.
31. As a parallel training developer, I want replicated DP, FSDP2, and HSDP supported, so that I can choose appropriate draft state replication and sharding.
32. As a parallel training developer, I want tensor and sequence parallelism supported for DSpark's context and query streams, so that draft layouts remain correct across attention and auxiliary heads.
33. As a parallel training developer, I want complete DSpark vocabulary-parallel losses, so that vocabulary sharding preserves every enabled supervision term while avoiding full-logit gathering.
34. As a parallel training developer, I want context parallelism to retain DSpark-specific attention semantics, so that long-context training works with the model's mixed inputs.
35. As a parallel training developer, I want TorchTitan to own draft pipeline execution while DeepSpec schedules complete phases, so that PP and lifecycle orchestration have distinct owners.
36. As a parallel training developer, I want pipeline stages to propagate differentiable teacher-derived context correctly, so that splitting decoder layers preserves the complete backward path.
37. As a MoE training developer, I want GLM-5.3-Flash DSpark used for initial expert-parallel acceptance, so that EP is tested on a real trainable expert model.
38. As a MoE training developer, I want routed experts, shared experts, router behavior, and expert gradient scaling preserved, so that dispatcher changes do not change the MoE training objective.
39. As a training operator, I want a documented matrix of accepted parallel combinations and restrictions, so that unsupported configurations are identifiable before a costly run.
40. As a training developer, I want every accepted parallel combination tested through checkpoint, unload, restore, and another update, so that forward/backward success is not mistaken for complete lifecycle support.
41. As a training researcher, I want fixed-feature numerical comparisons on real draft model classes, so that toy-model tests do not stand in for DSpark correctness.
42. As a training developer, I want tests to exercise the DeepSpec phase entry and TorchTitan training process through observable results, so that implementation refactoring does not require low-level mock expectations.
43. As a performance engineer, I want draft training, saving, unloading, and restoring measured separately and together, so that optimization targets the costs included in the user's performance objective.
44. As a performance engineer, I want target feature generation excluded from draft timings, so that vLLM variability does not obscure draft improvements.
45. As a performance engineer, I want first-use compilation and reconstruction costs included, so that warmed-up step throughput does not hide repeated phase overhead.
46. As a performance engineer, I want recomputation, prefetch, resharding, compilation, and checkpoint optimizations evaluated against fixed training semantics, so that faster runs remain meaningful comparisons.
47. As a training researcher, I want software versions, topology, workload, tolerances, and actual GPU participation recorded, so that correctness and performance claims are reproducible.
48. As a project maintainer, I want the specification to distinguish confirmed requirements from unverified implementation candidates, so that agents do not treat assumptions as completed support.
49. As a training operator, I want each draft phase to commit its state and exit all training workers under DeepSpec control, so that the next target phase receives the released GPU resources.
50. As a training operator, I want a restart to reconcile committed training progress with DeepSpec's phase record, so that an interruption after checkpoint commit does not repeat completed updates.
51. As a training developer, I want training component settings to come from TorchTitan's native configuration, so that duplicate defaults in the orchestration layer cannot silently change my recipe.
52. As a training operator, I want feature partition size and count configured separately from local/global batch size and GAS, so that phase scheduling preserves complete optimizer updates.
53. As a training researcher, I want TorchTitan to define tokenization, chat templates, truncation, supervision masks, and feature layers, so that target production and draft training use the same prepared inputs.
54. As a training operator, I want input preparation to run before constructing GPU draft state, so that preparation leaves the shared GPUs available for target inference.
55. As a training developer, I want feature artifacts to describe producer facts independently of the draft consumption plan, so that draft layouts can change between runs without changing the meaning of stored features.
56. As a training researcher, I want feature readers to validate sample identity, token order, layer order, lengths, masks, dtype, shape, and final-layer semantics, so that malformed or mismatched supervision cannot silently enter training.
57. As a training operator, I want each phase request to identify its resolved recipe, feature artifacts, whole-run plan, stopping point, and recovery checkpoint, so that the training process has an unambiguous continuation contract.
58. As a training researcher, I want the scheduler to follow whole-run progress across phase restarts, so that a new feature partition does not restart warmup or shorten the intended schedule.
59. As a training operator, I want a completed phase to report its committed checkpoint, consumed sample range, completed updates, and next input position, so that orchestration can verify progress before proceeding.
60. As a training operator, I want recovery to reject incompatible recipes, plans, topology, or feature identities, so that an unrelated checkpoint cannot silently resume the wrong task.
61. As a training operator, I want DeepSpec to confirm target worker exit before starting draft training on the same GPUs, so that both phases honor the resource handoff.
62. As a training operator, I want training failure or unexpected worker exit to stop phase advancement, so that a failed phase is not reported as completed work.
63. As a training operator, I want recovery without a valid new commit to replay only work after the last valid checkpoint, so that uncommitted work can be retried without losing durable progress.
64. As a training operator, I want incomplete target feature outputs excluded from ready partitions, so that restart cannot mistake a partial write for usable supervision.
65. As a project maintainer, I want existing baseline and checkpoint evidence distinguished from acceptance of the new TorchTitan training process, so that earlier completed work does not imply that the new architecture has been validated.

## Implementation Decisions

1. **训练循环与模块归属。** DeepSpec 拥有阶段调度、特征分区数量/数据量和资源卸载控制；TorchTitan 拥有 batch 消费、forward/backward、GAS、裁剪、optimizer/scheduler、checkpoint 与恢复实现。Draft 模型主体、attention/decoder block、配置及模型专属并行适配组成 TorchTitan 下的独立 DSpark draft 模型模块；训练适配及数据准备同属训练侧。DeepSpec 引用原生训练配方并启动独立训练进程，不再用旧 trainer 执行训练计算。每个模型、mesh、同步操作及状态对象具有明确的管理方。

2. **只优化 draft。** TorchTitan 组件、并行策略、SAC、compile、optimizer/checkpoint 和生命周期优化仅作用于 draft。Target model 的模型实现、推理内核、精度、并行配置、生成策略和 vLLM worker 保持现有行为。共享配置与模型并行模块通过 draft 专用入口或明确角色分支生效，不能依靠同步修改 target 设置绕过 draft 校验。

3. **保留现有环境。** 继续使用现有环境及源码编译的 vLLM，保留 target 解释器、源码位置、编译产物和依赖组合。显式固定 target 的启动解释器与源码解析位置，避免它默认跟随 draft 设置。TorchTitan 参考基线为完整 commit f6b9152e9bedcc18f5dc339b9f88265e5a07e988；兼容性核对以当前实际环境为准，不默认重建环境或替换 PyTorch/CUDA/vLLM。发现必须改变基线的具体冲突时回到设计讨论。

4. **DSpark loss 与梯度归一化。** 每个逻辑 microbatch 先按独立监督分片汇总有效位置权重，得到该 microbatch 的全局加权均值，再按 GAS 对各 microbatch 等权平均。保留 CE、概率分布 L1、confidence BCE、系数、mask、位置衰减、epsilon、零分母分支和 confidence target detach。TP 副本和 PP 阶段不能增加样本分母。归约补偿须匹配实际选用的 FSDP 梯度平均或求和约定，不能照搬旧倍率；每个逻辑 microbatch 分母、GAS 平均和并行归约补偿在完整更新中恰好应用一次。专家参数按实际 dispatcher 和 sparse mesh 推导。

5. **更新与精度契约。** 保留 anchor 采样、feature/label 对齐、冻结参数集合、参数精度、通信精度、FP32 master weights、Adam moments/计数、梯度裁剪及学习率调度语义。冻结 LM head 仍须向可训练 draft hidden 传回梯度。改变物理执行切块时仍需保持逻辑 microbatch 的分母、样本归属与 RNG；保持 global batch 相同并不自动使改变 local batch/GAS 等价。

6. **首阶段 SelectiveAC。** 通过 TorchTitan 原生 AC 配置接入 SelectiveAC，与所选 draft FSDP2 路径组合验证。首次正确性对照关闭外层 model compile，并保持重计算所需 RNG 行为。每个 block 使用一套 checkpoint 策略，避免叠加 HF checkpoint 与 TorchTitan 包装。后续将 AC 保存策略和 compile 作为性能候选，按实际兼容性与收益决定配置。当前 native ring CP 与外层 model compile 的限制不能被视为已经消除。

7. **特征生产与消费解耦。** TorchTitan 定义 tokenizer/chat template、截断、loss mask 和特征层需求，DeepSpec 调用其数据准备入口并按特征分区组织请求。编排层固定既有 producer 模型身份、推理参数、owner/device mapping 和输出 CP 布局，不从 draft DP/TP/CP 推导。特征产物记录样本身份、原始顺序、teacher 语义和文件/shard 布局；TorchTitan 消费计划与 reader 记录本次训练的 microbatch/update 归属、读取布局和恢复游标。原始 feature 内容和 vLLM 推理行为保持既定语义。

8. **输入重分发与所有权。** Producer 为 CP1 时，TorchTitan draft reader 读取完整 features 后构造所需布局；producer 已分片时，收齐同一样本 shards 并恢复 token 顺序后再分发。TP peers、CP 分片和 PP 阶段各自收到模型需要的视图，监督内容不变。索引、checkpoint 游标与下一逻辑 microbatch 必须一致。DeepSpec 仅在所有相关消费者完成且阶段 checkpoint 已提交后回收缓存，不能以 draft rank 推测 producer 文件所有者。读取、索引和重分发开销计入 draft。

9. **optimizer-step 对齐的阶段。** 缓存分区允许调整大小和数量，但保持样本顺序、GAS 与每次 update 的 microbatch 分组。正常卸载发生在完整 optimizer update 及对应 scheduler/进度更新之后。保留既有每 epoch 洗牌后按完整 global batch 截断的规则，以及完整 update 的停止边界；不为提前卸载缩短 GAS、提前更新、补样本或额外丢样本。

10. **完整 GPU 状态卸载。** 按 DeepSpec 阶段请求，先完成阶段内更新、结束在途计算和通信，PP 路径排空流水线，再同步提交完整 checkpoint，随后退出所有 TorchTitan 训练 worker。DeepSpec 确认 worker 退出和相关 GPU 资源释放后才能启动下一 target 阶段。释放范围包含模型、梯度、FP32 优化状态、FSDP/专家通信缓冲、预取 batch 及其他 draft GPU 引用。正常阶段切换无须序列化半步梯度，第一版不采用训练进程常驻的卸载方案。

11. **完整恢复。** 同一训练任务各 draft 阶段保持相同并行拓扑。下一阶段由 DeepSpec 新启动 TorchTitan 训练进程，恢复模型、FP32 优化状态、Adam moments/计数、scheduler、各 rank RNG、样本索引和数据进度。训练 RNG 在构建与加载完成后、下一次训练计算前恢复。使用原生 DCP 的状态扩展入口补齐 DSpark 所需状态，不能仅依赖默认 step/token 计数；本次保证新流程自身完整恢复，不要求转换旧 DeepSpec 完整训练状态。

12. **持久化与导出。** 每阶段同步提交完整 draft DCP 和模型/训练配置、身份、进度等必要元数据；每份阶段 checkpoint 包含 frozen 权重，可独立恢复。HF/safetensors 在评估或最终交付需要时导出。状态完整性校验须适配无需 HF 权重的阶段格式，不能依靠某个 HF 文件存在来证明 DCP 完整。提交成功后才发布恢复入口和推进阶段交接。存储根目录沿用现有训练配置。

13. **保留与失败语义。** 新 checkpoint 成功提交后滚动保留最近两份完整阶段 checkpoint，显式里程碑和最终产物单独保留。保存失败时协调停止任务，不进入下一 target 阶段；上一成功提交仍可恢复。重启使用最近成功提交且通过验证的 checkpoint，忽略未提交目录。首次提交前失败从初始状态重跑。首版不要求保存失败后的原地自动重试。

14. **并行能力与组合。** 总体范围覆盖 DP replicate、FSDP2、HSDP、TP、SP、完整 DSpark loss parallel、CP、PP 和 MoE EP，按明确的合法组合验收。HSDP 是 DP replicate 与 shard 的组合；SP 和 loss parallel 是 TP 相关布局/算法策略；EP 与专家 FSDP 使用同一 dense rank 域的 sparse 视图，不能把 EP 再乘入 world size。后续接入以 spmd_types 为主要后端，PP 首先使用 1F1B；其他后端、schedule、async TP 和 dispatcher 不隐含要求任意交叉。

15. **DSpark 并行适配要求。** TP/SP 必须覆盖 teacher context 与 draft query 的双输入布局、残差、norm、Markov/confidence 和输出 head。Loss parallel 必须保留全词表归一化下的 CE、L1、acceptance/confidence 及相关 head 语义；先 gather logits 的过渡实现不构成完整 loss-parallel 验收。CP 允许保留 DSpark 专用 attention 通信实现。TorchTitan 拥有训练循环和 pipeline 执行，DeepSpec 调度完整特征分区；PP stage 设计须传播可微 teacher-derived context、对应梯度及必要监督输入，不能只自动切 decoder layers。

16. **模型与交付顺序。** 首先完成单机八卡 Qwen3.8 dense draft、SelectiveAC、完整卸载恢复和真实尺寸 128K 验收；初始候选拓扑为 DP shard2 × TP4、CP1。随后覆盖 dense 并行组合，最后以 GLM-5.3-Flash DSpark 验证 MoE EP，先使用真实模型类的小规模配置和 native dispatcher。首个 EP 候选为 DP shard8、EP8、其余并行轴为 1。真实 Qwen 有 5 个 draft layers、24 个 Q heads 和 4 个 KV heads，当前 GQA 分片规则下 TP 候选为 1/2/4，不能承诺 TP8 或自动 PP8。

17. **性能目标与优化纪律。** 性能主指标为固定工作量下各阶段 draft 训练、保存、卸载和恢复 wall time 的总和，排除 target 特征生成和等待该生产阶段完成的时间。训练包括已就绪 features 的读取/搬运、真实模型与 loss、backward、同步、裁剪和 optimizer/scheduler；保存包括 DCP 写入/提交及实际发生的 HF 导出；恢复包括重建、并行化、optimizer 构建和状态加载。初始化与编译开销按实际位置归入训练或恢复并单列，各计时互不重复且覆盖对应 GPU 异步工作的完成。

18. **性能候选而非预设收益。** 保留已有阶段 DCP 无需重复 HF 导出的行为，迁移后的训练侧同样只在需要时导出 HF；进一步评估被 DCP 覆盖的初始化、CPU 配置复用、完整 vocab-parallel loss、SAC 策略、FSDP child reshard/预取、compile 和缓存命中，以及不改变 update 分组的分区大小调整。所有候选都需记录数值正确性、draft 分项耗时、总耗时和资源开销。每阶段持久化成功后卸载的承诺保持不变；异步 API 后立即等待完成不能被直接记作隐藏了全部保存时间。

19. **全程进度与阶段停点。** DeepSpec 通过训练侧数据准备获得符合 batch/GAS 规则的分区计划并持久化，阶段请求引用计划、特征产物和恢复 checkpoint。TorchTitan 按全程总训练量和已完成进度推进 scheduler，当前分区停点仅控制本阶段退出，不能重置 warmup 或把本阶段步数当成全程 scheduler 长度。

20. **提交与编排进度一致性。** 已提交 checkpoint 记录消费分区、样本位置和 update 进度；DeepSpec 恢复时据此核对阶段记录。若 checkpoint 已提交而编排记录尚未更新就中断，应识别已完成更新；若没有有效新提交，则从上一有效提交恢复并重跑未提交工作。正常推进还必须确认所有 worker 已退出并释放资源。

21. **配置来源与无 GPU 数据准备。** TorchTitan 的 DSpark 训练适配以原生 Trainer Config 和配置构建入口为基础，尽可能直接复用训练、并行、optimizer、scheduler、AC/compile、checkpointer 和 metrics 组件的配置。DeepSpec 持有编排配置、训练配方引用及独立 vLLM 配置，不重复维护训练组件的同义默认值。训练侧数据准备入口只解析配方并准备确定的输入与特征需求，无须构建 GPU draft 模型；DeepSpec 在其返回后组织 target 请求。

22. **阶段交换契约。** 以下内容明确交换双方的责任与可观察结果；具体类型、序列化字段和 CLI 参数名由实施确定。

| 交接 | 必须表达的内容 | 校验与完成条件 |
| --- | --- | --- |
| TorchTitan 数据准备 → DeepSpec | 有序样本身份、确定的 token 序列、监督 mask、特征层需求、合法 microbatch/update 分组 | 依据训练配方生成，保持既定样本顺序、截断及完整 update 规则 |
| DeepSpec → vLLM | 本分区输入、target 身份与特征需求、独立推理配置、产物位置 | 沿用已准备的 token 序列；vLLM 不接管 GAS、loss 或 optimizer 配置 |
| 特征产物 → TorchTitan | 输入对应关系、选定层 hidden states、所需最终归一化 hidden states、长度和 mask、teacher 语义、原始文件与 shard 布局 | 校验样本及 tokens、层顺序、dtype/shape、长度和最终层语义；不完整输出不能标记为就绪 |
| DeepSpec → TorchTitan | 已解析训练配方引用、本分区特征清单、完整 update 停点、全程计划身份、最近有效 checkpoint 或首阶段初始状态 | 恢复时核对配方、计划、拓扑、checkpoint 与分区的对应关系；不兼容时停止并报告原因 |
| TorchTitan → DeepSpec | 有效 checkpoint 引用、已消费分区及样本范围、完成 update 和下一消费位置 | 提交记录与计划一致，全部训练 worker 退出并释放资源后才推进阶段 |

23. **两端资源与缓存交接。** 同一批 GPU 依次用于目标特征生产阶段与 draft 训练阶段。DeepSpec 确认 target 产物完整，并按既定调度确认 vLLM worker 退出和资源释放后，再启动 TorchTitan。Draft 阶段正常完成、提交并退出后才能开始下一目标特征生产阶段；训练异常、保存失败或遗留 worker 均不能按正常完成推进。恢复时依据任务计划与特征身份处理未完成分区，未提交的工作允许重跑；缓存删除仍受全部消费者完成和 checkpoint 成功提交约束。

## Testing Decisions

1. **主要测试边界（待确认）。** 采用一个主要集成边界：DeepSpec 阶段入口 → 独立 TorchTitan draft 训练进程。输入固定、已就绪的 target feature fixture 和训练配置，执行真实 draft 模型、DSpark loss、optimizer、DCP 保存、进程退出及新进程恢复，观察阶段结果及下一次 update。优先延续现有阶段入口和 fixture；迁移确实需要新接口时，只在编排与训练进程交接处提供统一阶段请求与结果，不为每个 TorchTitan wrapper 或内部 helper 建立独立测试入口。覆盖模块为 DeepSpec 阶段编排，以及 TorchTitan 的 DSpark 数据准备/reader、真实模型与训练适配、checkpoint 状态扩展。

2. **好的测试验证外部行为。** 主要断言输入样本与特征对应关系、各 loss、参数更新、完整 checkpoint 的可恢复性、GPU 资源释放、失败后的可恢复入口，以及性能统计范围。不以特定包装类名、调用次数、私有 buffer 名称或是否调用某个清理函数代表正确。目标特征生产可由固定 fixture 替代，draft 数学计算、优化器和分布式保存/恢复不可用 mock 替代其数值证据。

3. **数值基线。** 固化实施前当前工作树的可复现基线，记录初始权重、features、样本、anchors/RNG、GAS、冻结集合、精度和拓扑。先以真实 Qwen3.8 DSpark 模型类的小规模配置进行 FP32 对照，再进行 BF16。比较输出、CE/L1/confidence 项、全部可训练参数梯度、clip norm、FP32 master weights、Adam moments/计数、scheduler 和连续更新；包含至少两个 optimizer updates。

4. **不等有效分母与累积。** 构造同一累积窗口中各 microbatch 有效权重和不同的输入，证明保留 microbatch 等权平均。包含有效 mask、位置衰减、边界与零分母分支。至少一组 FSDP 测试使用 GAS 不小于 2，验证累积、同步和重计算组合；不能只用等长无 mask 数据使两套归一化偶然相同。

5. **连续训练与恢复对照。** 对同一固定拓扑和输入，分别运行连续训练，以及在完整 update 后保存、销毁全部 draft GPU 状态、重建恢复后继续训练。下一阶段样本、RNG、loss、梯度及 optimizer 更新应在预先确定的容差内一致。覆盖新进程从 DCP 恢复，防止测试意外依赖仍存活的 CPU/GPU 对象。改变初始化的随机消耗不能改变恢复后的训练序列。

6. **分区与缓存契约。** 对齐前后保持实际训练样本顺序和每次 update 分组；保留原有 epoch 截断与停止规则。同一 producer cache 经不同 draft reader 布局后，tokens、loss mask、hidden features、样本顺序和恢复游标一致。覆盖 producer CP1 及分片特征重组，不需要重新优化或修改 target 推理。所有消费者完成前不得删除相应缓存。

7. **角色隔离。** 改变 draft 拓扑或仅启用 draft AC/compile，不应改变固定 target 配置、vLLM interpreter/source、owner/device mapping 或监督输出契约，也不应因为无关变换字段差异被误判成生产布局不兼容。集成验证只确认既有 target 交接可用，不能以 target 加速作为 draft 优化的测试成果。

8. **持久化故障与保留。** 在真实文件保存/提交边界注入失败，验证未完成目录不会成为恢复入口、上一提交保持可用、任务不会进入下一 target 阶段。覆盖首份 checkpoint 前失败、已有 checkpoint 后失败，以及新提交后的滚动保留。DCP-only 阶段 checkpoint 必须可以恢复完整模型与 optimizer；HF 按需导出和里程碑保留单独验证。

9. **完整卸载。** 连续运行多个 draft 阶段，验证完整提交后所有训练 worker 退出及相关 GPU 资源释放，以及下一阶段由新进程恢复。覆盖遗留 worker、FSDP/专家通信、预取输入和异步工作；其他进程的显存不计为 draft 泄漏。每阶段启动、通信组重建和实际编译开销计入性能验证。

10. **并行候选矩阵。** 下表是需要逐项验证的候选，不代表已实现、已证明兼容或全部能跑 128K。每行先用短序列固定 features 验证完整更新，再覆盖持久化、卸载、恢复和下一次更新。真实 128K 首验从 D 行开始。不同 DP degree 导出的 GAS 不同，跨行不能直接宣称同一更新目标或无条件速度优劣；每行使用匹配逻辑 microbatch 语义的参考。

| 候选 | DP replicate | DP shard | TP | CP | PP | 验收重点 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 8 | 1 | 1 | 1 | 1 | 复制 DP |
| B | 1 | 8 | 1 | 1 | 1 | FSDP2 |
| C | 2 | 4 | 1 | 1 | 1 | HSDP |
| D | 1 | 2 | 4 | 1 | 1 | 首阶段；随后分别验证 SP、loss parallel 及二者组合 |
| E | 1 | 4 | 1 | 2 | 1 | 专用 CP |
| F | 1 | 1 | 4 | 2 | 1 | TP 与 CP |
| G | 1 | 4 | 1 | 1 | 2 | PP 1F1B |
| H | 1 | 1 | 2 | 2 | 2 | TP/CP/PP，依赖双输入 stage/layout 设计 |

11. **MoE 独立验收。** GLM-5.3-Flash DSpark 先以小规模真实模型和 native dispatcher 验证 routed/shared experts、router、专家梯度归一化、optimizer state 与完整 DCP 恢复，再扩展规模和组合。288 个 routed experts 满足 EP8 的整除条件，但这不构成 Titan sparse-layout 已适配的证据。Qwen dense 的通过结果不能替代 EP 验收。

12. **性能比较。** 在固定工作量和逻辑训练配方下记录 draft 训练、保存、卸载、恢复的分项及总计，另报稳态吞吐、首次编译、后续缓存命中和峰值/卸载后显存。记录旧常驻路径参考和满足卸载契约的基础实现，分别说明生命周期成本与组件收益。计时使用多卡关键路径，不能把各 rank 同时进行的耗时简单相加。实际发生的按需 HF 导出计入保存并注明频率；target 生成及等待时间不计入 draft。

13. **利用现有测试先例。** 复用现有分布式启动、固定输入和持久化 fixture，按新的主要集成边界验证新流程。以下测试提供不同方面的先例，其文件存在或旧路径的通过记录不表示新 TorchTitan 独立训练流程已验收：

| 先例 | 可复用的验证方式 |
| --- | --- |
| [DSpark 训练基线](../../tests/test_dspark_training_baseline.py)与 [SelectiveAC 对照](../../tests/test_dspark_selective_ac.py) | 真实 Qwen 固定 features、不等 microbatch 分母、完整更新、梯度和 optimizer 状态对照 |
| [Qwen producer 隔离](../../tests/test_qwen_producer_isolation.py)、[draft feature reader](../../tests/test_draft_feature_reader.py)与 [Qwen vLLM 特征](../../tests/test_qwen38_vllm.py) | 生产配置与消费布局分离、输入与特征语义校验、producer 分片重组 |
| [Qwen 阶段 checkpoint](../../tests/test_qwen_phase_checkpoint.py)与 [checkpoint roundtrip](../../tests/test_checkpoint_roundtrip.py) | 真实训练状态快照、跨进程场景、DCP-only 提交、RNG/数据游标和保存故障 |
| [GLM 分区生命周期](../../tests/test_glm5_partitioned_model_swap.py) | 分区计划、缓存回收条件、journal 与 checkpoint 进度核对、worker 退出与异常场景；其中 mock 生命周期案例只提供契约先例 |

旧训练入口中的基线继续用于算法对照；新架构验收必须经过实际 TorchTitan 训练入口。正常阶段始终采用完整 update 边界。

14. **验收证据。** 数值容差依据固定环境中的基线误差预先确定，记录确定依据，不以任意放宽容差获得通过。记录实际参与的 GPU/rank 数、模型规模、上下文长度、拓扑、软件构建与工作量。被 skip 的多卡测试不算通过，CPU/toy 测试不能替代真实 Qwen GPU 数值与 128K 验收。本规格尚无新组件数值或性能已通过的声明。

15. **编排中断与进度核对。** 覆盖 checkpoint 提交前中断、提交成功而 DeepSpec 尚未记录阶段完成时中断，以及 worker 异常退出。验证重启选择有效提交，已提交更新不会重复，未提交工作从恢复游标重跑；连续多个分区的 scheduler 不重新 warmup。

16. **准备与交接契约。** 通过同一阶段入口验证准备出的 tokens/mask/特征需求与训练输入一致，数据准备期间没有 draft GPU 模型占用；验证 target 资源释放后才开始 draft。输入中缺失 shard、错误样本或 token 对应关系、层序/dtype/shape 不符、最终层语义错误以及不兼容恢复元数据应得到明确失败结果，不能消费不完整产物、推进阶段或错误清理可恢复数据。固定 feature fixture 用于隔离数值测试；另外保留既有 vLLM 入口的小规模交接验证，确认真实生产结果能被同一训练入口消费，不以 fixture 替代 target 集成证据。

## Out of Scope

- 优化、替换或迁移 target model、vLLM worker、target 推理内核/精度/并行方案/生成策略；target 特征生产耗时不属于本规格的性能目标。
- 旧 DeepSpec 完整训练状态转换、第一版的常驻 draft 训练进程卸载方案，以及将两端阶段调度或资源卸载控制转交 TorchTitan。
- 将 DSpark 目标切换为 TorchTitan 默认的整步有效 token 平均，或通过调整 loss、样本分组、GAS、anchors、冻结集合、优化器精度、通信精度或 LR 配方获得不同算法的提速。
- 正常阶段切换中的半步梯度快照、提前 optimizer update、丢弃未完成累积，或保留 draft GPU 状态冒充完整卸载。
- 同一训练任务各 draft 阶段之间的拓扑变化、弹性恢复，以及任意后端、schedule、dispatcher、degree 的笛卡尔积支持。
- 第一阶段就完成所有 dense/MoE 组合。后续 dense 组合与 MoE EP 仍属于整体交付范围，必须分阶段完成。
- 把真实 Qwen TP8、默认 PP8、native ring CP 与外层 compile、DeepEP 与 SAC 等未满足约束的组合直接标记为可用。
- 首版将 frozen 权重从完整 DCP 外置为共享引用、在持久化完成前进入下一 target 阶段，或要求保存失败后的原地自动重试。
- 默认重建 Python 环境、重装源码编译 vLLM 或替换其 PyTorch/CUDA 依赖栈；若现有基线存在具体兼容冲突，需要重新讨论。
- 凭当前讨论宣称已测得加速比或规定未经基线支持的吞吐目标。
- 在本轮设计讨论中直接修改训练实现、安装依赖或执行训练。

## Further Notes

- 本规格综合仓库记录的已确认讨论与本次 `/to-spec` 请求，沿用项目术语 Draft model、Target model、Feature partition 和 Draft training phase，并遵循 [ADR-0001](../../docs/adr/0001-preserve-dspark-training-semantics.md)（DSpark 训练语义）、[ADR-0003](../../docs/adr/0003-unload-draft-between-training-phases.md)（完整阶段卸载与恢复）和 [ADR-0004](../../docs/adr/0004-separate-feature-delivery-and-draft-training.md)（最新职责、配置与进程生命周期）。ADR-0002 的 DeepSpec 训练循环归属已被替代；旧设计分析保留为历史依据。
- 用户最后确认继续使用现有环境及源码编译的 vLLM；独立新建 draft 环境没有被确认为默认方案。性能目标已经由早期“暂不考虑”修正为尽量提高 draft 训练、保存、卸载和恢复性能。
- 当前工作树包含与本规格有关的既有未提交代码和测试修改。实施前需固化实际工作树基线，不能仅以历史分支或上游 HEAD 作为数值参考。
- 既有子 issue 01–05 记录了旧路径的基线、producer 隔离、完整 update 分区、SelectiveAC 和阶段 DCP 工作；这些完成记录保留为先例，不表示 ADR-0004 下的新训练所有权与进程生命周期已实现。新入口上的真实模型、SelectiveAC/FSDP2、完整恢复、候选并行矩阵、兼容性和性能收益仍需验证。
- 本规格与 ADR-0004 是新架构的实施依据。已有子 issue 中继续要求 DeepSpec 持有训练循环、optimizer 或 checkpoint 的条款已被替代；继续实施这些子 issue 前，应据本规格更新其职责与验收描述。本次不重新拆分任务或改写已有完成证据。
- 沿用仓库记录的本地 Markdown issue tracker，在原有母规格上更新。当前为 `design-review`，仅待主要测试边界确认后发布为 `ready-for-agent`；本次调用不另设整体设计审批。
- 本次只整理和发布规格；未修改训练实现、安装依赖或运行训练/GPU 验证。`ready-for-agent` 表示规格可供后续实施，不表示实现或验收已经完成。
