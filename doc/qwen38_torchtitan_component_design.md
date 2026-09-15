# DeepSpec 的 TorchTitan 并行适配与阶段卸载设计草案

2026-09-14 职责边界更新：最新整合方案见 [DeepSpec 编排与 TorchTitan 训练设计](deepspec_orchestration_torchtitan_design.md) 和 [ADR-0004](../docs/adr/0004-separate-feature-delivery-and-draft-training.md)。DeepSpec 按特征分区调度两端并控制资源卸载，TorchTitan 拥有 draft 训练与原生组件配置；本次先 Qwen3.8 DSpark，保存后退出、下阶段启动恢复。本文关于保留 DeepSpec 训练循环和常驻训练进程的讨论属于历史方案，下文保留其分析依据；实施约束以已修订规格及最新设计为准。

本轮讨论已整理为 [功能规格](../.scratch/dspark-draft-torchtitan/spec.md)，包含用户故事、实施约束和测试决定，已发布至用户选定的本地 Markdown issue tracker，发布时状态为 `ready-for-agent`。本文件保留设计讨论依据，后续规格内容与状态以 tracker 中的文件为准。

补充已确认边界：TorchTitan 定义数据预处理与特征需求的规则和配置，DeepSpec 调用其数据准备入口。本次保证新流程完整断点恢复，旧 DeepSpec 完整训练状态转换不纳入本次范围。Q1–Q9 已明确，当前整合方案等待用户整体核对。

日期：2026-09-14。状态：设计讨论中。已确认保持 DSpark 训练语义和训练循环，首个组件选择 SelectiveAC，初始验证场景为单机八卡 Qwen3.8 DSpark。总体目标为 draft 侧 TorchTitan 全部并行能力适配，以及 draft 阶段之间完整 GPU 训练状态卸载和恢复。Q5、Q8–Q21 已记录：Q16 仅统计 draft 训练、保存、卸载和恢复，排除 vLLM 生成 target features；Q20 分开特征生产与 draft 消费布局；Q21 按用户要求继续使用现有环境和源码编译的 vLLM。所有优化限定在 draft 侧；性能越高越好，须同时满足训练语义、完整卸载与持久化恢复约束，尚未指定吞吐或加速比数值。设计选择的确认不代表授权实施，代码改动和训练执行等待用户明确要求开始实现。早先完整 Trainer 迁移分析保留在 [历史设计](dspark_torchtitan_refactor_design.md)。

## 已确认的决定

- 训练语义遵循 [ADR-0001](../docs/adr/0001-preserve-dspark-training-semantics.md)。
- DeepSpec 保留训练循环和 target/draft 生命周期，以组件方式复用 TorchTitan，先用 Qwen3.8 DSpark dense draft 验证，见 [ADR-0002](../docs/adr/0002-retain-dspark-loop-for-torchtitan-components.md)。
- 首阶段选择 TorchTitan SelectiveAC，单机八卡先做小规模数值对照，最终包含当前真实 Qwen draft 尺寸与 128K 输入；八卡初始拓扑拟沿用当前启动器的 DP shard2 × TP4、CP1。
- 每个 draft 训练阶段结束后卸载完整 GPU 训练状态，下一阶段恢复继续训练，包括模型、FP32 master weights、optimizer 状态及训练进度，见 [ADR-0003](../docs/adr/0003-unload-draft-between-training-phases.md)。
- 缓存分区对齐完整 optimizer step，保持样本顺序、GAS 和每次 update 的 microbatch 分组，允许分区大小和数量变化。
- 每个阶段同步将完整 draft DCP 与必要元数据落盘并确认提交成功后再卸载；HF/safetensors 按评估或最终交付需要导出。同一训练任务的各 draft 阶段保持拓扑一致，target 沿用独立配置。
- 新 checkpoint 提交成功后滚动保留最近两份，显式里程碑和最终产物单独保留；保存失败停止任务，重启后从最近成功提交的完整 checkpoint 恢复，首次提交前失败从初始状态重跑。
- 全部并行能力按明确的合法组合验收；后续接入以 `spmd_types` 为主要后端，PP 首先采用 1F1B，CP 允许保留 DSpark 专用 attention 通信实现。
- 交付顺序为单机八卡 Qwen + SelectiveAC + 阶段卸载恢复，然后 dense 并行组合，最后 MoE EP；后续阶段属于已确认的总体目标。
- MoE 首个验收模型选择 GLM-5.3-Flash DSpark，先用真实模型类的小规模配置和 native dispatcher 建立数值对照。
- 优化对象限定为 draft；在已确认的训练与恢复约束下尽量提高 draft 训练、保存、卸载和恢复性能，统计中排除 vLLM target 特征生成。
- 固定既有 target 特征生产配置与样本规划，新增 draft 专用索引和 reader，解耦生产布局与消费布局，保留原始 feature 内容。
- 继续使用现有环境和源码编译的 vLLM，保留解释器、源码、编译产物及依赖组合；TorchTitan 接入以现有环境为兼容基线，不默认要求新建 draft 环境。

## Draft 与 target 的职责边界

TorchTitan 组件、并行策略、SelectiveAC/compile、optimizer/checkpoint、卸载恢复及性能调优均限定在 draft 侧。target model 和 vLLM 作为既有监督来源，沿用其模型实现、推理内核、并行配置、精度及特征生成策略。使用共享并行或配置模块时，draft 专用适配通过角色入口或显式配置隔离，避免改变 target 默认行为。

保留既有 target/draft 交接，只进行已确认的 optimizer-step 分区对齐与 draft 状态生命周期适配。缓存分区大小调整保持样本顺序、feature/label/anchor 契约和每次 update 分组；它不授权修改 target 推理算法或开展 target 性能调优。target features 准备好后，draft 对这些输入的消费、布局及通信属于 draft 适配范围。

源码存在需要解除的角色耦合：`deepspec/trainer/base_trainer.py:391` 默认让 target 配置/上下文引用 draft 配置，`:409–410` 用整个 `ParallelConfig` 的差异判断异构，连仅 draft 开启 AC/compile 都可能触发。Qwen vLLM trainer 拒绝异构布局（`qwen3_8_vllm_trainer.py:63`），并依据 draft TP/CP 计算 vLLM owner/device 分组（`:74`）。因此新 draft 拓扑或重计算策略不能直接写入共享默认配置并沿用这些推导；也不能通过同步修改 target 的 AC/compile 配置绕过检查。

当前 Qwen feature 记录包含 tokens、loss mask 和 hidden states，但没有稳定 sample identity 或全局消费顺序；这些信息隐含在 producer rank、partition 和文件序号中。producer 请求又来自 draft DP sampler，现有 worker completion 不能代替拓扑独立的索引。GLM manifest 的身份字段可作参考，但现有 reader 也不能直接承担通用重分发。

Q20 已确认：在编排层显式固定既有 target 运行配置与 producer 样本规划，新增 draft 输入索引和 reader，索引记录样本身份/全局顺序、原始文件及 CP shard 布局、update 归属。沿用 vLLM worker 和原始 feature 文件，由 draft 侧完成读取、分发与布局适配；相关开销计入 draft。producer 为 CP1 时读取完整 features 并按 draft 所需布局分发；producer 为 CP>1 时须收齐同一样本的 shards 并按原始顺序重组。缓存删除须等待所有 draft 消费者完成，不能继续按 draft CP rank 推导 producer 文件所有者。

解耦后的验收约束是：同一固定 producer cache 经不同 draft 读取布局后，对应样本的 token、loss mask、hidden features 与顺序一致；每个所选 draft 配置的逻辑 microbatch 分组、loss 分母、GAS 和 anchors/RNG 按其 DSpark 参考保持一致。跨布局重分发仅改变数据位置，不构成更改训练目标或样本顺序的授权。恢复索引和 checkpoint 的数据游标必须指向同一消费位置。

`apply_parallelism` 当前两个调用点都在 draft 首次构建/恢复路径（`base_trainer.py:611`、`dspark_trainer.py:1106`）；新适配入口应显式接收 draft 模型、拓扑与策略。GLM 的模型并行模块同时服务 target/draft，后续 EP 改造须限定在 draft 分支。

## 并行适配范围

以下是已确认要覆盖的并行能力；具体合法拓扑仍需列成验收矩阵，尚未实现。

| 能力 | 适配含义 | 当前需要解决的问题 |
| --- | --- | --- |
| DP replicate、FSDP2、HSDP | 数据复制、参数/梯度/optimizer 分片及其组合 | 与 DSpark 的 loss 归一化、阶段卸载和恢复保持一致 |
| TP | 张量维度分片 | teacher 特征、draft attention、Markov/confidence 与 head 的输入输出布局 |
| SP | TP 内的激活布局策略 | 当前 DSpark 明确拒绝；混合 target-context/draft-query 输入需自有适配 |
| Loss parallel | 按词表分片计算 loss | 当前 DSpark 明确拒绝；需覆盖 CE、分布 L1、confidence target 的全词表语义 |
| CP | 上下文维度分片 | 当前 Qwen 有 native ring 路径；Titan 通用 CP 与双序列 FlexAttention 契约不能直接等同 |
| PP | 将模型拆为多个流水线阶段 | 当前 DeepSpec 拒绝 PP；保留外层训练循环时，内层 microbatch forward/backward 需委派给 pipeline schedule |
| EP | 专家维度分片及 sparse 数据并行视图 | 用 GLM-5.3-Flash DSpark 验证；Qwen dense 的结果不能证明 EP 正确 |

`spmd_types`/`partial_dtensor` 是实现后端，AC、reshard、预取、async TP 和 PP schedule 是相关策略。用户已接受后续接入以 `spmd_types` 为主要后端、PP 首先采用 1F1B；其他后端和策略的范围单列，不隐含要求任意交叉。性能优化可评估这些策略，但须单独验证兼容性与收益。EP 是同一 dense rank 域的另一种视图；HSDP 是 DP replicate 与 shard 的组合，SP 和 loss parallel 也不是额外乘入 world size 的独立轴。

单机八卡可以分多组拓扑验证不同能力，不能据此证明全部规模与组合。当前 Qwen draft 有五层，PP 的合法 stage 数与实际分层方案有关；纯 PP8 不能仅由八张卡推导为可用。

### 八卡 draft 验收候选矩阵

真实 Qwen draft 从 target text config 继承 24 个 Q heads、4 个 KV heads，覆写为 5 层；在当前 GQA 整除规则下 TP 候选为 1/2/4，TP8 不成立。以下仅为算术与模型维度约束下的候选，尚不能证明已实现、组合合法或能跑 128K；先用短序列和固定 feature fixture 验证。

| 候选 | DP replicate | DP shard | TP | CP | PP | 目的 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 8 | 1 | 1 | 1 | 1 | 复制 DP |
| B | 1 | 8 | 1 | 1 | 1 | FSDP2 |
| C | 2 | 4 | 1 | 1 | 1 | HSDP |
| D | 1 | 2 | 4 | 1 | 1 | 首阶段拓扑；分别开启 SP、loss parallel，再组合二者 |
| E | 1 | 4 | 1 | 2 | 1 | DSpark 专用 CP |
| F | 1 | 1 | 4 | 2 | 1 | TP 与 CP 组合 |
| G | 1 | 4 | 1 | 1 | 2 | PP 1F1B 首验 |
| H | 1 | 1 | 2 | 2 | 2 | TP/CP/PP 代表组合，依赖双输入 stage/layout 设计 |

每行采用匹配该配置的 DSpark 数学参考，固定该行的样本、逻辑 microbatch、GAS 和 anchors。不同 DP degree 在相同 local/global batch 下会推导出不同 GAS，不能把跨行结果直接当作同一目标的更新等价或无条件性能对比；对照须保持逻辑 microbatch 的分母和权重。每个最终接纳的组合还须通过完整 DCP 提交、销毁 GPU 状态、恢复及下一次 update 与连续训练的对照。首次真实 128K 验收仍从 D 开始。

MoE 另列 GLM-5.3-Flash 候选：DP replicate1、DP shard8、TP/CP/PP1、EP8、EFSDP1，先用 native dispatcher。288 routed experts 可被 EP8 整除，但仍需验证 Titan sparse layout、专家缩放和恢复。

## 阶段卸载与训练连续性

当前 Qwen exact partition 按 microbatch 划分，允许阶段结束落在梯度累积窗口中间。按现有八卡 DP2×TP4、local batch=1、global batch=512 配置，GAS=256；分区结束不必已经发生 optimizer update。

当前通用 checkpoint 要求 optimizer-step 对齐，保存模型、optimizer、scheduler、RNG 和进度，但不保存未完成的累计梯度。GLM 现有 swap 通过 optimizer-step 对齐满足这个前提。FSDP 的未同步梯度可能位于内部累计 buffer，仅复制 `parameter.grad` 不能证明快照完整。

用户已选择重新划分 target 缓存分区，使每次正常卸载发生在完整 optimizer step 后；保持样本顺序、GAS 与每次 update 的分组，允许分区大小和数量变化。不能为提前卸载而提前 optimizer.step、缩短 GAS 或额外丢弃样本。恢复未完成梯度、布局和归约状态的半步快照不纳入正常阶段切换契约。

现有 `deepspec/trainer/base_trainer.py:261–295` 将每 epoch 样本数截为完整 global batch，`deepspec/utils/distributed.py:95–101` 按 `seed + epoch` 洗牌后取对应数量的样本。因此正常 epoch 结束不存在不足 GAS 的训练尾部；`max_train_steps` 也停在完整 update。新分区方案保留这些取样和停止规则。

完整 GPU 卸载范围已确认：模型参数与必要 buffers、optimizer 的 FP32 master/moments、梯度、FSDP 临时 buffer、预取 batch 及其他 draft 引用持有的 GPU 状态。阶段切换先完成 update、排空在途计算和通信，再完成状态保存并释放；PP 路径须排空流水线。释放后无需保留半步梯度。保留进程的 CUDA context 与通信运行时基础占用不要求归零。

用户已确认每阶段完整 DCP 与必要元数据同步落盘、提交成功后再卸载。同一训练任务中，各 draft 阶段采用相同拓扑，target 沿用其独立配置；跨拓扑恢复另行设计。下一阶段恢复模型、FP32 master/moments、Adam 计数、scheduler、RNG 和数据游标；训练 RNG 在重建完成、下一 draft forward 前恢复。HF 按需导出、最近两份阶段 checkpoint 保留与保存失败后重启恢复的规则见 [ADR-0003](../docs/adr/0003-unload-draft-between-training-phases.md)，具体存储位置依环境细化。

## 已确认的交付顺序

1. 单机八卡 Qwen3.8 DSpark：SelectiveAC 接入、完整 optimizer-step 分区、完整 GPU 状态卸载与恢复，验证 DSpark 更新语义连续；包含真实 draft 尺寸与 128K 输入，并评估性能。
2. Dense 并行组合：覆盖 DP replicate、FSDP2/HSDP、TP/SP/loss parallel、CP、PP，并逐项验证与阶段卸载的组合。以 `spmd_types` 为主要后端、PP 先用 1F1B、CP 保留模型专用适配；具体拓扑矩阵待细化。
3. MoE EP：使用 GLM-5.3-Flash DSpark，先以小规模配置与 native dispatcher 验证专家布局、通信、梯度缩放及完整状态恢复，再扩展规模和组合。Qwen dense 验证不能代替这一阶段。

## 当前接口与可复用范围

现有 `BaseTrainer.train` 调用 `iter_training_batches` 和模型家族的 `run_batch`，并协调梯度累积、同步、裁剪、optimizer、scheduler 和 checkpoint。模型并行化集中在 `deepspec/distributed/parallelize.py:apply_parallelism`，顺序为模型并行、activation checkpoint、compile、FSDP2。具体模块的适配应集中在相应接口内。

| 候选模块 | 本地源码事实 | 首阶段判断 |
| --- | --- | --- |
| TorchTitan SelectiveAC | `ActivationCheckpointing.apply` 遍历 `layers.named_children()`，兼容当前 Qwen 的 `ModuleList`；保存算子策略包括 FlexAttention 和部分通信算子 | 已选为首阶段模块；梯度及显存/计算取舍仍需验证 |
| TorchTitan FullAC | 当前 DeepSpec 已有 HF/PyTorch full activation checkpoint，包装逻辑较短 | 可以作对照；仅替换这段包装没有明确的维护收益 |
| TorchTitan decoder FSDP helper | 要求 `layers.items()`、`tok_embeddings` 等模型结构，并关闭 FSDP 梯度除法；其显式预取针对 MoE/EP 路径 | 需要额外模型结构与归一化适配；当前 Qwen 已有 FSDP2 和预取，直接替换的收益待证 |
| optimizer / checkpoint manager | 当前有 FP32 master weights、optimizer state、分区与恢复元数据契约 | 阶段卸载设计需要明确训练状态保存与恢复；是否更换 manager 另行决定 |

本地 TorchTitan 参考版本为 `f6b9152e9bedcc18f5dc339b9f88265e5a07e988`。上述源码判断尚不足以证明真实 Qwen + SelectiveAC + FSDP2 的数值与性能表现。

## 首阶段设计：SelectiveAC 与当前 FSDP2 配合

用户已确认首个组件选择该方案，并要求尽量提高性能。它需要与全并行适配和阶段卸载的总体设计协调。

拟在现有 activation checkpoint 接口中选择 TorchTitan SAC，随后仍由当前入口应用 FSDP2。保留现有 FSDP 梯度平均约定，原 DSpark loss 的归一化代码可以继续使用。首次验证关闭 model compile，保持 RNG 状态；SAC 与 HF 内置 gradient checkpoint 由同一个策略选择管理，每个 block 使用一套重计算机制。

TorchTitan SAC 的普通 import 会涉及 `spmd_types`、`torch_remat`、`tyro` 等依赖，并调用私有 Dynamo 接口。已核对的共享生产环境 metadata 显示 torch 2.13.0；默认 shell Python 则是另一套 torch 2.11.0 环境。此前在生产环境 site-packages 中未发现上述三个依赖的发行包 metadata；它们在现有环境基线上的兼容性与交付方式仍需验证，真实 Qwen 的数值与恢复验证尚未完成。

Q21 已按用户要求调整：vLLM 是源码编译安装，继续使用现有环境，保留当前解释器、源码位置、编译产物和依赖组合。TorchTitan 参考固定完整 commit `f6b9152e9bedcc18f5dc339b9f88265e5a07e988`，优先核对其与现有 Python/PyTorch/CUDA/Triton/Transformers 的兼容性；不默认新建 draft 环境或替换源码编译 vLLM。现有 `VLLM_PYTHON_BIN` 默认跟随 draft 解释器（`config/dspark/dspark_qwen3_8_27b_vllm.py:25`），设计中应显式固定为当前 target 解释器，并固定既有源码解析位置。现有 worker 参数支持这些配置，无需修改 worker。若后续核对发现必须改变现有环境基线，应带着具体冲突回到设计讨论，不能把环境升级当作已确认决定。

Selective activation checkpoint 通过选择保存部分算子结果、重算其余结果，调整显存与计算的取舍，参见 [PyTorch 官方说明](https://pytorch.org/blog/activation-checkpointing-techniques/)。对本项目的实际收益需要测量。

## 验证草案

先使用固定的 target feature fixture，固定权重与 anchors/RNG，将 teacher 生成差异从组件对照中排除。数值对照执行真实 Qwen3.8 DSpark 模型类及其 CE/L1/confidence 目标；测试模型规模和性能测量规模分别记录。

| 阶段 | 模型与拓扑 | 需要证明的行为 |
| --- | --- | --- |
| 数值基线 | tiny Qwen3.8 DSpark，TP1/CP1；先 FP32，再 BF16 | 对照输出、各 loss 项、所有可训练参数梯度、clip norm、FP32 master/moments 与连续更新 |
| FSDP2 组合 | 2 rank，DP shard2、TP1/CP1，GAS 至少为 2 | 不等有效分母、阶段内梯度累积、重计算与 FSDP 的组合；对齐分区后卸载恢复与连续更新对照 |
| 当前启动拓扑 | 8 rank，DP shard2 × TP4、CP1 | 旧路径与候选路径在同拓扑下的真实 DSpark 更新对照；连续多个阶段验证 GPU 状态释放和恢复 |
| 目标规模 | 8 rank，当前真实 Qwen draft 尺寸与 128K 输入 | 验证既有 target/draft 交接中的 draft 卸载、持久化恢复和资源释放，仅测量 draft 侧性能；不能由 tiny/smoke 代替 |

已有 Qwen 多卡测试主要检查 shape 与梯度存在性；已有严格 FSDP 更新对照使用 toy 模型。这些测试可提供 harness，但不构成真实 Qwen DSpark 与新组件的数值等价证据。具体数值容差依据固定环境中的基线误差确定，验收还应检查实际执行的 world size，避免将 skip 记为通过。

## 性能设计方向

用户已明确性能口径：固定训练工作量，只统计 draft 训练、保存、卸载和恢复的累计 wall time，排除 vLLM 推理生成 target features 的时间及等待该生产阶段完成的时间。主指标为：

```text
T_draft = sum_over_phases(T_train + T_save + T_unload + T_restore)
```

各分项计时互不重复：训练包含对已就绪 features 的读取/搬运、forward/loss/backward、梯度同步/裁剪和 optimizer/scheduler；保存包含 DCP 写入与提交，以及实际发生的按需 HF 导出；卸载包含 draft 计算/通信结束与资源释放；恢复包含 draft 重建、并行化、optimizer 构建和状态读取。draft 初始化或编译开销按实际发生位置归入恢复或训练并单独标注，不能因为处于首阶段或首次调用而从总计中隐藏。计时区间需要覆盖相应异步 GPU 工作的完成，以各 rank 的关键路径耗时反映多卡性能。

另报稳态 draft 吞吐、首阶段与后续阶段的分项耗时和显存。固定输入 features、样本顺序、逻辑 microbatch/GAS 与训练工作量，对照常驻旧路径参考值和满足卸载契约的基础实现，分别说明生命周期成本与组件收益；上述所有对比都排除 target 特征生成。阶段卸载会引入额外开销，不能只用稳态 step 时间代表 draft 全流程收益，也尚不能承诺相对常驻旧路径的加速比。

候选优化保持 DSpark 的 loss 权重、microbatch 分组、GAS、样本顺序、精度和 optimizer 语义。增加 local batch 并减少 GAS 即使保持 global batch，也可能改变当前 microbatch 等权平均的目标；不能当作无条件等价的提速手段。已获允许的缓存分区大小调整可在资源约束内摊薄阶段切换开销，仍须保持原有 update 分组。

先建立 SelectiveAC 与卸载恢复的正确性基线，再按 draft 实际瓶颈评估 AC 保存策略、FSDP 预取/reshard、并行拓扑、编译与缓存复用、checkpoint 数据路径。所有优化都需要记录上述 draft 分项与总计收益和资源开销。首次数值对照关闭 compile 是隔离变量的验证步骤，后续允许将 compile 作为性能候选；PyTorch 的 [编译缓存机制](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)提供跨进程复用相同图和配置的可能，模型重建后的实际命中与成本需在所选环境验证。

源码核对发现以下候选空间，尚未测量收益：

- `deepspec/trainer/ckpt_manager.py:243–283` 先导出完整 HF 权重，再保存 DCP。HF 导出会收集完整 CPU state dict（`deepspec/distributed/distributed_checkpoint.py:278–284`），阶段恢复使用 DCP。用户已确认 draft 阶段提交只写完整 DCP 与 config/identity/progress 元数据，HF 导出按评估和最终交付需要触发；现有完整性校验硬要求 safetensors，实施时需要同步调整。第一版 DCP 仍包含 frozen 权重，维持各份 checkpoint 独立可恢复。
- 现有 GLM 恢复先 `build_models` 读取初始化权重，再由 DCP 覆盖（`deepspec/trainer/dspark_trainer.py:1090–1148`）。候选方案复用 CPU config/tokenizer 和固定拓扑的 mesh，重建 GPU 对象后直接填充恢复状态，避免重复读取原 target 初始化权重；meta/empty 初始化与 FP32 master optimizer 的组合仍需验证。
- 调整 SAC 保存策略、FSDP child reshard 与预取深度可以评估通信/计算/显存取舍；保持 root-owned heads 的生命周期及原归约、通信精度。当前 DSpark native ring CP 明确拒绝外层 model compile，不能将 CP 与 compile 同时标成已支持。
- 完整 DSpark vocab-parallel loss 是减少全词表 gather 的候选，需同时保留 CE、L1、acceptance/confidence 和相关 head 的数学定义及梯度，不能只替换 CE 就宣称完成。

每阶段同步持久化后卸载是已确认约束。PyTorch [异步 DCP](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)涉及 staging 内存与并发开销；仅切换到异步 API，随后立即等待落盘，不能据此推断会隐藏整个保存开销。若后续希望在持久化完成前进入 target 阶段，须另行讨论这一时序变更。

## 当前待决定的问题

Q5、Q8–Q21 已记录：Q16 按用户修正统计 draft 训练、保存、卸载和恢复，排除 target 特征生成；Q17–Q19 采用已推荐的 DCP/按需 HF 导出、最近两份保留和保存失败后重启恢复方案；Q20 解耦 producer 配置与 draft 消费布局；Q21 继续使用现有环境和源码编译的 vLLM。所有优化限于 draft。

checkpoint 存储位置沿用现有 `logging.checkpoint_dir` 配置，具体部署路径不硬编码；draft 并行验收矩阵列为待验证候选。上述性能方向仍需实验验证，未运行 benchmark，也未授权实现。

剩余技术验证包括现有环境下的 TorchTitan 兼容性、候选并行组合可行性、数值容差与性能收益；这些尚未验证，不以环境重建、target 改造或训练语义变化绕过。
