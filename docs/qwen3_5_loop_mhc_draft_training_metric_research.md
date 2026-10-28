# Qwen3.5 Loop/mHC 图文 DSpark 草稿训练速度指标调研

调研日期：2026-10-23；代码基准：`064ff2f`。范围：`examples/qwen3.5-loop-mhc-image-text-dspark-1-node` 的 DSpark 草稿模型训练速度。用户已限定只搜索 draft TP/SP，保持模型结构不变。数据、监督、anchors、block、loss、精度和 global batch 同样固定，以隔离并行策略的效果。本次没有启动训练或 autoresearch 实验。

## 建议选择

**对仅搜索 TP/SP 的当前任务，首选单一指标为固定真实特征重放负载的平均 optimizer update 耗时：`draft_replay_seconds_per_update`，越小越好。** 每个 update 固定消费同样的 16 个真实窗口；用一整个测量块的墙钟时间除以更新次数，再对相同负载的重复测量取中位数。目标特征生成在测量前完成，结束边界等待所有训练 rank 完成。初步沿用示例的 4 张草稿训练 GPU，目标推理仍为原来的 TP=4。

前一轮建议的输入 token 吞吐仍可附报；在相同数据和相同卡数下两者排序等价。改用耗时作为 TP/SP 搜索评分，可以避免新 TP 维度使 token 重复计数：当前 [`_merge_training_metrics`](../torchspec/controller/loop.py)直接相加所有 rank 的 token 数，尚没有排除 TP rank 上复制的同一批输入。现有计数可处理 USP 分片，不能因此推定已支持 TP 去重。

这是建议建立的 benchmark 契约，**当前仓库尚未证明已具备可直接使用、与本示例等价的高吞吐重放基准**。现有 `perf/train_tgs` 可以复用输入 token 计数和在线诊断，但不应未经验证就当成纯草稿训练能力的评分。相关边界来自[性能指标定义](performance_metrics.md)和[离线训练说明](offline_training.md)。

在线固定任务 wall-clock 用于验证最终收益。模型结构、训练算法和监督量不属于本次搜索空间，因此本次不以收敛算法变化或 time-to-quality 搜索代替 TP/SP 测速。

## 当前 TP 和 SP 的支持边界

现有示例的 `inference.vllm.tp_size=4` 控制目标模型，草稿侧采用四卡 FSDP。[`TrainingConfig`](../torchspec/config/train_config.py)有 `sp_ulysses_size` 和 `sp_ring_size`，没有草稿 TP 字段；[`DFlashTrainer.init_model`](../torchspec/training/dflash_trainer.py)只给继承此初始化路径的 DSpark 应用 FSDP，没有 TP 切分步骤。

现有 SP 也不能直接用于本例。[`_validate_usp_args`](../torchspec/train_entry.py)要求 `inference_engine_type=sgl`、`fsdp_strategy=REPLICATE`、micro batch=1；本例采用 vLLM 和 FULL_SHARD。DSpark 使用的 [`DFlashAttention`](../torchspec/models/draft/dflash.py)调用本地 FlexAttention/SDPA；[`_draft_backbone`](../torchspec/models/dflash.py)根据本地输入长度生成位置和采样 anchors。去掉入口校验不足以证明它正确支持跨 rank 的完整上下文。

CPU 上直接执行源码的 batch 推导和 USP 校验函数得到：

| 设置 | 推导的 DP | accumulation=4 时的 global batch | 本例的校验结果 |
| --- | ---: | ---: | --- |
| SP=1 | 4 | 16 | batch/USP 校验通过，沿用原有路径 |
| SP=2 | 2 | 8 | 拒绝 vLLM；将后端改为 SGLang 后仍拒绝 FULL_SHARD |
| SP=4 | 1 | 4 | 同上 |

这只是无 GPU 的配置支持检查，没有测出任何组合的训练速度。当前可用的原路径可表示为 TP=1、SP=1，不能据此声称它是最快组合。

如果采用独立的 TP 与 USP 维度，并补齐等价的并行实现，在固定四卡、micro batch=1、global batch=16 的条件下，候选度数如下。SP 在这里指仓库的 Ulysses/Ring 序列切分，不是与 TP 共用同一组 rank 的 Megatron Sequence Parallel。

| draft TP | draft SP | 派生 DP | 保持 global batch=16 所需 accumulation |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 4 | 4 |
| 2 | 1 | 2 | 8 |
| 4 | 1 | 1 | 16 |
| 1 | 2 | 2 | 8 |
| 1 | 4 | 1 | 16 |
| 2 | 2 | 1 | 16 |

这里 `DP × TP × SP = 4`，accumulation 是维持有效 batch 的派生值，不是额外搜索维度。SP 的 Ulysses/Ring 分解仍需按实现合法性确定；这些是待实现并验证的候选，不是六个已经可运行的配置。现有[全量驱动](../examples/qwen3.8-27b-image-text-1node/full_train.py)还固定 accumulation=4，因此公平比较需要让其接受按并行度推导出的 accumulation，同时维持原来每步 16 个窗口和一轮 38 次更新。

## 为什么先隔离目标端供给

示例为单节点 8 卡：目标模型 TP=4，草稿模型 FSDP=4；草稿 micro batch=1、梯度累积=4，对应全局 batch=16。最大序列长度为 131072；DSpark 配置为 512 anchors、block size=7、五组目标特征，并使用 CE、L1 和 confidence loss。`train_with_decode=false`，已有 assistant 监督段是学习目标。来源：[示例配置](../examples/qwen3.5-loop-mhc-image-text-dspark-1-node/config.yaml)、[领域词汇](../GLOSSARY.md)、[所有 assistant 轮次监督决策](adr/0003-multimodal-dspark-all-assistant-supervision.md)。

历史运行提供了一个重要线索。只读统计 [`outputs/loop-image-text-20261014/metrics.jsonl`](../outputs/loop-image-text-20261014/metrics.jsonl) 的 38 条训练记录，选取 step 2–37 共 36 条，以避开首步和不足全局 batch 的尾部：

| 历史量 | 数值 | 可以说明什么 |
| --- | ---: | --- |
| `perf/dispatch_wait` 均值 | 325.172 s/update | 控制器等待可分发样本非常显著 |
| `perf/step_time` 均值 | 15.363 s/update | 当时训练侧记录的 step 区间远短于 dispatch 等待 |
| `perf/compute_time` 均值 / 中位数 | 6.800 / 6.097 s/update | rank 0 的计算路径诊断量 |
| `sum(dispatch_wait)` | 11706.209 s | 所选记录的 dispatch 等待之和 |
| `sum(step_time)` | 553.078 s | 所选记录的 step 时间之和 |
| `sum(dispatch_wait) / sum(dispatch_wait + step_time)` | 95.49% | 两个历史指标之间的比例，不能解释为 GPU 空闲率 |

这些数据提示：直接优化当前在线 `perf/train_tgs`，容易让评分主要跟随目标端样本供给、队列及调度条件变化，掩盖草稿实现的小幅加速。它们不能证明瓶颈就是 vLLM 的某个 kernel；背压、窗口长度和数据准备也需要分开诊断。

这不是正式 baseline：step 2–37 没有经过稳态确认；旧记录缺少当前的 `perf/draft_train_time`、`perf/train_tokens`、`perf/train_gpus` 和 `perf/train_tgs`。不能用旧的 `step_time` 或 `wait + compute` 拼装出可与新实现比较的 TGS。

历史[准备 manifest](../outputs/loop-image-text-20261014/preparation_manifest.json)包含 327 个训练源会话、595 个 DSpark 监督窗口和 28872 个 assistant 监督段。窗口展开后的输入 token 共 68934575，单窗口长度 11065–131071，中位数 129175；assistant 监督 token 共 21819758，单窗口图片数 0–22。这是一个长度和多模态负载都需要保留的真实工作负载，不能用固定短文本或随机张量替代后仍声称获得了该示例的训练加速。

## 候选指标比较

| 指标 | 本次用途 | 局限与选择理由 |
| --- | --- | --- |
| **`draft_replay_seconds_per_update`：固定全局 batch 的整块时间 / 更新次数** | **TP/SP 搜索的单一评分，最小化** | 固定每次更新的真实窗口、监督、卡数和完整完成边界 |
| `draft_replay_tgs`：固定真实特征重放的整块输入 token / draft GPU / s | 附报吞吐；固定负载下与耗时排序等价 | 引入 TP 后必须对同一逻辑输入去重，不能直接相加复制输入的 rank 计数 |
| 现有 `perf/train_tgs` | 在线复验和运行诊断 | 包含 dispatch/data waits，可能主要反映目标端供给；逐步时间口径不是整项任务完成时间 |
| `perf/draft_train_time` / update | 单步定位与现有日志观察 | 固定工作量时有用；不同长度、batch 或累积方式下，单步快并不等于完成同一训练量更快 |
| `perf/compute_time` | 计算、通信路径优化的辅助诊断 | 来自 rank 0，不能代表最慢 rank；不覆盖完整 CPU、输入与传输成本 |
| sampled supervised positions / s | 检查实际监督工作量的辅助计数 | DSpark 有稀疏 anchor 和不同 loss；需定义采样、位移、有效 mask 后的位置，不能直接使用输入长度或 `loss_mask.sum()` |
| samples / s、GPU utilization、MFU | 容量和瓶颈观察 | 窗口长度、图片和监督密度不同；它们不直接回答同一训练任务完成得多快 |
| 固定在线任务 wall-clock | 最终部署流程复验；优化整个训练任务时可作主指标 | 包含目标模型供给，适合整体收益，不适合单独筛选小幅草稿 kernel 改进 |

上表是结合本仓库边界作出的指标选择，不是外部论文对本实现的性能结论。已有指标的精确定义见[性能文档](performance_metrics.md)。

## 单一评分的定义

对每次重复测量，固定一个包含若干完整 optimizer updates 的负载块：

```text
K = 测量块内固定的 optimizer update 次数，每次消费相同的 16 个真实窗口
N = 测量块的逻辑 input tokens 总数，TP 复制输入只计一次，SP 分片无重复合计
T = 从开始消费该块，到所有 rank 最后一次更新、GPU 工作及该块必需收尾完成的墙钟秒数
G = 4  # 固定草稿训练 GPU 数，目标推理 GPU 不计入此指标

draft_replay_seconds_per_update = T / K
score = 多次相同负载重复测量的 draft_replay_seconds_per_update 中位数
direction = lower  # codex-autoresearch 的方向参数
draft_replay_tgs = N / (G * T)  # 附报，计数必须先适配 TP
```

这里的 GPU 归一化只用于固定 4 卡的草稿训练效率比较，不计目标特征生成成本。若允许重新分配目标/草稿 GPU 数，应另外比较同一整机资源下的在线任务完成时间，不能靠改变分母选择“更快”的方案。

`N` 沿用 `perf/train_tokens` 的语义：排除 padding、dummy sample、USP lookahead 重叠；保留输入上下文。它不是新生成 token，也不是唯一源会话 token，且不是 DSpark 被采样监督位置。历史上下文在不同窗口中重复处理时仍会计数。固定完整窗口集合和监督契约，是这一计数有意义的前提。[仓库性能指标](performance_metrics.md)

`T` 包含重放条件下的数据取得、H2D、CPU 准备、forward/loss、backward、训练通信、optimizer 更新、必需的训练指标归约和更新间调度。目标特征生成、模型启动、编译 warmup、评估和 checkpoint 保存不计入这一稳态实现评分，分别记录其耗时。基准中的磁盘日志和曲线导出统一放在计时块外；这一规则在 baseline 与候选间保持一致，训练所需的指标归约仍在计时内。不要把输入完全预置到 GPU 的纯计算 microbenchmark 与上述评分混用。

应在测量块外部设置同步边界：开始前确保 warmup GPU 工作完成；结束时确保所有 rank 的最后更新和该块必需收尾完成，再停止统一的 monotonic wall-clock。不要在每个 micro-batch 额外同步。PyTorch 官方说明 GPU 调用默认异步，未同步的 CPU 计时可能只得到 kernel 提交耗时；CUDA events 和适当同步可定义真实完成边界。[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)

不要简单平均每步 `perf/train_tgs`。输入量不同的 step 需要先汇总工作量，再除以可比的时间。当前 DSpark 继承的 [`DFlashTrainer._aggregate_metrics`](../torchspec/training/dflash_trainer.py)通过 `.item()` / `.tolist()` 立即取出指标，并非 deferred 路径；但 `perf/draft_train_time` 本来就排除了指标归约和更新间开销，因此 `sum(draft_train_time)` 不等同于整个测量块的 elapsed wall time。[仓库性能指标](performance_metrics.md)

外层整块计时还可以正确评价后续异步化收益：若未来采用 deferred metrics，相邻 step 的 CPU 提交和 GPU 完成区间可能重叠，更不能未经检查就叠加各自持续时间。保留现有逐步时间作为诊断量，不把此潜在问题误报为当前 DSpark 已存在的问题。

## 建议的测量协议

1. **先建立可复现输入。** 固定真实训练源、窗口 IDs、顺序、长度、图像/文本构成、mask 和目标特征。选取可承载的代表性测量块，覆盖实际长度与图片分层；记录选择规则和 input/监督计数。不要通过删长样本、删图片或少监督 assistant 轮次提高分数。
2. **验证重放等价。** 必须保留执行层 3、27、51、75、78 的五组特征与独立 post-norm 最终状态，以及真实 embedding/LM head、位置和监督布局。[捕获契约 ADR](adr/0002-loop-mhc-dspark-capture-contract.md)明确这些语义是兼容性约定。[现有离线流程](offline_training.md)说明其尚未针对吞吐优化，不能假定可以直接作为本示例的基准；应先检查 Loop/mHC 和图文支持、缓存容量、传输路径与供给速度。
3. **固定可比状态。** 保持相同模型/checkpoint、优化器状态、随机种子、精度、4 卡硬件、global batch、监督窗口、anchor/block/loss 语义与训练日程。每次重复运行相同 warmup 和测量序列，避免从不同训练进度比较。若配套调整 microbatch/累积，在维持 global batch=16 之外，还必须验证相同样本、anchor 目标和梯度归一化保持等价。
4. **先 warmup，再测固定块。** 排除初始化、lazy loading、编译及形状缓存建立造成的冷启动；warmup 规则在 baseline 确立后固定，不针对候选版本挑选有利区间。保证重放供给能持续满足训练，且没有把磁盘读取瓶颈误认为草稿能力。PyTorch 官方基准教程明确要求 warmup，并用分块重复测量评估波动。[PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
5. **重复、成对比较并保留波动。** 建议从至少三次相同负载重复开始，交错比较 baseline/candidate；每次计算整个块的 `T / K`，报告中位数及范围，并附报正确去重后的 `N / (4T)`。比较全部合法 TP/SP 候选，选择正确性通过且重复测量最快者；本次不预设未经测量的百分比目标。
6. **再做原在线流程复验。** 用同一 workload 检查在线完成时间、`perf/train_tgs`、dispatch/data/compute、显存峰值和 OOM。草稿重放加速不保证当前目标端供给受限的作业同比加速。完整任务耗时应包含其实际发生的初始化、预填充、评估和保存成本，不能从训练 step 时间简单相加得出。

## 防止得到“更快但训练变了”的结果

性能改动必须维持本示例的图文上下文和所有 assistant 监督段语义，包括已有 reasoning/tool-call 内容；目标预填充后用于验证完成的单 token 不属于监督目标。[监督 ADR](adr/0003-multimodal-dspark-all-assistant-supervision.md)、[术语表](../GLOSSARY.md)

需要保留 loss 各分量、训练接受概率估计、实际有效监督位置/anchors、有限梯度及参数更新等检查。并行切分及归约次序可能影响浮点结果，数值允许偏差应根据原实现的重复波动和验证结果设定；精度不作为本次搜索维度。loss 曲线不能单独证明草稿推理效用保持。

训练接受概率估计只描述已有训练位置上的草稿/目标分布重叠，不是实际解码接受率。现有图文 prefill 训练不产生真实草稿提出/接受计数；当前实现优化先以固定输入上的 loss、梯度与参数更新等价性作检查。真实解码验收还需要确认 Loop/mHC 后端和图文评测入口支持，然后在固定验证集、固定 decoding 配置下检查真实 acceptance length 和实际推理加速。EAGLE-3 原文把实际 speedup、平均 acceptance length、逐步 acceptance rate 分开报告；其定义支持把这些作为草稿效用的验收指标，而不能用它们取代训练速度评分。[示例说明](../examples/qwen3.5-loop-mhc-image-text-dspark-1-node/README.md)、[EAGLE-3 原论文，第 4 节](https://arxiv.org/html/2503.01840v2#S4)

减少 anchors、缩短 context/block、更换 loss、改变数据覆盖、模型结构、全局 batch 或学习率日程均不属于本次 TP/SP 搜索。还需验证各并行布局具有相同的全局 anchor 目标、上下文可见范围和 loss/梯度归一化，避免因多 rank 各采一份 anchors 或只看本地上下文而改变训练。

## 下一步交付边界

本次指标限定为 **`draft_replay_seconds_per_update` 最小化，在线 wall-clock 复验收益**。当前前置问题是是否允许补齐缺失的 draft TP/SP 并行实现；若严格限于现有配置，则本例没有可比较的 TP>1 或 SP>1 路径。允许补齐时，下一阶段需要完成并行等价性检查、真实特征回放和整块计时，再测 baseline 并搜索合法组合。当前没有启动或恢复自动实验。
