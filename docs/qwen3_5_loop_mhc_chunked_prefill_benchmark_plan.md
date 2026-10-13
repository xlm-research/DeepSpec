# Qwen3.5 Loop/mHC：128K prefill 分块预算测速需求

状态：2026-10-13 五组主对照及 32K/64K 的完整复测均通过；额外一轮外层采集中断的 32K 计时也已保留，八卡资源已释放。

## 已明确的目标

用户要求在 `examples/qwen3.5-loop-mhc-dspark` 的单机八卡场景下，比较 chunked prefill 开启及不同预算下处理 128K 上下文的速度。沿用已有四卡 TP=4 target、四卡 FSDP 草稿训练和 Triton mHC 加速配置。

用户进一步明确：“只处理 prefill 阶段的 128k 并且只生成一个 token”“只处理 128k 的上下文”。因此测速重点是长输入 prefill 后产生第一个 token 的耗时；整体训练吞吐不作为默认排名指标。各组输入长度相同，不能把不同分块预算误写成不同上下文长度实验。

## 参数和长度口径

- `enable_chunked_prefill` 是布尔开关。
- `max_num_batched_tokens` 是单次调度的 token 预算；在本实验单请求、无长 decode 的配置下，它控制长输入如何分块。
- “128K 输入 + 1 token 输出”已确认为输入 131072 tokens、输出 1 token；本次临时将 `max_model_len` 和训练长度上限均设为 131073，示例默认配置文件保留原值。
- 现有示例的 `max_model_len=131072` 和边界数据实际对应 131071 输入加一个内部输出 token。不得把旧数据直接标作恰好 131072 输入。
- 关闭分块时，预算必须覆盖有效 `max_model_len`；不能仅将开关改为 false 而保留 8192 预算。

依据：[单 token 采集入口](../torchspec/inference/engine/vllm_engine.py:679)、[输入长度校验](../vllm/vllm/v1/engine/input_processor.py:427)、[关闭分块的预算约束](../vllm/vllm/config/scheduler.py:249)。[vLLM 官方调优文档](https://docs.vllm.ai/en/stable/configuration/optimization/#performance-tuning-with-chunked-prefill)也将开关与 token 预算分开说明；具体合法配置以本仓库代码为准。

## 已核实的数据与计时条件

默认 `mydata/test_20.jsonl` 的有效输入长度为 7177–105647，没有满 128K 输入。此前默认数据训练成功，不能用作本轮满长性能结果。依据：[上轮训练验收](qwen3_5_loop_mhc_8gpu_revalidation_plan.md)。

`outputs/loop_dspark_verification_inputs/128k.jsonl` 与同名 parquet 含四条合成数据：token 序列相同，为八个 token 的周期重复，各 131071 tokens，最后 512 tokens 参与监督。四条仅够当前配置一次 optimizer 更新，不能只跑这一更新就声称取得稳态速度。若选择严格 131072 输入，应另建等长 fixture、保留原文件，并重新核对实际 token 数和监督 mask。依据：[原始边界校验](../outputs/loop_dspark_verification_inputs/boundary_fixture_validation.json)。

引擎强制关闭 prefix caching，Loop 模型也拒绝启用它，因此重复 fixture 不会自动跳过跨请求重复前缀计算；请求内部的 KV/SSM 历史仍用于连续分块。依据：[引擎设置](../torchspec/inference/engine/vllm_engine.py:438)、[模型约束](../vllm/vllm/model_executor/models/config.py:849)。

现有 `perf/infer_batch_time` 从发出 `engine.generate.remote()` 到结果返回，包含 Ray、目标计算和前向内的隐状态采集，排除取得引擎并发槽之前的等待。它不是纯 GPU prefill 时间，也不是该请求全部 Mooncake 数据已经发布可读的完成屏障：direct connector 的发布还有后续调度路径。该指标是近期请求聚合值，默认普通日志也没有完整持久化所有原始请求计时。依据：[计时边界](../torchspec/controller/inference_manager.py:470)、[延后发布](../torchspec/inference/engine/mooncake_hidden_states_connector.py:834)。

本地 vLLM 可返回逐请求首 token 时间：需要在现有 extra_args 中启用 `disable_log_stats=false`，保留 `RequestOutput.metrics.first_token_latency`。本次已在 TorchSpec 引擎补充逐请求 `PREFILL_METRICS` JSON 日志，并保留原训练结果字段。`first_token_ts - scheduled_ts` 可辅助观察调度后的 prefill 阶段墙钟时间。不能把 monotonic 的 `first_token_ts` 与 wall-clock 的 `arrival_time` 相减。这些指标包含实际引擎执行路径中的捕获及调度影响，不代表纯 CUDA kernel 时间，也不能据此拆出 Mooncake 发布耗时。

主指标已确认为逐请求首 token 耗时（TTFT），同时记录调度后的首 token 耗时、引擎 generate 返回耗时以及对应的输入 tokens/s。分母按各自计时边界记录，不声称是纯 GPU 时间；批次大小固定为 1。

训练进度条 `thru` 来自样本入池速度，不能当成 GPU prefill 吞吐或 optimizer 完成吞吐。依据：[进度条来源](../torchspec/controller/loop.py:417)。

## 已确认对照矩阵

| 组别 | chunked prefill | 单步 token 预算 | 每条输入 |
| --- | --- | --- | --- |
| 当前预算基线 | 开启 | 8192 | 固定满 128K |
| 16K 预算 | 开启 | 16384 | 与基线相同 |
| 32K 预算 | 开启 | 32768 | 与基线相同 |
| 64K 预算 | 开启 | 65536 | 与基线相同 |
| 不分块对照 | 关闭 | 131073 | 与基线相同 |

用户同意推荐的五组矩阵；不增加开启且预算 131072 的第六组。

除实验开关和预算之外，各组固定模型权重、草稿初始化、实际输入 token 和顺序、监督 mask、GPU 分工、Triton/BF16/FP32、eager、请求并发、特征采集、Mooncake、显存比例及训练 batch。必要的长度边界调整在所有组一致应用。

## 决策树与测量协议

```text
128K prefill 速度对比
├─ 已明确：指定 DSpark 示例、单机八卡、已有加速实现
├─ 已明确：各组相同的长输入，只产生一个 token
├─ 已确认：输入恰好 131072，总长度临时设为 131073
├─ 已确认：开启 8K/16K/32K/64K，加关闭组
├─ 已确认：合成满长负载
├─ 已确认：主计时取首 token 耗时，记录显存
└─ 执行协议：每组预热 8 请求、测量 8 请求，保存逐请求记录
```

数据采用合成固定满长负载，便于隔离预算变化，结论明确限定为该负载。沿用原八-token 周期，生成 16 个独立样本 ID，输入各 131072，末尾 512 token 监督；所有组复用同一份数据。每组从同一初始模型和 seed 开始，运行四次训练更新。

各组排除前八次请求作为预热，按请求顺序统计接下来的八次；异步预取产生的额外请求另存，不计入主测量。每秒采样八卡显存与利用率，报告采样峰值。训练保存移到四次更新结束后，计时窗口中不保存 checkpoint。

报告 TTFT 的均值、中位数、最小/最大及标准差；输入速率按 `8 × 131072 / sum(TTFT)` 计算，明确它是逐请求 TTFT 折算的速率，而非含请求间空闲的端到端训练吞吐。先完成五档扫描；若候选最快两档差距小于 5% 或测量明显不稳，再复测这两档，避免由单次波动选出赢家。

后续方案应明确计时边界，排除模型加载、首次编译和预热，保留每次请求的长度、延迟、有效输入 tokens/s、峰值显存及异常。大预算若失败，应记录失败阶段；不得通过缩短输入或改变精度，把失败配置变成貌似通过的同一组。

## 交付与执行状态

已交付各组配置、原始测量、延迟和吞吐对照表、显存及成功/失败状态。所有测量保存在独立运行目录，实测结果与建议如下。

沿用 [CONTEXT.md](../CONTEXT.md) 已有的分块预填充、单步 token 预算、请求总长度及目标端样本供给定义。该实验不涉及难以逆转的架构决定，无需新增 ADR。

## 首轮实测结果

以下五组均运行指定 DSpark 示例，使用单机八张 H800：目标模型 TP=4 占 GPU 4–7，草稿 FSDP 训练占 GPU 0–3。每组舍弃八次预热，统计随后八次请求；各组另有两次异步预取请求，已保留但不参与排名。

| chunked prefill | 单步 token 预算 | 平均 TTFT（秒） | TTFT 标准差（秒） | 输入 tokens/s | 目标端单卡峰值（GiB） | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 开启 | 8192 | 22.557 | 0.205 | 5810.6 | 66.29 | 通过 |
| 开启 | 16384 | 21.781 | 0.222 | 6017.8 | 67.19 | 通过 |
| 开启 | 32768 | 21.515 | 0.247 | 6092.2 | 69.18 | 通过 |
| 开启 | 65536 | 21.516 | 0.210 | 6091.9 | 73.17 | 通过 |
| 关闭 | 131073 | 21.566 | 0.221 | 6077.7 | 72.20 | 通过 |

目标显存为整次运行中每秒一次的 `nvidia-smi` 采样最大值，包含分配的 KV cache，并非精确的 CUDA 活跃张量峰值。自动显存预算及 allocator 行为会随配置变化，因此不能假设关闭分块一定比 64K 分块占用更多显存。训练端各组单卡峰值约 64.80 GiB。

首轮说明关闭分块也能处理完整 128K 输入，不是必须开启才能跑通。32K 相比 8K 的平均 TTFT 减少约 4.6%，而 32K、64K 和关闭组的 TTFT 接近。32K/64K 的微小先后顺序不足以作为选型依据，因此按既定协议对这两组复测。

五组均校验了相同的数据哈希、除实验参数和独立输出/缓存目录外一致的有效配置、连续覆盖 `0..131071` 的完整特征采集、每请求一个内部输出 token、零前缀缓存命中及四次有限值训练更新。五组全部正常退出，单组结束后八卡显存均已释放。原示例默认长度与分块配置未改写。

证据目录：[完整报告](../outputs/loop_dspark_prefill_sweep_20261013/report.md)、[各组统计 CSV](../outputs/loop_dspark_prefill_sweep_20261013/results.csv)、[逐请求原始记录](../outputs/loop_dspark_prefill_sweep_20261013/requests.csv)、[配置和输入一致性核对](../outputs/loop_dspark_prefill_sweep_20261013/config_comparison.json)。

## 独立复测与记录缺口

每次重新启动完整 DSpark 作业，仍预热八次、测量八次。以下两档各合并两轮有完整采集记录的实验，均为 16 个正式样本；五组首轮表仍各保持八个样本，不混用不同样本量排序。

| 单步预算 | 首轮平均 TTFT（秒） | 完整复测平均 TTFT（秒） | 两轮合并均值（秒） | 两轮输入 tokens/s | 目标端单卡最大采样峰值（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32768 | 21.515 | 21.559 | 21.537 | 6085.9 | 69.18 |
| 65536 | 21.516 | 21.601 | 21.558 | 6079.9 | 73.19 |

32K 与 64K 的合并 TTFT 仅差约 0.1%，远小于各自单请求的波动，不能将其解释为稳定的速度优势。32K 的目标端单卡采样峰值低约 4.01 GiB。

另有一次 32K 复测完成了全部八个正式计时和四次训练更新，TTFT 均值 **21.976 秒**、标准差 **0.493 秒**。外层采集进程在正式计时前中断，独立运行的 DSpark 子进程继续完成，但未留下退出码和完整显存采样。因此该轮不计入上面的完整采集表，原始计时完整保留，并另行补跑得到 21.559 秒这一轮。它也说明轮次间存在波动，不能只取有利的一轮声称 32K 绝对最快。

这一轮的有效输入、单 token 输出、零前缀缓存、完整特征采集和有限值训练更新均已独立核对。详见 [中断记录](../outputs/loop_dspark_prefill_sweep_20261013/on_32768_repeat/instrumentation_interrupted.json)、[原始计时](../outputs/loop_dspark_prefill_sweep_20261013/on_32768_repeat/requests.json)、[补充验收](../outputs/loop_dspark_prefill_sweep_20261013/on_32768_repeat/supplemental_validation.json)。

## 建议与最终验收

对本次固定合成负载，建议开启 chunked prefill 并将预算设为 **32768**，作为速度与显存的折中：相比当前 8192 有可见的延迟下降；继续扩大至 65536 或关闭分块，没有观察到足以抵消显存差异的稳定速度收益。关闭分块也已通过完整 128K 测试，因此开启并非跑通的必要条件。结论不外推到真实业务分布或多请求并发。

复用本次严格 131072 输入、一个输出 token 的实验设置时，覆盖参数如下；示例默认配置文件未修改：

```text
training.max_seq_length=131073
inference.vllm.extra_args.max_model_len=131073
inference.vllm.extra_args.enable_chunked_prefill=true
inference.vllm.extra_args.max_num_batched_tokens=32768
```

七次有完整采集记录的运行全部退出码为 0、无 OOM，每次完成四次训练更新；目标输入均完整采集，分块数分别符合 16/8/4/2/1。额外中断采集轮的 TTFT 与训练执行证据单列，不伪造退出码或显存峰值。最终确认无本次基准的存活进程，八卡均恢复到 4 MiB、0% 利用率。

本次仅增加逐请求计时日志和相关说明，原训练结果字段保持不变。计时路径回归测试 `tests/test_vllm_engine.py` 为 **67 passed**；验证记录见 [validation.json](../outputs/loop_dspark_prefill_sweep_20261013/validation.json)，实验核对见 [completed_audit.json](../outputs/loop_dspark_prefill_sweep_20261013/completed_audit.json)。
