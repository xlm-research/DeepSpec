# Qwen3.5 Loop/mHC 性能优化：需求澄清

状态：需求澄清中，第一轮 Q1–Q3 待回答。记录日期：2026-10-12。

用户本轮调用 `grill-with-docs`，要求总结此前讨论并明确需求。本文记录事实、候选目标和待确认选择；本轮没有修改模型、训练或传输配置，也没有启动性能优化实验。此前关于分块、eager 和 Mooncake 的提问不代表已经选择某条优化路线。

关联：[领域词汇表](../CONTEXT.md)、[目标推理规格](qwen3_5_loop_mhc_vllm_spec.md)、[DSpark 训练规格](qwen3_5_loop_mhc_dspark_spec.md)、[采集契约 ADR](adr/0002-loop-mhc-dspark-capture-contract.md)、[性能指标口径](performance_metrics.md)。

## 对话收敛过程

1. 用户报告训练日志中的 `wait` 很长，询问是否由 Mooncake 配置导致。
2. 核对日志与控制器后，确认 `wait` 衡量训练前备齐样本的派发等待；目标端供给不足会让草稿训练等待。
3. 用户进一步询问关闭 chunked prefill、关闭 eager 是否能提速，以及本地 vLLM 是否已经优化 Loop-Qwen。
4. 当前实现复用了标准 Qwen3.5 算子，并做了权重共享及部分采集搬运改进；mHC router 仍是未融合的 eager PyTorch 路径。既有交付是功能与结构验收，没有已接受的速度优化验收指标。

## 已核实的运行基线

以用户提供的[日志](../running_logs/qwen3.5-loop-mhc-dspark-3-node_20261012_150414.log)和对应的[运行配置快照](../outputs/qwen3.5-loop-dspark-128k_20261012_150414/config.yaml)为准；示例 YAML 的后续变化不能替代本次运行配置。

| 项目 | 本次运行 |
| --- | --- |
| 任务 | 已有回答监督的 DSpark 草稿训练；目标端做 prefill、隐状态采集和发布 |
| 硬件与拓扑 | 三个八卡 H800 节点；两个独立 TP=8 目标实例，一个八卡 FSDP 训练实例 |
| 长度与并发 | 上下文上限 131072；每个目标实例一次一条请求；分块预算 8192 |
| 执行模式 | V2 runner、eager、BF16 主干、FP32 mHC router/SSM |
| 目标特征 | 五个执行层边界特征，以及独立 post-norm 最终目标状态 |
| 训练批次 | microbatch 1，8 个训练 rank，梯度累积 4，全局 batch 32 |
| Mooncake | TCP；GPU Direct 关闭；生产者发布缓冲池大小 1 |
| 本次数据 | `mydata/sensenova-flash-lite-v42-text-10000.jsonl`；后续实验样本与顺序仍待确定 |

目标端在此路径只使用一个内部生成 token 完成采集流程，不能将本次问题概括为在线回答的 decode 吞吐问题。依据：[目标引擎 generate](../torchspec/inference/engine/vllm_engine.py)。

## 性能证据及边界

- 固定统计原日志第 2–22 步，排除首步后平均 `wait` 为 **46.35 秒**，占这段墙钟时间约 **82.4%**；首步 `wait` 为 400.1 秒，应单独分析启动与首批样本准备开销。
- 第 20 步记录 `dispatch=62.797s`、训练阶段 `step=10.105s`、`data=1.308s`。控制器每次派发 8 条，并在备齐 4 次派发后启动一次训练更新。依据：[训练主循环](../torchspec/controller/loop.py)、[派发控制器](../torchspec/controller/training_controller.py)。
- 上轮对节点 `10.120.0.218` 的 TP0、TP1 各做了 20 秒非阻塞 Python 栈采样。有效样本分别为 428、345；其中 `Qwen3_5MHCRouter.forward` 两行 `logsumexp` 的叶节点样本为 282、272，占 **65.9% / 78.8%**。原始临时产物为 `/tmp/torchspec-wait-20261012-tp0.raw` 和 `/tmp/torchspec-wait-20261012-tp1.raw`，本轮重新读取核对过计数；临时文件不作为长期验收产物。
- 上述比例是 Python 栈样本比例，**不是 GPU 时间占比**；采样仅覆盖一个节点的两个 rank，不能代表完整运行或另一个目标实例。它支持将 mHC 列为优先验证的热点，尚不能独立证明最终主因。
- 目标端供给链路包括前向、采集、发布和调度。训练侧 `data_time` 较低不能排除生产侧 Mooncake 发布阻塞；尚未进行同负载 TCP/RDMA 对照。
- 当前 router 每个完整 chunk 前向、每个 TP rank 有 `24 × 3 × 2 × 20 × 2 = 5760` 次 `logsumexp` 调用。这是源码调用次数推导，不是测得的 CUDA kernel 次数或 GPU 耗时。依据：[Loop/mHC 实现](../vllm/vllm/model_executor/models/qwen3_5_loop.py)。

此前对话中“主要原因就是 mHC 前向”“Mooncake 不是主因”的表述应收紧为：**已观察到训练派发等待严重，目标端供给是排查入口；mHC 是优先验证的热点，计算与传输各自的贡献仍需分阶段测量和对照实验。**

## 配置可行性与候选手段

| 候选手段 | 已知条件与尚未证明的部分 |
| --- | --- |
| 增大 prefill chunk | 8K→16K/32K 是候选实验。减少分块次数及重复的调度、算子启动开销；每个 token 的模型层和 mHC 迭代语义仍须保留。收益和显存峰值需要实测。 |
| 关闭 chunked prefill | 当前实现允许，但必须使 `max_num_batched_tokens >= max_model_len`。保留 128K 上限意味着按 128K 单次预算规划激活和启动 profiling，不能只改布尔开关。 |
| mHC 局部编译或融合 | 直接针对当前热点，是优先评估的候选；不能通过降低循环轮数、Sinkhorn 迭代次数或改变归一化语义冒充实现提速。库内已有其他 mHC 融合算子，但 QwenLoop 尚未接入，计算定义也需要核对。 |
| 整模型非 eager / CUDA Graph | 当前 Loop 配置强制 `enforce_eager=True`。TorchSpec prefill 路径默认设置 `cudagraph_mode=NONE`，该处允许显式覆盖；解除模型限制和验证执行兼容性是独立工作，不能仅删除保护后宣称支持。 |
| 采集、发布与 RDMA | 批量采集和发布、host RDMA 等可独立测量；暂未选定为必须交付项。当前 TCP 开销尚未被因果排除。 |

依据：[Loop 支持范围校验](../vllm/vllm/model_executor/models/config.py)、[引擎构造](../torchspec/inference/engine/vllm_engine.py)、[隐状态 connector](../torchspec/inference/engine/mooncake_hidden_states_connector.py)。关闭 eager 或分块属于候选手段，不能先作为用户目标或必交付结果。

## 拟定需求，待用户确认

在既有三节点 DSpark 训练场景中，提高本地 vLLM 对 Qwen3.5 Loop/mHC 的目标端样本供给速度，降低训练派发等待并提高训练有效吞吐；保留既有模型和监督契约，以可复现的前后对照证明收益。

这是推荐的下一阶段需求表达，并非用户已经确认的实施范围。下一阶段可以研究配置和代码优化，具体路线由测量与本轮问答决定。

## 设计树与当前问题

| 决策分支 | 当前前沿 | 回答后再收敛的事项 |
| --- | --- | --- |
| 目标与范围 | Q1：训练目标端供给，还是独立生成推理 | 允许修改的模块、优先处理的路径、是否包含传输与调度 |
| 语义与兼容 | Q2：是否保留现有模型、128K 和采集监督契约 | 数值对照范围及容差、既有产物兼容、最长输入回归 |
| 验收方式 | Q3：实测收益报告，还是预设硬性加速门槛 | 固定样本、长度分布、预热及测量窗口、GPU 测量和运行资源安排 |

### 第一轮，待回答

**Q1 — 优化目标**：下一阶段是否以当前三节点 DSpark 的目标端供给和训练有效吞吐为目标，允许评估本地 vLLM 代码与配置优化，在线回答的 decode 性能暂不作为验收目标？

推荐：是。以训练 `wait` 和端到端吞吐验证价值，并单独测量目标端瓶颈；不预先指定必须关闭 eager 或分块。

**Q2 — 必须保留的契约**：是否保留原 checkpoint、三轮 Loop、四条 mHC 流、20 轮 Sinkhorn、128K 长度支持、五份辅助特征及独立 post-norm 最终状态，以及已有回答和最后一轮监督语义？

推荐：是。通过实现优化提速；融合或编译导致的浮点差异需要专门验证，具体误差标准下一轮确定。

**Q3 — 性能验收**：是否采用同硬件、同输入和测量口径的前后对照，分别报告冷启动与稳态、目标端供给速率、训练有效吞吐、`wait` 占比及显存峰值，暂不预设未经验证的加速倍数？

推荐：是。先建立可复现基线；若存在硬性倍速或等待占比要求，由用户明确目标值。

## 文档与执行边界

本轮沿用仓库已有 `CONTEXT.md` 作为领域词汇表，补充目标端样本供给、训练派发等待和训练有效吞吐三个术语。当前没有已经选择且难以逆转的架构取舍，因此暂不新增 ADR。问答收敛后更新本文的确认状态，再进入用户确认的实施范围。
