# Research: Ray 管理推理与 draft 训练集群

- **Slug**: torchspec-ray-management
- **Created**: 2026-09-20
- **Evidence confidence (overall)**: high（源码结构）；unknown（同条件性能优劣）。
- **Upstream snapshot**: `b68a7a0fc712523b6407082c9a423299b9ac1406`，提交日期 2026-09-19。
- **Local snapshot**: HEAD `589c8b8ebef766a6f979cc7007e3b8f19f707c4c` 加当前工作区；实际审阅文件 SHA256 见 [source-manifest.json](source-manifest.json)。不能用 HEAD 代替工作区状态。

## Users & Demand

用户明确要求比较两套管理方式并提出修改点；未提供目标节点数、扩容频率、故障恢复 SLA 或吞吐目标。因此“下一步一定需要大规模弹性集群”不是已确认需求。**high / cited**：[intake.md](intake.md)。

## Prior Art — 本地事实

| 编号 | 事实 | 证据与置信度 |
|---|---|---|
| L1 | 单机入口创建私有 Ray，固定 8 GPU / 24 CPU，推理 TP4 与训练 4 GPU 使用不同 PG。已有 Ray 集群地址会进入另一套集群入口。 | **high / cited**：[run.py:426](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/run.py:426)、[run.py:480](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/run.py:480) |
| L2 | 集群入口选两个不同节点，分别承载 producer 和 consumer；强制 `consumer_nodes=1`。consumer_command 虽有多节点 torchrun 参数分支，当前入口没有启用它。 | **high / cited**：[cluster.py:39](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:39)、[cluster.py:432](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:432)、[actors.py:51](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/actors.py:51) |
| L3 | producer 支持 DP1/2，TP 固定 4；DP2 使用原生 AsyncLLM、Ray DP backend 和显式 data_parallel_rank。DP1 由项目预留 worker PG，DP2 由 vLLM 创建自己的 PG。 | **high / cited**：[actors.py:102](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/actors.py:102)、[actors.py:209](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/actors.py:209)、[cluster.py:567](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:567)、[topology.py:4](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/topology.py:4) |
| L4 | 一个 Consumer Ray actor 取得整组 GPU，再启动原生 torchrun/TorchTitan；Ray 没有直接管理每个 draft rank。当前 streaming recipe 是 TP4、FSDP DP1/2、CP=PP=1。 | **high / cited**：[actors.py:268](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/actors.py:268)、[recipe.py:14](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/recipe.py:14)、[trainer.py:24](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/trainer.py:24) |
| L5 | Ledger 同时限制样本数和实际字节数，要求容量能完成一个 optimizer update；只在删除成功后退还容量，读者归属由 DP/TP 计划决定。 | **high / cited**：[buffer.py:12](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/buffer.py:12)、[buffer.py:116](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/buffer.py:116)、[buffer.py:369](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/buffer.py:369) |
| L6 | producer 按固定输入计划准入；消费侧校验 teacher identity、样本 identity 和特征 shape/dtype；默认完整传输校验。各 TP rank 读取独立特征副本，有额外内存和读取成本。 | **high / cited**：[buffer.py:60](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/buffer.py:60)、[data.py:40](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/data.py:40)、[data.py:105](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/data.py:105)、[README.md:130](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/README.md:130) |
| L7 | ACK 表示读取完成、已有独立副本，并非 optimizer 已提交。源对象可在 backward/最终 checkpoint 前删除；本流式入口明确不实现特征重放或故障恢复。 | **high / cited**：[trainer.py:41](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/trainer.py:41)、[README.md:135](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/README.md:135) |
| L8 | 原生 DCP checkpoint 带 commit、更新数和全局微批游标；launcher 重读并核验 commit。但仅此不能恢复内存中的特征生产状态。 | **high / cited**：[checkpoint.py:103](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/torchtitan/torchtitan/models/dspark_draft/checkpoint.py:103)、[cluster.py:681](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:681) |
| L9 | actors/torchrun 禁用自动重启，异常使任务失败；有 namespace、进程归属、节点环境一致性检查、运行 deadline 和显式清理。Mooncake master 由 driver 启动受监护的子进程。 | **high / cited**：[cluster.py:468](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:468)、[cluster.py:654](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:654)、[cluster.py:715](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/cluster.py:715)、[runtime.py:38](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/deepspec/pipeline/runtime.py:38) |

## Prior Art — TorchSpec 事实

| 编号 | 事实 | 证据与置信度 |
|---|---|---|
| T1 | 默认将训练和推理 GPU 合成一个 PG，探测并排序物理落点，再按角色划分；有 custom 和 colocate 分支。默认 PACK 不是“每个 TP 组必定单节点”的保证。 | **high / cited**：[placement_group.py:248](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/ray/placement_group.py#L248)、[placement_group.py:547](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/ray/placement_group.py#L547) |
| T2 | vLLM factory 支持单节点多引擎、跨节点单副本及多副本；Ray actor 封装引擎，vLLM 内部明确使用 mp executor，并用 base GPU ID 构造连续 CUDA_VISIBLE_DEVICES。 | **high / cited**：[factory.py:338](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/inference/factory.py#L338)、[vllm_engine.py:252](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/inference/engine/vllm_engine.py#L252) |
| T3 | RayTrainGroup 为每个训练 rank 创建 actor，world_size 为训练节点数乘每节点卡数；后续调用本项目内置 trainer，不是 TorchTitan launcher。DSpark 分支确实存在。 | **high / cited**：[train_group.py:72](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/ray/train_group.py#L72)、[trainer_actor.py:36](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/training/trainer_actor.py#L36) |
| T4 | 推理管理器主要按 max_sample_pool_size 限流；controller 虽统计 pool bytes，但 dispatch 时即从 pool 扣除，不能将该数字解释为全部在途/驻留特征内存。 | **high / cited**：[inference_manager.py:375](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/inference_manager.py#L375)、[training_controller.py:489](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/training_controller.py#L489) |
| T5 | 训练 dispatch 可在每 rank 多样本时按序列长度做贪心均衡；这与本地固定 position→DP 映射有语义差异。 | **high / cited**：[training_controller.py:512](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/training_controller.py#L512)、本地 L6 |
| T6 | RayActorError 会报告致命推理错误；其他 generate 异常可转成失败结果并跳过，不能据此承诺每个输入都被训练。 | **high / cited**：[inference_manager.py:477](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/inference_manager.py#L477)、[inference_manager.py:512](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/inference_manager.py#L512) |
| T7 | 有模型、优化器、LR、RNG/step checkpoint；但训练 loop 明确将流式 resume 标为 best-effort，异步尾部可能丢失或重放。 | **high / cited**：[checkpoint.py:208](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/training/checkpoint.py#L208)、[loop.py:229](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/loop.py#L229) |
| T8 | 清理推理管理器和 inference future 的 ray.get 未设置 timeout；部分引擎 shutdown 有 timeout。不能把其 cleanup 直接视为本地清理机制的全面升级。 | **high / cited**：[loop.py:128](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/controller/loop.py#L128) |

完整上游核查，包括 offload、Mooncake、训练并行和未验证风险，见 [torchspec-evidence.md](torchspec-evidence.md)。

## Data & Constraints

- **high / cited**：此前本会话 H800 验证记录显示，Qwen3.8-27B target、推理 TP4 + TorchTitan TP4、64 GiB Mooncake pool，在 4K 与 128K 各完成 12 样本、3 次 optimizer update、48 次 rank 读取校验，最终 checkpoint 独立核验通过。[setup-result.json](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/debug_logs/h800_setup_20260920/setup-result.json)
- **high / cited**：128K 修复后运行的 pool 峰值 reserved 为 48,330,965,088 字节，resident 为 32,220,643,392 字节，说明长序列特征的内存账目是实际约束。[128K verification.json](/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm/outputs/h800_setup_20260920_128k_fixed/128k/verification.json)
- **high / cited**：这些是短程正确性验证，不是长期稳定性测试或 TorchSpec 对照实验；本次评估没有启动 TorchSpec GPU 任务。证据范围由上列文件中的 steps/samples 限定。
- **high / cited**：Ray PG 对一个 group 的资源原子预留；PACK 尽量集中、STRICT_PACK 必须同一节点。统一 PG 对部分资源先占后等的行为有影响，但不自动实现业务恢复或运行时伸缩。[Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)、[API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.placement_group.html)
- **high / cited**：TorchSpec 的 PG 预留完整 GPU bundles，内部 engine/trainer actor 使用 fractional GPU；不能将 0.2/0.4 解读为显存硬限制或外部任务必然超卖。设备访问和显存使用仍需应用管理。[placement_group.py:258](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/ray/placement_group.py#L258)、[factory.py:394](https://github.com/lightseekorg/TorchSpec/blob/b68a7a0fc712523b6407082c9a423299b9ac1406/torchspec/inference/factory.py#L394)、[Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)

## Market & Context

本次是仓库架构评估，市场规模与商业需求不适用。现有替代方案是继续使用固定布局完成调试；其优势是已有验证记录，代价是修改节点布局需要调整入口代码（L1/L2）。扩大集群会遇到这些限制，是基于代码的 **medium / cited inference**，不是已发生的生产事故。

## Evidence Against the Idea

这里的“idea”特指“因为 TorchSpec 管理更通用，就整体替换本项目执行与数据链路”。

1. 本地已有长序列内存准入与所有读者 ACK 协议；上游 pool 计数不能等价替代（L5、T4）。**high / cited**。
2. vLLM 原生 Ray DP 与上游多 mp 引擎的 ownership 不同；直接叠加 PG 可能产生重复资源预留。此风险需设计/测试确认，未在本地复现。**medium / cited inference**（L3、T2）。
3. 本地 TorchTitan trainer 和上游 DSpark trainer 不是同一个实现，不能从模型名称推导数值/优化器/并行等价。**high / cited**（L4、T3）。
4. 两者都没有已经证明的透明流式故障恢复；上游明确允许 resume 尾部变化，本地明确 fail-fast（L7、T7）。**high / cited**。
5. 上游连续 GPU ID 假设在碎片化资源上存在需验证的风险；不能将默认 PACK 与排序解释为硬件拓扑验证。**medium / cited inference**（T1、T2）。
6. 本次没有同条件吞吐/显存/恢复时间对照数据；不能写“切换后更快/更省显存”。**unknown / assumption not established**。

## Gaps & Open Questions

- 下一阶段目标节点数、每角色卡数及是否需要不同 GPU 型号分配，尚未指定；不阻碍判断当前硬编码限制。
- 原生 vLLM DP backend 接受外部已创建 PG 的方式，需以本仓库 vLLM 版本进一步验证，不能预设支持。
- 128K 长跑、跨机多 consumer、故障重放的一致性和性能均需后续专项验证。
- TCP/CPU pool 与 RDMA/GPU 接收路径的端到端收益未比较；本次不据源码宣称 RDMA 性能已验证。

## Source Handling

上游源码 URL 均固定于同一提交，host 为 `github.com`（allowlisted / 用户指定仓库）；Ray 官方文档 host 为 `docs.ray.io`（官方语义核查，补充来源）。引用不含凭据。外部仓库未执行，只读检查；本地源码没有在本次评估中修改。
