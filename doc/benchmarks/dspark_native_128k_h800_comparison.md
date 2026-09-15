# Qwen 128K：H800 连续训练与分阶段恢复对照报告

日期：2026-09-15，时区 Asia/Shanghai。硬件：单机 8 × NVIDIA H800 80GB。

## 结论

**本轮固定工作负载的连续训练与分阶段恢复一致性验收通过。**

- 最终 checkpoint 的 **754 个状态字段、344 个张量精确一致**，差异为 0，覆盖模型、优化器、调度器、训练状态和全局数据进度。
- 8 个 rank、每个 rank 20 次前向的监督数据、顺序和 CPU/CUDA 随机数状态精确一致。
- rank 0 的 25 项训练指标 × 10 步，共 250 对保存值精确一致；最终平均 loss 为 **3.704998**。
- 两轮均完成 10 次更新并释放所有训练 worker；连续训练的八卡指标未记录 OOM。

本次对照使用相同初始权重、随机数状态、教师特征和输入计划，比较以下两种执行方式：

| 实验 | 执行方式 | 完成情况 |
| --- | --- | --- |
| 分阶段训练 | 训练 1–5 步 → 保存、退出并释放 GPU → 新进程恢复 → 训练 6–10 步 | 10 步完成，最后一条训练完成日志为 15:36:41 |
| 连续训练 | 从相同初始状态出发，在同一组 worker 中连续训练 1–10 步 | 10 步完成，16:37:19 完成最后一次更新，16:37:44 完成退出及结果写入 |

连续训练实际训练进程运行 **35 分 32.553 秒**；计入启动前的特征校验、准备及退出后的资源检查后为 **58 分 49.274 秒**。

## 1. 固定工作负载

| 项目 | 本次实际配置 |
| --- | --- |
| 训练对象 | Qwen3.8-27B 教师对应的 DSpark 草稿模型 |
| Draft 规模 | 5 层，hidden 5120，FFN 17408，词表 248320 |
| Attention | 24 个 Q heads、4 个 KV heads，head dimension 256 |
| DSpark | 512 anchors，block size 7，Markov rank 256，启用 confidence head |
| 数据 | 40 条计划内样本，每条 131072 tokens，总计 5,242,880 个上下文 token |
| 并行 | TP4 × DP shard2；CP1、PP1；8 个训练 worker |
| Batch | 每个 DP rank 的 microbatch 为 1；global batch 4；梯度累积 2 次 |
| 更新 | 10 次 optimizer update，每次处理 524288 个上下文 token |
| 精度 | BF16 模型参数，FP32 reduction、master weights 和 Adam 状态 |
| 训练设置 | SelectiveAC；关闭 SP 和外层 model compile；编译线程数 32 |
| 优化器和调度 | AdamW，配置 LR 6e-4，weight decay 0，clip norm 1；1000 步调度周期、40 步 warmup |
| 保存 | 阶段边界同步写入完整 DCP；本次关闭 HF 导出 |

两轮 `training_identity` 相同；输入计划 SHA256 均为
`efd6377dd1b18e4f2dadf32161d4ce151ee59933068609434642cd7275ae1987`。
连续训练读取首次训练保存的 `scale-initialization-h800/initial-weights.pt` 和 8 份 rank RNG 状态，复用两个原始分区的实际教师特征。
连续训练额外开启全部 rank 的 TensorBoard 记录；该设置不属于训练状态身份。

配置与输入约束、8 个 worker 的完整提交、checkpoint 元数据摘要及各阶段 GPU 释放记录，均由现有汇总脚本重新核验。
证据：[分阶段汇总](../../outputs/dspark_torchtitan_orchestration_20260914/h800-phased-final-summary.json)、[连续训练汇总](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-summary.json)。

## 2. 一致性核验

### 最终 checkpoint

**通过：754 个字段、344 个张量全部精确一致，`differences=[]`。** 核验进程正常退出，未初始化 CUDA。

| 状态类别 | 完整 checkpoint 中的字段数 | 结果 |
| --- | --- | --- |
| 模型参数 | 64，均为 BF16 张量 | 精确一致 |
| 优化器状态及参数组 | 620，含 248 个 FP32 张量 | 精确一致 |
| 训练状态 | 52，含各 rank 保存的随机数状态等 | 精确一致 |
| 学习率调度器 | 17 | 精确一致 |
| Dataloader | 共 3 项；比较全局位置，排除以下 2 个局部字段 | 全局位置均为 20 |

比较对象是两轮各自的 `checkpoints/step-10`，使用 CPU 加载完整 DCP。
比较方法要求张量 dtype 相同且 `torch.equal` 为真，非张量状态递归精确比较，不设置浮点容差。

仅排除两个分区局部字段：`dataloader.feature_identity` 和 `dataloader.cursor`。
它们分别描述本阶段的特征清单和局部游标，连续运行与最后一个分区的表示不同；全局消费位置 `dataloader.next_global_microbatch` 仍必须相同。
各阶段输入计划、完整 update 分组、实际前向次数和消费顺序另行核验，因此排除局部字段不等于跳过数据进度检查。

本次局部游标分别为 10（最后一个 5-step 分区）和 20（完整 10-step 连续运行），与各自分区长度一致。
证据：[完整 checkpoint 比较 JSON](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-checkpoint-comparison.json)、[核验日志](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-checkpoint-comparison.log)、[核验脚本](../../tests/compare_torchtitan_checkpoints.py)。

### 每次前向的监督与随机数状态

**通过精确比较。** 两轮均为 8 个 rank，每个 rank 20 次前向，共 160 对前向记录。
比较内容包括 `target_ids`、`eval_mask`、`block_keep_mask`、CPU RNG、CUDA RNG 和归一化后的全局前向顺序。

原生 Trainer 会先预取一个完整梯度累积窗口，因此实际 hook 中的读取游标为 `2,2,4,4,…,20,20`。
核验脚本先检查各分区的前向次数和每个预取游标，再赋予连续的已执行前向位置；原始监督张量和 RNG 数据均未改写。
证据：[连续训练汇总中的 supervision_bitwise_equal](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-summary.json)。

### 逐步训练指标

**rank 0 的 25 项 TensorBoard 指标 × 10 步，共 250 对数值精确一致，差异为 0。**
覆盖平均/最大 loss、梯度范数、学习率、token 计数以及记录的训练监督与 loss 分项等指标。
比较使用 TensorBoard 中保存的数值；下表仅为便于阅读而四舍五入。

| Step | 两轮平均 loss | 两轮梯度范数 | 两轮记录的 LR | 分阶段单步秒数 | 连续单步秒数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.735296 | 0.722656 | 0.000015 | 215.051 | 211.193 |
| 2 | 3.734601 | 0.718750 | 0.000030 | 198.938 | 206.371 |
| 3 | 3.733957 | 0.726562 | 0.000045 | 200.093 | 204.675 |
| 4 | 3.732739 | 0.734375 | 0.000060 | 202.191 | 204.704 |
| 5 | 3.730846 | 0.753906 | 0.000075 | 201.919 | 203.790 |
| 6 | 3.728036 | 0.800781 | 0.000090 | 208.302 | 193.903 |
| 7 | 3.724183 | 0.890625 | 0.000105 | 201.908 | 200.633 |
| 8 | 3.718644 | 0.988281 | 0.000120 | 208.057 | 198.356 |
| 9 | 3.712725 | 1.070312 | 0.000135 | 203.568 | 200.372 |
| 10 | 3.704998 | 1.148438 | 0.000150 | 204.176 | 199.136 |

记录的 LR 连续增长到第 10 步，恢复后的第 6 步延续原 warmup 进度。
证据：[指标与统计核验 JSON](../../outputs/dspark_torchtitan_orchestration_20260914/h800-report-audit.json)。

## 3. 耗时与吞吐

| 统计项目 | 分阶段 5+5 | 连续 10 步 |
| --- | --- | --- |
| 特征校验与准备 | 1341.556 s | 1396.581 s |
| Worker 生命周期，启动至退出 | 2219.869 s（37 分钟） | 2132.553 s（35 分 33 秒） |
| 其中：10 次训练更新累计 | 2044.203 s | 2023.132 s |
| GPU 释放检查 | 0.275 s | 0.140 s |
| 完整 draft 累计成本 | **3561.700 s（59 分 22 秒）** | **3529.274 s（58 分 49 秒）** |
| 共同稳态步平均耗时 | 202.606 s | 202.255 s |
| 共同稳态步标准差 | 2.598 s | 2.790 s |
| 共同稳态上下文吞吐，全机 | 2587.72 tokens/s | 2592.22 tokens/s |

统计口径：

- 单步耗时覆盖该步最早 rank 开始到最晚 rank 结束；包含实际训练、通信及该区间内的输入读取，不把并行 rank 的耗时相加。
- 公平比较使用两轮共同的第 **2、3、4、5、7、8、9、10** 步，均排除第 1 步和分阶段恢复后的第 6 步。连续训练原生汇总另将第 2–10 步视为稳态，均值 201.327 s；两种步集合不同。
- 吞吐按输入上下文位置计数；DSpark 使用 anchor 监督，该数值不表示逐 token 生成吞吐或有效监督 token 吞吐。
- 完整 draft 成本包含特征校验、准备、模型启动/恢复、训练、保存、退出和资源检查；其中训练项嵌套于 worker 生命周期，不重复相加。
- 分阶段数字只累计最终成功提交的两个阶段。**此前失败尝试、诊断 probe、人工等待和教师特征生产均未纳入本表**，因此该数字不代表从首次提交任务到最终完成的总历时。

本次分阶段 worker 生命周期比连续运行增加 **87.316 s（4.09%）**；包含准备后的完整 draft 成本增加 **32.426 s（0.92%）**。
共同稳态步均值差 **0.352 s（0.17%）**。
这是每种方式各一次运行的观测：分阶段首轮包含初始化快照捕获，连续运行包含快照读取，全 rank 指标设置也不同，不能把全部时间差解释为保存/恢复的固定开销，也不能据此推断稳定性能提升。

补充边界计时：分阶段两次 checkpoint 保存分别为 14.601 s、13.489 s；第二阶段恢复为 42.854 s；连续运行最终保存为 14.395 s。
这些来自每个阶段耗时最长 rank 的实际时间线，已包含在阶段总时间中。

## 4. 显存与资源释放

| 观测 | 分阶段 5+5 | 连续 10 步 |
| --- | --- | --- |
| 逐步 TensorBoard 显存覆盖 | rank 0，共 10 步 | 全部 8 个 rank，各 10 步 |
| 记录中的 active 峰值 | 61.630 GiB | 61.627 GiB |
| 记录中的 reserved 峰值 | 74.234 GiB | 73.271 GiB |
| 记录中的 OOM 计数 | 成功阶段 rank 0 为 0 | 8 个 rank 均为 0 |
| allocator retry | 第一阶段 0；第二阶段 rank 0 累计至 3 | 8 个 rank 均为 0 |
| 阶段结束后 worker 退出及 GPU 释放 | 两个成功阶段均通过 | 通过 |

这里使用每次指标重置前的训练区间峰值。阶段退出时的 CUDA peak counters 已被训练指标重置，不能用它们推断整轮峰值。
分阶段只记录了 rank 0 的详细显存指标，不能把该列视为八卡最大值；active、reserved 和 `nvidia-smi` 的进程总占用含义也不同。
生成报告时于 17:02 再次查询，8 张 GPU 均为 0% 利用率、0 MiB 显存占用，见[报告复核记录](../../outputs/dspark_torchtitan_orchestration_20260914/h800-report-verification.json)。

## 5. 运行中发生过的故障

分阶段训练首次从 step 5 恢复时，在第 6 步梯度范数归约处发生过 NCCL CUDA OOM。
已有诊断显示，首次参数通信所需的 NCCL 资源遇到 PyTorch 大量缓存显存占用；修复在模型初始化时提前建立相关参数通信。
保持原 step-5 checkpoint 和 128K 输入的 probe 成功后，原任务恢复并完成 step 6–10。

本报告的分阶段结果保留了这条实际恢复路径：step 1–5 在修复前完成，step 6–10 和连续基线使用修复后的路径。
证据：[诊断](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resume-oom-diagnosis.json)、[恢复 probe 验收](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resume-warmup-acceptance.json)、[历史过程记录](dspark_native_128k_h800.md)。

## 6. 结论适用范围

本报告针对单机八卡、固定 TP4 × DP shard2 拓扑、同一份真实 128K 特征和 10 次更新的连续/恢复对照。
10 步处于 warmup 期间，不能据此判断收敛质量或真实推理的草稿接受率；梯度证据为保存的梯度范数，未另存每一步的完整梯度张量。
后续完成的常驻旧实现性能对照见[常驻参考报告](dspark_native_128k_h800_resident.md)。本报告未包含多节点、跨拓扑恢复，以及后续HF导出、保留策略和故障注入验收。因此本报告不宣告整个迁移项目或全部验收项完成。

## 7. 证据与复核

所有原始训练文件保留在 `outputs/dspark_torchtitan_orchestration_20260914/`：

- [分阶段完成记录](../../outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800/complete.json)、[分阶段日志](../../outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800.log)。
- [连续训练完成记录](../../outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800-continuous/complete.json)、[连续训练日志](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous.log)。
- [分阶段最新汇总](../../outputs/dspark_torchtitan_orchestration_20260914/h800-phased-final-summary.json)、[连续与监督对照汇总](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-summary.json)。
- [最终 checkpoint 全量比较](../../outputs/dspark_torchtitan_orchestration_20260914/h800-continuous-checkpoint-comparison.json)。
- [指标与性能核验结果](../../outputs/dspark_torchtitan_orchestration_20260914/h800-report-audit.json)、[可复现的 CPU 核验脚本](../../outputs/dspark_torchtitan_orchestration_20260914/h800-report-audit.py)。

在仓库根目录执行以下命令可重新核验已有产物；这些命令不启动训练。完整 DCP 比较会读取两份各约 31 GB 的 checkpoint。

```bash
REPORT_ROOT=outputs/dspark_torchtitan_orchestration_20260914
REPORT_PYTHON=.envs/orchestration/bin/python

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$REPORT_PYTHON" -m tests.compare_torchtitan_checkpoints \
  "$REPORT_ROOT/qwen38-128k-h800/checkpoints/step-10" \
  "$REPORT_ROOT/qwen38-128k-h800-continuous/checkpoints/step-10" \
  "$REPORT_ROOT/h800-continuous-checkpoint-comparison.json"

"$REPORT_PYTHON" -m tests.summarize_torchtitan_scale \
  "$REPORT_ROOT/qwen38-128k-h800" "$REPORT_ROOT/h800-phased-final-summary.json"

"$REPORT_PYTHON" -m tests.summarize_torchtitan_scale \
  "$REPORT_ROOT/qwen38-128k-h800-continuous" "$REPORT_ROOT/h800-continuous-summary.json" \
  --reference "$REPORT_ROOT/qwen38-128k-h800"

"$REPORT_PYTHON" "$REPORT_ROOT/h800-report-audit.py"
```
