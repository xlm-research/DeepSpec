# DSpark target-first 数据批训练说明

## 结论

DeepSeek-V4-Flash DSpark 在线蒸馏现在支持按比例分块的 `data_batch_size`。这里的 size 表示“总共分成几块”，不是“一块有多少条”。训练不再对每个 micro-batch 交替执行一次 target 和一次 draft，而是：

1. 连续为整个 data batch 执行冻结 target 推理；
2. 将这一块的 KV/隐藏层监督写入临时磁盘目录（当前代码实际落盘的是 selected hidden state 和 final hidden state）；
3. target 阶段结束后，从磁盘逐条回读并执行这个 data batch 内的所有 draft forward/backward 和 optimizer step；
4. 所有 rank 完成这一块的 draft 训练后，删除整块临时缓存；
5. draft 参数、梯度以及 FP32 optimizer state 始终保留在 GPU，进入下一个 data batch 时不会卸载。

这里是严格的全局串行，不只是 rank 0 的日志顺序：

```text
块 1 target-only 推理
  -> CUDA synchronize + all-rank barrier
块 1 draft-only 训练
  -> CUDA synchronize + all-rank barrier
删除块 1 缓存 + all-rank barrier
  -> 块 2 target-only 推理
  -> ...
```

draft 阶段只能读取已经落盘的 target 特征。若缓存内容缺少 `target_hidden_states`，训练器会直接报错，不允许从 `run_batch` 内回退调用 target，因此不会重新变成“推理一条、训练一条”的交替模式。

这里隔离的是计算阶段：target forward 与 draft forward/backward 不并发、不交替。模型驻留策略保持不变，draft 参数、梯度和 optimizer state 不会在 target 阶段被卸载。

启动脚本的默认训练数据已经改为：

```text
/mnt/afs-agentpro/hongjiawei/code/DeepSpec-old/scripts/data/sensenova_flash_v15_use_data_json.thinking_toolcall.readable_full.flash_conversations.jsonl
```

## 参数关系

- `GLOBAL_BATCH_SIZE`：一次 optimizer step 消耗的全局样本数。
- `DATA_BATCH_SIZE`：把本次计划训练的全部样本分成多少块。
- `DATA_BATCH_SIZE=3` 表示三等份；如果总量为 15,000 条，每块就是 5,000 条。
- 块边界必须落在完整 optimizer step 上，不能把一个梯度累积窗口拆到两个磁盘块中。

训练器先计算剩余 optimizer step 数，再做近似等分。不能整除时，前面的块各多一个 optimizer step。例如：

```text
total_steps = planned_samples / GLOBAL_BATCH_SIZE
base_steps, remainder = divmod(total_steps, DATA_BATCH_SIZE)
block[i] = (base_steps + (i < remainder)) * GLOBAL_BATCH_SIZE
```

数据集尾部不足一个 `GLOBAL_BATCH_SIZE` 的记录不会进入本轮训练，这是原有全局 batch 对齐规则，不是分块额外丢数据。

| 计划样本数 | GLOBAL_BATCH_SIZE | DATA_BATCH_SIZE | 每块全局样本数 |
|---:|---:|---:|---:|
| 15,000 | 8 | 3 | 5,000 / 5,000 / 5,000 |
| 64 | 8 | 3 | 24 / 24 / 16 |
| 1,000 | 8 | 3 | 336 / 336 / 328 |

`DATA_BATCH_SIZE=3` 是当前默认值。增大它会得到更多、更小的块，降低临时磁盘峰值，但 target/draft 阶段切换更频繁。磁盘上只保留当前块；draft 训练时每次只把一条缓存传回 GPU，整块消费完成后立即删除。

## 启动方法

### 单机 8 卡验证

下面的命令使用真实 JSONL，总共处理 64 条数据并分成三块：24、24、16 条，对应 3、3、2 次 draft optimizer step：

```bash
cd /mnt/afs-agentpro/lezewei/DeepSpec

OUTPUT_ROOT=/mnt/afs-agentpro/lezewei/DeepSpec/output/dspark_partitions3_8gpu \
DATA_BATCH_CACHE_DIR=/mnt/afs-agentpro/lezewei/DeepSpec/output/dspark_partitions3_8gpu/target_data_batch_cache \
GLOBAL_BATCH_SIZE=8 \
DATA_BATCH_SIZE=3 \
MAX_LENGTH=32768 \
MAX_TRAIN_STEPS=8 \
SAVE_CHECKPOINTS=false \
PRODUCTION_RUN=false \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29684 \
bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

### 正式训练

正式训练不要设置 `MAX_TRAIN_STEPS`，并启用 checkpoint 保护：

```bash
DATA_BATCH_SIZE=3 \
PRODUCTION_RUN=true \
SAVE_CHECKPOINTS=true \
bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

多机环境继续由调度器注入 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。脚本会在启动前检查 world size、并行拓扑、batch 整除关系、模型路径和数据路径。

## 成功标志

日志中应按以下顺序出现：

```text
Data batch partitions = 3
Optimizer steps per data batch = (3, 3, 2)
Global samples per data batch = (24, 24, 16)
Data batch 1/3: starting isolated target inference; draft training is idle.
Target data batch 1 ready: global_samples=24; all ranks finished inference and cached it on disk; starting isolated draft training.
Draft GPU residency verified: ... device=cuda.
epoch=1 step=1/8 loss=...
...
epoch=1 step=3/8 loss=...
Data batch 1 draft training finished; deleted its transient disk cache before the next target inference phase.
Data batch 2/3: starting isolated target inference; draft training is idle.
Target data batch 2 ready: global_samples=24; all ranks finished inference and cached it on disk; starting isolated draft training.
...
Data batch 3/3: starting isolated target inference; draft training is idle.
Target data batch 3 ready: global_samples=16; all ranks finished inference and cached it on disk; starting isolated draft training.
epoch=1 step=8/8 loss=...
Data batch 3 draft training finished; deleted its transient disk cache before the next target inference phase.
[deepspec-launch-exit] ... exit_code=0
```

块内 loss 指标仍使用异步归约，但训练器会在块边界同步刷完最后一条 step 日志；因此下一块的 target start 必然出现在上一块的 step 和 delete 日志之后。

如果只看到 `Running training`，暂时没有 step loss，不代表进程停止。data batch 模式会等完整 target 窗口写盘完毕后才开始 draft 更新；可通过 `nvidia-smi`、`DATA_BATCH_CACHE_DIR` 的容量变化、worker CPU 使用率以及 per-rank 日志判断是否仍在处理超长样本。

## 实机验证

2026-08-27 使用单节点 8 张 NVIDIA B300 对“总数据分块数”语义和磁盘生命周期完成第一轮真实链路验证：

- PyTorch `2.11.0+cu130`，CP=8、FSDP=8、target EP=8；
- `GLOBAL_BATCH_SIZE=8`、`DATA_BATCH_SIZE=3`、`MAX_TRAIN_STEPS=8`、`MAX_LENGTH=32768`；
- 数据集识别为 713,144 条记录；
- 调度结果严格为三块 `(24, 24, 16)` 条，对应 optimizer step `(3, 3, 2)`；
- 第一块由 8 个 rank 各写 24 个样本文件，连同 8 个所有权标记共 200 个文件，观察到的磁盘峰值约 8.3 GiB；第二块观察到约 5.9 GiB；
- 每一块都先完成 target `ready`，再完成 draft 更新和删除；下一块生成前缓存回落到 68 KiB/8 个标记文件；
- 连续完成 8 次 optimizer step，loss 为 `2.4103, 3.8143, 2.2950, 2.4817, 0.6727, 2.5413, 1.1743, 1.7010`；
- 运行时验证了 78 个参数张量、76 个梯度张量和 304 个 optimizer-state 张量全部位于 CUDA；
- 进程正常退出后缓存目录为 0 个文件、0 个子目录、4 KiB 空目录；
- torchrun 最终 `exit_code=0`。

验证日志位于：

```text
output/dspark_partitions3_8gpu_disk_smoke_20260827/
```

在加入严格全 rank 阶段 barrier 和禁止 inline target 的断言后，又执行了一次 3 块隔离验证：

- `DATA_BATCH_SIZE=3`、`MAX_TRAIN_STEPS=3`，三块全局样本数为 `(8, 8, 8)`；
- 三次日志均严格按 `target start -> all ranks inference ready -> draft step -> delete -> next target start` 排列；
- loss 为 `2.4098, 3.8226, 2.2988`，torchrun `exit_code=0`；
- 正常退出后缓存目录为 0 个文件、0 个子目录、4 KiB。

严格隔离验证日志位于：

```text
output/dspark_isolated_partitions3_8gpu_smoke_20260827/
```

## 已解决的问题

1. 原执行顺序是 `target(micro-batch) -> draft(micro-batch)`，无法按总数据比例控制离线特征的磁盘峰值。现在改为严格串行的滚动临时磁盘块：把计划训练数据分成 `DATA_BATCH_SIZE` 份，完整推理当前块，全 rank 同步后只训练 draft，删盘并再次同步后才生成下一块。
2. 新 JSONL 第一次读取需要建立 713,144 行的偏移索引。现在由全局 rank 0 建立共享缓存，其他 rank 直接读取，避免每个 rank 同时扫描约 33.7 GB 文件。
3. 第一次实机启动因配置系统不认识 `data.jsonl_index_cache_dir` 而失败。该字段已经加入基础配置，命令行覆盖可以正常解析。
4. 训练首步加入 draft GPU 驻留断言；如果参数、梯度或 optimizer state 被错误卸载，会立即抛出错误，而不是静默降低后续更新性能。
5. target/draft 转换点加入 CUDA 同步和全 rank barrier；draft 路径缺少预计算特征时直接失败，不能隐式调用 target。块末还会同步刷完 loss 日志，便于从日志直接审计阶段隔离。

## 注意事项

- 该 JSONL 中存在非常长的 system/user prompt。小 `MAX_LENGTH` 可能在 assistant token 之前截断样本，使 `loss_mask` 为空。实机验证发现 `MAX_LENGTH=1024` 不适合作为该数据集的完整 smoke test；`MAX_LENGTH=32768` 的前 64 条样本均有监督 token。
- 正式 128K 训练时，当前块的磁盘占用远高于 32K smoke test。磁盘紧张时应增大 `DATA_BATCH_SIZE`，因为它表示分块数；例如从 3 改成 6，单块数据量约减半。
- 正常完成一块时日志会明确打印缓存已删除。若进程在块中途失败，该块会保留用于排障；同一输出目录再次启动时，只会清理带 DeepSpec 所有权标记的 rank 临时目录，不会递归删除未标记目录。
- `DRY_RUN=true` 只打印 torchrun 命令，不会启动训练。正式运行建议同时设置 `PRODUCTION_RUN=true`，以防残留的 step 限制或关闭 checkpoint 导致训练“正常但过早”退出。
- 查看原始错误时优先检查 `OUTPUT_ROOT/logs/torchrun_node_rank_*/.../<rank>/stderr.log`，不要只依赖聚合 node log。
