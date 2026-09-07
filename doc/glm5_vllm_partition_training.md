# GLM-5.3：按训练分区使用离线 vLLM

在原有 `partitioned_model_swap` 流程中选择 `TARGET_BACKEND=vllm`，每次只生成当前分区的教师特征。启动入口仍是 `scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh`。

```bash
TARGET_BACKEND=vllm \
PYTHON_BIN="$PWD/vllm/.venv/bin/python" \
PARTITIONED_MODEL_SWAP=true \
DATA_BATCH_SIZE=8 \
OUTPUT_ROOT="$PWD/output/glm5_vllm_partitions" \
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

当前环境可使用包含教师和 draft 的快捷入口：

```bash
TRAIN_DATA_PATH=/path/to/your/full_dataset.jsonl \
bash scripts/fsdp/fast_vllm_glm-5.3-flash.sh
```

快捷入口默认使用仓库的 `vllm/.venv/bin/python`、128K 上下文和完整数据集的 8 个分区，沿用主启动脚本的多节点拓扑与梯度累积。权重预缓存保持开启；权重缓存、原始特征、分区特征、checkpoint 和日志默认位于 `output/glm5_3_flash_dspark_fsdp2/` 下。未指定 `TRAIN_DATA_PATH` 时沿用主脚本的测试数据选择。

当前环境的 `vllm/` 子仓库已包含 GLM mHC 特征导出和 hidden-state KV cache 修复。其他环境需要安装包含这些实现的 vLLM。可用 `VLLM_PYTHON_BIN` 指定推理解释器，用 `VLLM_SOURCE_DIR` 指定 vLLM 源码目录；默认优先使用本仓库的 `vllm/`，否则使用解释器中安装的包。训练解释器仍由 `PYTHON_BIN` 指定。

## 分区与训练语义

1. 分区对象是 `TRAIN_DATA_PATH` 指定的完整数据集。默认 `DATA_BATCH_SIZE=8`，按数据集对应的完整 global batches 尽量均匀地划为 8 区，并保留原 sampler 顺序和梯度累积边界。每个 epoch 复用这组分区边界；`MAX_TRAIN_STEPS` 只截断执行位置，不决定如何划分数据集。数据不足 8 个 optimizer steps 时只生成非空分区。
2. 每个训练 rank 将当前分区的 token、loss mask、dataset index、逻辑样本编号和训练位置写入 CPU 请求文件。tokenization、chat template 和监督 mask 都由原来的 collator 处理。
3. 每个节点按连续 4 张卡启动独立 vLLM TP4 进程。因此每节点 8 卡对应两个推理副本。副本为这 4 个训练 rank 生成各自缓存，不改变训练时的 DP/HSDP/EP 布局。
4. vLLM 运行完整 45 层教师，提取配置指定的 decoder 输出，默认 `[2, 22, 42]`。额外提取第 45 层的 pre-norm 状态，再应用 checkpoint 中的 final RMSNorm，得到 L1/confidence 监督所需的最终状态。
5. 完整训练序列作为 prompt。用于结束导出的额外 sampled token 不写入训练样本。逐项验证 token、形状、BF16 类型和有限值，然后写入原来的 `.pt` 分区格式。
6. 所有 vLLM 进程及其子进程退出，所有 rank 的特征验证并提交 READY，才加载 draft 和优化器。训练完成后沿用原来的原子 checkpoint 提交，再删除本分区缓存并进入下一分区。

启动脚本默认使用 `TARGET_BACKEND=vllm`、`DATA_BATCH_SIZE=8`，将 `train.data_batch_size=8` 和 `train.partitioned_model_swap.max_samples=null` 传入训练器。`train.data_partitions` 不设置。

显式设置 `PARTITION_MAX_SAMPLES` 时改用每区全局样本数上限，并将 `train.data_batch_size` 设为 null；上限向下对齐到完整 global batch，最后一个分区可以更小。需要 native 后端时设置 `TARGET_BACKEND=native`；视觉调试入口默认选择 native。

## 显存与缓存

训练进程在生成阶段保留分布式上下文，不持有 draft、优化器或 native teacher。vLLM 使用独立进程；等待推理完成的控制通信使用 Gloo，避免在这些 GPU 上挂起训练 NCCL 操作。

当前每个 vLLM 副本一次处理一个请求，使用 chunked prefill，并关闭 prefix caching。原始导出文件每个请求转换后即删除；完整的分区特征保存在 `DATA_BATCH_CACHE_DIR`。128K、三个 4096 维中间层加最终层的 BF16 特征，每条约 4 GiB，分区大小应按磁盘容量设置。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DATA_BATCH_SIZE` | `8` | 完整数据集的分区数，各 epoch 复用边界；`auto` 表示每个 step 一区 |
| `PARTITION_MAX_SAMPLES` | `null` | 显式设置后改用每区全局样本数上限 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | 每次 prefill 的 token 预算 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.8` | 每张推理 GPU 的显存预算 |
| `VLLM_LOAD_FORMAT` | `instanttensor` | 权重加载方式 |
| `VLLM_RAW_CACHE_DIR` | 主入口：`/tmp/deepspec-vllm-raw`；快捷入口：`OUTPUT_ROOT/vllm_raw_cache` | 单条请求的临时导出位置；支持在 AFS 锁竞争时等待后重试。 |
| `VLLM_TIMEOUT_SECONDS` | `86400` | 单个分区推理进程的超时 |

AFS 在锁被占用时可能对阻塞 `flock` 返回 `EAGAIN`。特征读取会保留锁同步并等待重试，直到写入完成或达到推理超时；其他读取错误直接报错。共享权重预缓存同样等待竞争锁，`TARGET_MODEL_CACHE_LOCK_TIMEOUT_SECONDS` 默认为 86400 秒。将缓存放在 AFS 不代表能复现历史本地 NVMe 缓存的加载速度；多节点实际性能仍需实测。

首次接入建议用两个小分区验证，保留完整模型，只缩短输入长度和训练步数：

```bash
TARGET_BACKEND=vllm \
PYTHON_BIN="$PWD/vllm/.venv/bin/python" \
GLOBAL_BATCH_SIZE=8 PARTITION_MAX_SAMPLES=8 MAX_TRAIN_STEPS=2 \
MAX_LENGTH=2048 NUM_ANCHORS=2 VLLM_MAX_NUM_BATCHED_TOKENS=1152 \
OUTPUT_ROOT=/tmp/glm5_vllm_two_partitions \
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

## 恢复与边界

使用相同配置和 `OUTPUT_ROOT` 重启即可恢复。journal 记录数据、模型、训练配置和教师特征契约；vLLM 版本、导出代码、checkpoint 配置与权重文件信息均参与身份检查。更换教师后端、权重或导出实现时，应使用新的输出目录。

- GENERATING 阶段失败：保留已提交 READY 的 rank，仅重新生成未完成的 rank。
- TRAINING 阶段失败：从分区起点的完整 checkpoint 恢复模型、优化器、scheduler 和 RNG，复用 READY 特征，重跑本分区。
- checkpoint 已完整提交：按现有 journal 恢复规则跳过重复训练，再完成缓存清理。

EP 专家参数在训练时仍是各 rank 的本地张量。保存 DCP 时将其描述为带全局形状和分片位置的 DTensor，避免把不同专家及其优化器状态误判为副本；HF 导出会收集完整专家权重。随机数状态按 rank 保存。旧实现生成的、缺失专家分片的 EP checkpoint 会被拒绝恢复，应使用新的输出目录重新开始。

当前支持文本输入、draft CP=TP=1，以及每节点 GPU 数为 4 的倍数。多节点的 checkpoint/output 根目录需要共享存储；特征缓存可放在各节点本地，但同一节点的训练 rank 和 vLLM 进程必须可互相读取。

当前 checkpoint 使用 FP8 教师计算、BF16 特征输出。它与原来反量化 BF16 教师的数值有差异。链路正确性和断点连续性可以用测试验证；是否提高草稿接受率仍需在同一验证集上比较训练结果。

## 验证记录

2026-09-07，在当前环境完成以下回归检查：

```bash
vllm/.venv/bin/python -m pytest -q \
  tests/test_glm5_partitioned_model_swap.py \
  tests/test_glm5_multinode_launcher.py \
  tests/test_checkpoint_roundtrip.py

CUDA_VISIBLE_DEVICES=0,1 vllm/.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m pytest -q tests/test_checkpoint_roundtrip.py
```

单进程检查为 52 passed、3 skipped；双 GPU 检查每个 rank 为 3 passed、3 skipped，覆盖 FSDP2 与本地 EP 参数混合时的保存、恢复和完整 HF 导出。单进程测试还验证了含 dropout 的模型在分区恢复后，下一步 loss、参数和学习率与连续训练完全一致。相同 EP 测试在原 checkpoint 实现上会失败。

分区测试包含 token/层顺序/监督 mask 校验、相同 prompt 的独立样本归属、部分 rank 已 READY 时的恢复，以及子进程成功或失败后的整组进程清理。

同日，使用 8 张 B300、完整 45 层教师和完整 3 层 draft，完成了两个分区的真实训练。数据为当前 `spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl` 的 8 条样本，连续运行两个 epoch；使用上面的 2K 长度、每分区 8 条样本测试配置。

| 分区 | 逻辑样本编号 | token 总数 | optimizer step | loss |
| --- | --- | --- | --- | --- |
| 0 | 0–7 | 9,174 | 1 | 4.0779 |
| 1 | 8–15 | 9,174 | 2 | 3.2317 |

独立读取了两轮全部缓存文件，确认 token、loss mask 与请求逐项一致，特征为有限值 BF16，形状分别是 `[1, T, 12288]` 和 `[1, T, 4096]`。两个 checkpoint 的 24 组专家权重及优化器张量均包含 8 个分片；HF 中三层的全部专家张量首维均为 288，并保留了 8 个 rank 各自的随机数状态。最终 journal 为 `CLEANED`，`step_latest` 指向 `step_2`，分区缓存已清理，进程退出后 8 张 GPU 的占用归零。

随后以相同配置和输出目录重新启动，退出码为 0，仍停在 `next_micro_step=2`、`global_step=2`。没有重新生成特征或训练分区；checkpoint 和缓存文件的大小、修改时间均保持不变。

当前机器上的训练日志为 `/tmp/deepspec-vllm-smoke-launch6.log`，重启日志为 `/tmp/deepspec-vllm-smoke-resume.log`；核验记录位于 `/tmp/deepspec-vllm-smoke5-20260907/` 下的 `feature_audit.json`、`checkpoint_audit.json` 和 `resume_audit.json`。本次完整训练 checkpoint 每步约 336.3 GiB，其中 HF 权重约 43.8 GiB，DCP 约 292.5 GiB，需与特征缓存一起计算存储预算。

这次测试覆盖短序列、单节点双分区切换与恢复。随后已补做 [128K 上下文单步训练实测](glm5_vllm_partition_step_128k_benchmark.md)：8 张 B300、global batch 8、512 个 anchors，首次训练 step 为 56.016 秒，完整保存和清理均通过。尚未验证多节点实跑和草稿接受率。小分区测试的耗时主要来自模型冷启动和 checkpoint 写盘，不能据此比较稳定训练吞吐。
