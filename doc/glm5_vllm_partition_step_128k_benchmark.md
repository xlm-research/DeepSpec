# GLM-5.3：128K 上下文单个训练 step 实测

2026-09-07，使用当前按分区运行的离线 vLLM 后端，真实完成一次 128K 上下文的 draft 参数更新。**首次训练 step 为 56.016 秒**，loss 为 **4.4018**。

## 测量条件

- 单节点 8 张 NVIDIA B300，完整 45 层 FP8 教师，完整 3 层 BF16 draft。
- 教师使用两个 TP4 副本；draft 使用 FSDP8、EP8，CP=TP=1。
- 每卡 local batch 1，global batch 8，梯度累积 1；每条输入实际为 131,072 tokens。
- 保留现有训练配置的 512 个 anchors、block size 7、目标层 `[2, 22, 42]`。
- 使用现有 `spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl` 的前 8 条 packed 样本，经过原训练 collator 截断。每条有 30,041–40,994 个监督 token，没有补齐或随机造 token。
- 本次分区包含 8 条样本，只训练 1 个 optimizer step，没有额外训练预热。遇到的首次编译开销包含在相应阶段中。

## 计时与显存

计时脚本在训练 batch 已传入 GPU 后同步 CUDA、同步各 rank，再记录墙钟时间；完成前向、loss、反向、梯度同步/裁剪及优化器更新后再次同步 CUDA。最终取 8 个 rank 中最慢的时间。教师生成、模型加载、缓存读取/H2D 和 checkpoint 写盘均在这个 step 计时范围之外。

| 项目 | 结果 |
| --- | --- |
| 首次训练 step，最慢 rank | **56.016 秒** |
| 各 rank 训练 step 范围 | 54.487–56.016 秒 |
| draft 训练阶段，含数据准备和日志等 | 63.860 秒 |
| 教师分区生成，含进程启动、权重加载、特征写入及校验 | 435.764 秒 |
| draft 和优化器加载/初始化 | 79.223 秒 |
| 完整 checkpoint 保存 | 261.421 秒 |
| 整次任务，从 launcher 启动到正常退出 | 1,081 秒，约 18 分 01 秒 |
| 单卡训练峰值 allocated，取所有 rank 最大值 | 92.285 GiB |
| 单卡训练峰值 reserved，取所有 rank 最大值 | 97.627 GiB |

阶段耗时使用 rank 0 的阶段边界；训练 step 使用各 rank 的最大同步耗时。PyTorch 显存统计不包含所有 CUDA/NCCL 开销，也不代表教师推理阶段的显存占用。

这是一轮首次 step 的测量，不是连续训练的稳定平均速度。它包含真实的 128K 上下文处理和现有 anchor 采样训练；未测量草稿接受率。

任务以退出码 0 结束，`step_1` 完整提交，HF 导出的三层专家张量均包含全部 288 个专家。分区缓存已清理，8 张 GPU 的占用归零。这里每个分区只有一个 step，因此整次任务包含一次完整教师启动与一次完整 checkpoint 保存；较大分区会将这部分开销分摊到多个 step。

完整冷启动耗时还包含 Python/分布式初始化和进程退出。输入子集准备及独立长度核验发生在 launcher 启动之前，不计入 1,081 秒。

原始记录已归档：[汇总](../output/glm5_vllm_partition_step_128k_20260907/summary.json)、[逐 rank 计时](../output/glm5_vllm_partition_step_128k_20260907/step_1.json)、[输入核验](../output/glm5_vllm_partition_step_128k_20260907/input_audit.json)、[特征核验](../output/glm5_vllm_partition_step_128k_20260907/feature_audit.json)、[环境与源码哈希](../output/glm5_vllm_partition_step_128k_20260907/metadata.json)、[完整日志](../output/glm5_vllm_partition_step_128k_20260907/launch.log)。

## 复现

原启动脚本可通过 `TRAIN_ENTRYPOINT=scripts/benchmark_glm5_partition_step.py` 使用计时入口，结果写入 `OUTPUT_ROOT/benchmark/`。未设置该变量时仍使用 `train.py`。

```bash
BENCH_ROOT=/tmp/glm5_vllm_128k_step_rerun
vllm/.venv/bin/python scripts/data/subset_jsonl.py \
  --input-path train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl \
  --output-path "$BENCH_ROOT/packed_first8.jsonl" \
  --num-records 8 --minimum-packed-tokens 131072

TARGET_BACKEND=vllm \
TRAIN_ENTRYPOINT=scripts/benchmark_glm5_partition_step.py \
PYTHON_BIN="$PWD/vllm/.venv/bin/python" \
TRAIN_DATA_PATH="$BENCH_ROOT/packed_first8.jsonl" \
OUTPUT_ROOT="$BENCH_ROOT/run" \
MAX_LENGTH=131072 NUM_ANCHORS=512 GLOBAL_BATCH_SIZE=8 \
PARTITIONED_MODEL_SWAP=true PARTITION_MAX_SAMPLES=8 MAX_TRAIN_STEPS=1 \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 VLLM_RAW_CACHE_DIR=/dev/shm \
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

本次原始运行目录为 `/tmp/deepspec-glm5-vllm-128k-step-20260907`；`input_audit.json` 记录实际输入长度和 token hash，`feature_audit.json` 核对所有缓存与输入对应，`run/benchmark/step_1.json` 保存逐 rank 计时和显存。权重从已有的本地 NVMe 模型缓存加载，checkpoint 和特征缓存也写入本地 NVMe。
