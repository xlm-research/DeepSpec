# Qwen3.5 Loop + mHC DSpark 训练

单机八张 H800：四卡运行 TP=4 的 vLLM V2 目标引擎，四卡 FSDP 训练 DSpark。
默认目标为 `ckpt-225-bf16`，数据为 `mydata/test_20.jsonl`。使用完整对话作为上下文，只监督最后一个 assistant 回答。目标模型、目标 embedding 和 LM head 冻结。

从仓库根目录运行：

```bash
bash examples/qwen3.5-loop-mhc-dspark/run.sh --dry-run
bash examples/qwen3.5-loop-mhc-dspark/run.sh --print-config-only
bash examples/qwen3.5-loop-mhc-dspark/run.sh

# 启用目标模型 mHC Triton 融合
VLLM_QWEN3_5_MHC_BACKEND=triton bash examples/qwen3.5-loop-mhc-dspark/run.sh
```

启动脚本使用已有 `.venv` 和 `h800conda.sh`，开启 V2 runner。默认输出目录带时间戳；可通过 `OUTPUT_DIR`、`CACHE_DIR`、`TRAIN_DATA_PATH`、`TARGET_MODEL_PATH` 指定路径，或追加 `key=value` 配置覆盖。设置 `TORCHSPEC_LOG_LEVEL=DEBUG` 可记录采集分块、训练样本标识和分项指标。

默认一轮 20 条有效数据，全局 batch=4，完成五次 optimizer 更新；在第 2、4 步及轮末保存，保留最近两份完整 checkpoint。第 4 步恢复到第 5 步：

```bash
LOAD_PATH=/absolute/path/to/original-output/checkpoints \
CKPT_STEP=4 \
OUTPUT_DIR=/absolute/path/to/new-resume-output \
bash examples/qwen3.5-loop-mhc-dspark/run.sh
```

恢复检查模型、optimizer、scheduler、RNG、步数和训练契约。目标身份、草稿配置、采集点、最终状态归一化位置或监督策略不一致时拒绝加载。异步数据队列恢复为 best-effort，不承诺逐样本或逐位重放。

五份条件特征来自零基执行层 `[3, 27, 51, 75, 78]`，对应辅助槽 `[4, 28, 52, 76, 79]`；独立最终槽为 80，保存实际进入目标 LM head 的 post-norm 状态。原 checkpoint 保留 32 个物理层。草稿为五层、隐藏宽度 4096、Q heads 16、KV heads 4、head dimension 256、Markov rank 256，从头初始化可训练参数。

当前 attention 头数与维度已按用户要求对齐目标 full attention。旧的 32/8/128 草稿 checkpoint 不能直接恢复到新配置，Q/K norm 参数也由 `[128]` 变为 `[256]`。2026-10-12 已用当前 16/4/256 结构和 Triton 融合完成默认 20 条数据、五次训练更新，并验证 checkpoint 参数与 optimizer 状态；见[单机八卡训练验收](../../docs/qwen3_5_loop_mhc_8gpu_revalidation_plan.md)。下述 2026-10-10 的恢复与满长记录仍属于旧结构。

128K 表示输入与内部生成 token 合计上限 131072，因此训练输入最多 131071。分块 prefill 的单步预算为 8192，每次仅一个请求；prefix caching 关闭。满长样本六份 BF16 表示约 6 GiB，配置使用 8 GiB host buffer、64 GiB Mooncake segment 和最多四样本的等待池。直接采集仅在 TP rank 0 保存 CPU 副本。

2026-10-10 已完成以下实机验收：

| 场景 | 结果 | 记录 |
| --- | --- | --- |
| 指定 20 条数据 | 五次更新；保存第 2、4、5 步，保留第 4、5 步 | [训练记录](../../outputs/loop_dspark_acceptance_fixed_20261010/run_summary.json) |
| 独立完整恢复 | 第 4→5 步；62 组 Adam 状态推进，草稿参数实际改变 | [恢复记录](../../outputs/loop_dspark_resume_fixed_20261010/run_summary.json) |
| 满 128K | 四条各 131071 tokens、各 16 块；完成前向、反向和一次更新，无 OOM | [满长记录](../../outputs/loop_dspark_128k_fixed_20261010/run_summary.json) |
| TP=8 原目标推理 | 15→8193→15 tokens 三次输入，均生成 16 tokens；前后短请求结果一致 | [推理结果](../../outputs/loop_dspark_tp8_regression_20261010/result.json) |

主训练的完整 checkpoint 位于 `outputs/loop_dspark_acceptance_fixed_20261010/checkpoints/`。三次训练检查点的模型张量均有限，冻结 embedding 与目标原权重逐元素一致；主训练、恢复和满长运行均有草稿参数变化证据。

[验收汇总](verification.json)记录环境、分项 loss、采集位置、参数比较、测试日志和历史异常。验收中修复了保存步数偏移及 JSON 整数键导致的恢复契约误判。128K 收尾日志中出现 Mooncake 取消挂载错误和 vLLM 终止剩余 worker 的记录；运行退出码为 0，GPU 全部释放，后续 TP=8 回归通过。

功能验收不代表草稿接受率、加速比或模型质量结论。完整合同见[训练规格](../../docs/qwen3_5_loop_mhc_dspark_spec.md)。
