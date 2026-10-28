# Qwen3.5 Loop/mHC 三节点 DSpark 训练

参考 [Qwen3.8 三节点示例](../qwen3.8-27b-3node/README.md)的目录结构、启动方式、
节点选择和恢复入口，复用 [Loop/mHC 单节点示例](../qwen3.5-loop-mhc-dspark/README.md)
的目标模型、草稿结构和隐状态采集契约。

当前为供用户检查的脚本与配置，尚未执行启动脚本、启动 Ray 或进行三节点训练验收。
默认数据与拓扑采用下述建议值，可在检查后调整。

## 默认配置

| 项目 | 默认值 |
| --- | --- |
| 目录 | `examples/qwen3.5-loop-mhc-dspark-3-node/`，与参考目录并列 |
| 目标模型 | `1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16` |
| 草稿配置 | `configs/draft_models/qwen3_5_loop_mhc_dspark.json` |
| 训练节点 | 1 台、8 张 H800，FSDP FULL_SHARD |
| 推理节点 | 2 台、每台 8 张 H800，各运行独立 TP=8、PP=1 的目标副本 |
| 数据 | `mydata/sensenova-flash-lite-v42-text-1000000.jsonl` |
| 监督范围 | 完整对话作为上下文，只监督最后一个 assistant 回答 |
| 全局 batch | `8 × microbatch 1 × accumulation 4 = 32` |
| 长度上限 | 131072 tokens（128K），输入最多 131071，预留内部 token |
| 训练时长 | 1 epoch；每 500 步及轮末保存，保留最近 2 份 checkpoint |
| 日志 | 本地文件与 W&B；可通过 `WANDB_MODE` 或 `logging.report_to=none` 调整 |

只优化 DSpark 的可训练参数；目标模型以及训练侧使用的目标 embedding、LM head
保持冻结。草稿为五层、隐藏宽度 4096、Q/KV heads 16/4、head dimension 256、
Markov rank 256，默认从头初始化。

## 启动方式（供检查后使用）

三台机器须共享仓库、模型与数据，使用 `h800conda-tmp.sh` 配置的 Python 环境和本仓库的
本地 vLLM。先按 [Ray 集群说明](../../docs/ray.md#multi-node-local-cluster)启动或加入
三节点 Ray 集群，再在 head 节点执行本目录脚本。脚本通过 `ray status` 检查连接；
实际 GPU 资源与节点分配由 TorchSpec placement group 校验。

以下命令仅作为使用说明，本次未执行：

```bash
# 只打印将执行的命令，不加载环境、不连接 Ray、不创建训练日志。
bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh --dry-run

# 加载环境、解析配置后退出；train_entry 会保存 config.yaml 快照，但不启动训练。
bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh --print-config-only

# 检查完成后，在既有 Ray head 上运行；此处 IP 沿用参考示例，可替换。
TRAIN_NODE_IPS=10.120.0.211 \
INFERENCE_NODE_IPS=10.120.0.218,10.120.0.187 \
RAY_ADDRESS=10.120.0.211:6379 \
bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh
```

不传节点 IP 时使用 TorchSpec 默认 placement；指定时必须同时提供一个训练 IP 和
两个推理 IP。脚本连接既有集群，默认 `RAY_ADDRESS=auto`，不自动通过 SSH 启动节点。
根目录 `bash run.sh dspark` 仍指向 Qwen3.8 示例，运行本模型应使用上面的完整脚本路径。

支持 `LOOP_TARGET_MODEL_PATH`、`TRAIN_DATA_PATH`、`OUTPUT_DIR`、`CACHE_DIR`、`LOG_DIR`、
`MOONCAKE_PROTOCOL`、`WANDB_NAME`、`WANDB_PROJECT`、`WANDB_ENTITY` 和 `WANDB_MODE`。
默认传输为 TCP。输出目录自动带时间戳，训练日志写入 `running_logs/`；普通配置可以
在命令后追加 `key=value`。

目标路径使用专属 `LOOP_TARGET_MODEL_PATH`，避免共享环境中预设的 `TARGET_MODEL_PATH`
把此示例切回 Qwen3.8；也可以追加 `model.target_model_path=/实际/Loop权重目录`。

若改用原先的 20 条数据，应同时降低梯度累积，避免 20 条数据不足全局 batch 32：

```bash
TRAIN_DATA_PATH="$PWD/mydata/test_20.jsonl" \
bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh \
  training.draft_accumulation_steps=1 training.save_interval=2 logging.report_to=none
```

此时全局 batch 为 8；实际步数按过滤后的有效样本数计算。

## Loop/mHC 采集与缓存

本模型使用 `VLLM_USE_V2_MODEL_RUNNER=1` 和 `direct_hidden_state_capture=true`。
五份条件特征取自零基执行层 `[3, 27, 51, 75, 78]`，对应辅助槽 `[4, 28, 52, 76, 79]`。
独立最终槽 80 保存进入目标 LM head 的真实 post-norm 状态，配置为
`last_hidden_states_prenorm=false`。执行层与物理层的区别见 [领域词汇表](../../CONTEXT.md)。

脚本固定已有回答监督的采集模式，关闭在线生成监督和草稿权重同步，并固定 direct
capture 与 post-norm 设置；这些字段不会被命令行覆盖。每个目标副本一次处理一条请求，
分块预填充预算为 8192 tokens，最多两副本同时处理两条；direct capture 路径关闭
prefix caching。

满长样本的六份 BF16 表示约 6 GiB。八个训练 rank 每次派发需要八条，因此等待池从
单节点示例的 4 调整为 8，避免池先满而无法凑齐一次派发。保留 `prefetch_depth=1`，
减少训练侧预取；梯度累积为 4 时，控制器仍可能先排入 `4 × 8 = 32` 条训练样本，
仅这些样本的有效隐状态就约 192 GiB。

`global_segment_size=256GB` 是每个推理生产者贡献的容量，两副本合计约 512 GiB；
训练侧不贡献大 segment。`local_buffer_size=16GB` 是每个训练 rank 的接收缓存，
八 rank 合计约 128 GiB；`host_buffer_size=8GB` 用于单样本发布暂存。
这些是依据满长数据体积和排队方式给出的初始容量，未做三节点内存峰值或吞吐验收。

## 恢复训练

`restart.sh` 默认从 `/tmp/deepspec_vllm_torchtitan_envs` 加载环境，并将 Triton 缓存放在
该环境的 `.cache/torchspec-triton-$UID` 子目录；设置 `TORCHSPEC_CONDA_ENV` 时缓存默认
随环境目录调整。直接使用 `run.sh` 时，缓存默认仍为各节点本地的
`/tmp/torchspec-triton-$UID`。两者均通过 Ray runtime env 将缓存路径传递给目标 worker，覆盖 Conda 或旧 Ray 节点继承的
共享盘缓存。需要自定义位置时设置 `TORCHSPEC_TRITON_CACHE_DIR`，该路径必须在所有
节点可用，建议使用本地 SSD。更改缓存或内核代码后须重启目标进程才能生效。
此启动脚本自动设置 `TORCHSPEC_TRITON_CACHE_DIR` 来启用跨节点覆盖；其他训练任务
未设置该变量时保留原来的节点缓存行为，不转发驱动进程继承的 `TRITON_CACHE_DIR`。

`restart.sh` 默认启用 `VLLM_QWEN3_5_MHC_BACKEND=triton` 和
`VLLM_QWEN3_5_MHC_FUSION=all`，保留显式环境变量覆盖。需要回退原 PyTorch 路径时，
分别设置为 `torch` 和 `none`。mHC 内核复用运行时长度参数，变长样本及最后一个
prefill chunk 不再各自产生新的长度特化版本。

```bash
LOAD_PATH=/absolute/path/to/previous-output/checkpoints \
CKPT_STEP=1000 \
OUTPUT_DIR=/absolute/path/to/new-resume-output \
bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh
```

未设置 `CKPT_STEP` 时读取最新 checkpoint。脚本检查 model、optimizer、lr_scheduler、
RNG 与 metadata 是否存在，并要求环境变量指定的恢复运行使用新的输出目录。
训练端进一步检查目标身份、草稿配置、采集点、归一化位置和监督契约；异步样本队列
恢复为 best-effort。旧的 32/8/128 草稿 attention 配置所产 checkpoint 不适用于当前
16/4/256 配置，不能将单节点旧验收记录当作当前结构的三节点恢复保证。

本次沿用已有 [采集契约 ADR](../../docs/adr/0002-loop-mhc-dspark-capture-contract.md)，
没有新增难以逆转的模型或接口决策。

## 本次检查

已通过 `bash -n` 语法检查，以及 `load_config(save_snapshot=False)` 的配置解析检查。
核对了默认目标的 Loop 架构、共享环境目标变量的隔离、专属变量与 CLI 覆盖、数据路径、
草稿采集层、24 卡拓扑、样本池派发门槛及 128K 长度一致性。
检查仅读取配置和模型元数据，没有执行启动脚本、连接 Ray、加载模型权重或运行训练。
