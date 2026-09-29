# Qwen3.8-27B 三节点 DFlash2 训练

目标模型通过 vLLM 对数据集已有的 prompt 和回答做前向推理，提取五层中间 hidden
states。训练使用原样本的 loss mask，以已有回答作为监督目标，优化 token CE loss
和 DFlash2 selector loss。训练端只优化 DFlash2 草稿模型；目标模型及训练端加载的
目标 embedding、LM head 保持冻结。vLLM 不加载训练中的 DFlash2 草稿权重，
训练期间不向推理端导出或同步草稿权重。

默认使用 3 台八卡机器：1 台运行 FSDP 训练，另外 2 台各运行一个 TP=8 的目标模型
副本。训练长度上限为 131072 tokens，global batch 为 `8 × 1 × 4 = 32`。
数据默认使用 `mydata/sensenova-flash-lite-v42-text-1000000.jsonl`。

草稿模型复用 `configs/ci/qwen3_8_27b_dflash2_draft.json`：5 层、hidden size 5120，
采集目标层 `[3, 15, 31, 47, 59]`，使用完整的 248320 词表。block size 为 8
（1 个已知 token + 7 个预测 token），每条样本最多采样 512 个 anchors；
`dataset.min_loss_tokens: 16` 满足至少 `2 × block_size` 个监督 token 的要求。
CE 和 selector loss 权重均为 1.0，loss decay gamma 为 7.0，L1 loss 权重为 0.0。
因此设置 `inference.store_last_hidden_states: false`，无需传输目标模型最后一层
hidden states，也无需加载目标 final norm。

在已有 Ray 集群的 head 节点执行：

```bash
bash examples/qwen3.8-27b-dflash2-3node/run.sh
```

指定节点（沿用原三节点地址时）：

```bash
TRAIN_NODE_IPS=10.120.0.211 \
INFERENCE_NODE_IPS=10.120.0.218,10.120.0.187 \
bash examples/qwen3.8-27b-dflash2-3node/run.sh
```

检查命令或解析后的配置：

```bash
bash examples/qwen3.8-27b-dflash2-3node/run.sh --dry-run
bash examples/qwen3.8-27b-dflash2-3node/run.sh --print-config-only
```

默认从头训练；恢复 DFlash2 完整训练状态时设置 `LOAD_PATH=/实际/checkpoints目录`，可选
`CKPT_STEP=1000`，未指定步数时读取该目录的最新 checkpoint。新运行写入独立的
时间戳输出目录。普通参数可以在命令后追加，如 `training.num_train_steps=100`。

`config.yaml` 和启动脚本都固定 `training.train_with_decode: false`、
`decode.target_only: false`、`decode.speculative_algorithm: null` 和
`decode.weight_sync_enabled: false`，采用已有样本监督的 prefill 路径。
目标模型仍进行前向推理；vLLM 使用 `extract_hidden_states` 提取训练所需状态，
其中的辅助缓存模块不加载 DFlash2 权重。所有节点应使用当前仓库的本地 vLLM 代码；
脚本通过 `h800conda.sh` 配置环境，并设置 `VLLM_USE_V2_MODEL_RUNNER=0`，
使用当前支持 `extract_hidden_states` 的 Model Runner V1。

`training.max_seq_length` 和 vLLM 的 `max_model_len` 都保持为 131072（128K）。
数据预处理为内部生成预留 1 个 token，将输入样本最多裁到 131071 tokens，
输入与该 token 的合计不超过 131072，避免满长输入触发 vLLM 的长度校验错误。
prefill 分支内部使用 `SamplingParams(max_tokens=1, temperature=0)`，训练只使用
输入样本的 hidden states 和原始 loss mask，不依赖新增 token 构造监督，因此不会
因为只生成 1 token 而被在线 decode 分支当作零 loss 样本丢弃。
YAML 中保留 `decode.max_new_tokens: 1`、`decode.min_new_tokens: 1`；
在当前 prefill 模式下实际执行的是上述内部采样参数。
