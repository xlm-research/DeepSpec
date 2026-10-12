# SenseNova Flash Lite v42 三节点 DFlash2 训练

默认目标模型为
`/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp`。
该 checkpoint 的架构为 `Qwen3_5ForConditionalGeneration`，文本模型有 64 层、
hidden size 5120、词表大小 248320。目录和启动脚本沿用原有的 Qwen3.8 命名。

目标模型通过 vLLM 对数据集已有的 prompt 和回答做前向推理，提取五层中间 hidden
states。训练使用原样本的 loss mask，以已有回答作为监督目标，优化 token CE loss
和 DFlash2 selector loss。训练端只优化 DFlash2 草稿模型；目标模型及训练端加载的
目标 embedding、LM head 保持冻结。vLLM 不加载训练中的 DFlash2 草稿权重，
训练期间不向推理端导出或同步草稿权重。

默认使用 3 台八卡机器：1 台运行 FSDP 训练，另外 2 台各运行一个 TP=8 的目标模型
副本。训练长度上限为 131072 tokens，global batch 为 `8 × 1 × 4 = 32`。
数据默认使用 `mydata/sensenova-flash-lite-v42-text-1000000.jsonl`。

默认训练节点为 `10.120.0.211`，推理节点为 `10.120.0.218`、`10.120.0.187`。
Mooncake 使用已通过三节点 10 步训练验收的 RDMA `mlx5_0`：每个推理节点
256 GiB segment、2 GiB host buffer、1 个异步 PUT buffer；推理 batch 为 1，
sample pool 为 8，训练 prefetch depth 为 1。脚本默认复用
`cache/qwen3_8_27b-dflash2-vllm-decode-3node` 中的数据缓存。
正式运行按一轮数据自动计算训练步数，每 500 步保存 checkpoint。

草稿模型使用专用配置 `configs/draft_models/sensenova_flash_lite_v42_dflash2.json`：
5 层 GQA、32 个 query heads / 8 个 KV heads、head dim 128，hidden size 5120，
FFN intermediate size 17408。`layer_types` 明确指定前 4 层为 `sliding_attention`，
窗口大小为 2048；第 5 层为 `full_attention`，不限制滑动窗口。
前 4 层只读取窗口范围内且位于 anchor 之前的目标特征，第 5 层可读取 anchor
之前的全部目标特征。所有层均设置 `is_causal: false`，草稿块内保持双向注意力，
不同草稿块之间互不可见。
采集目标层 `[5, 19, 33, 47, 61]`，使用完整的 248320 词表和 mask token 248070。
该 ID 在目标 tokenizer 中为 `<|audio_start|>`，本配置在 `language_model_only: true`
的文本训练中将其复用为 MASK 占位，训练和部署须保持一致。block size 为 8
（1 个已知 token + 7 个预测 token），每条样本最多采样 512 个 anchors；
`dataset.min_loss_tokens: 16` 满足至少 `2 × block_size` 个监督 token 的要求。
CE 和 selector loss 权重均为 1.0，loss decay gamma 为 7.0，L1 loss 权重为 0.0。
因此设置 `inference.store_last_hidden_states: false`，无需传输目标模型最后一层
hidden states，也无需加载目标 final norm。

在仓库根目录一键启动（任一节点均可）：

```bash
bash run.sh
```

入口会自动加载 `h800conda.sh`，在训练节点启动或复用 Ray head，并通过 SSH 并行
启动尚未加入集群的两个推理节点。确认三台机器各至少有 8 张 GPU 后，自动在 head
执行训练脚本；训练结束后保留 Ray，供下次复用。从推理节点执行时会自动转到 head。
三台机器需要共享当前仓库和环境，并已配置免密 SSH 及已知主机密钥。
SSH 默认使用当前用户名和 22 端口，可通过 `SSH_USER`、`SSH_PORT` 修改。

```bash
bash run.sh --dry-run                    # 只预览，不连接 SSH/Ray
bash run.sh training.num_train_steps=10  # 参数继续透传给训练脚本
LOAD_PATH=/实际/checkpoints目录 bash run.sh
```

`TRAIN_NODE_IPS`、`INFERENCE_NODE_IPS` 可覆盖上述默认节点，`RAY_PORT` 默认 6379。
原来的启动脚本仍可在已有 Ray 集群的 head 节点直接执行：

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

接着 2026-10-02 的 10 步验收 checkpoint 继续训练：

```bash
LOAD_PATH="$PWD/outputs/dflash2-rdma-acceptance-20261002_011532/checkpoints" \
bash examples/qwen3.8-27b-dflash2-3node/run.sh
```

该 checkpoint 的 `global_step` 为 10，当前保存逻辑将目录编号为 `iter_0000011`；
不设置 `CKPT_STEP` 即可按 tracker 选择它，并恢复模型、优化器和学习率状态。

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
