# Sensenova Flash Lite v42 单机图文 DSpark

目标 checkpoint、草稿结构及四卡 FSDP + 四卡 TP=4 布局沿用
`../qwen3.8-27b-1node`。原文本示例保持独立。

本例使用原生 Qwen3.5 模板保留结构化图片、顶层 tools、各轮 reasoning 和
tool_calls。所有 assistant 回答参与监督；user/tool 和视觉 token 不计算回答 loss。
vLLM 读取完整已有会话，执行 prefill 并生成 1 token；新增 token 不用于训练。
目标模型、视觉编码器及目标 embedding/head/norm 保持冻结，只优化 DSpark。

在仓库根目录准备样本（需要现有 H800 环境）：

```bash
source h800conda.sh
python examples/qwen3.8-27b-image-text-1node/prepare_data.py
```

默认选取原 JSONL 零起始索引 114、217、231，覆盖单图、多图和 tool 图片。
`media/...` 相对 `/mnt/afs_toolcall/juxiaolong1/Data/draftData` 解析为本地 URI。
输出位于 `cache/qwen3.8-27b-image-text-1node/data/`：三条原样本的
`prefill.jsonl`、160 条独立 ID 重复记录的 `train.jsonl`、processor 预检
`manifest.json`。原数据不修改；超长样本报错，不静默截断。
重复数据只用于验证训练链路，不能作为训练质量评估。

验证图文 prefill 和文本请求、从 Mooncake 实际取回五层特征和最后状态：

```bash
bash examples/qwen3.8-27b-image-text-1node/prefill.sh
```

默认使用 GPU 0–3。报告写入 `outputs/qwen3.8-27b-image-text-1node-prefill/`，
包含每条请求的展开长度、单 token 生成、有效监督数、捕获槽及张量形状。

训练十个 optimizer steps，并保存第十步 checkpoint：

```bash
OUTPUT_DIR="$PWD/outputs/image-text-smoke-10" \
  bash examples/qwen3.8-27b-image-text-1node/run.sh
```

恢复完整训练状态，累计跑到第十一步：

```bash
LOAD_PATH="$PWD/outputs/image-text-smoke-10/checkpoints" CKPT_STEP=10 \
OUTPUT_DIR="$PWD/outputs/image-text-smoke-11" \
  bash examples/qwen3.8-27b-image-text-1node/run.sh training.num_train_steps=11
```

首次运行和恢复均固定 `lr_total_steps=11`，保证恢复的一步仍有正学习率。
训练日志与 checkpoint 分别保存在运行目录和 `running_logs/`。
可追加配置覆盖或使用 `--dry-run`、`--print-config-only` 检查命令。

上下文上限保持 131072，并给内部生成预留 1 token。图文示例设置
`language_model_only=false`，保留 `VLLM_USE_V2_MODEL_RUNNER=0`，使用当前
`extract_hidden_states` 路径。视觉展开后由 renderer 重新生成 packed mask，
避免原占位符长度的 mask 进入训练。Mooncake pool 为 64GB；该 smoke 示例
不承诺全部 327 条会话、GIF/远程图片或完整训练 epoch 已通过验收。
