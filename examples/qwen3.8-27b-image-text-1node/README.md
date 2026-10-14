# Sensenova Flash Lite v42 单机图文 DSpark

全量模式使用全部 327 条不同训练源会话，从零训练 DSpark 草稿模型，
先进行真实 GPU 图文预填充验证、原文本回归和 17 个不同监督窗口的两步
训练预检（尾步包含 1 个真实窗口及 15 个零 loss padding），再完整遍历
全量 DSpark 监督窗口一次：

```bash
source h800conda.sh
pip install matplotlib tensorboard requests pillow
OUTPUT_DIR="$PWD/outputs/image-text-full" \
  bash examples/qwen3.8-27b-image-text-1node/run.sh --full-data
```

先检查启动命令、仅准备数据或仅运行 GPU 预检：

```bash
bash examples/qwen3.8-27b-image-text-1node/run.sh --full-data --dry-run
bash examples/qwen3.8-27b-image-text-1node/run.sh --full-data --prepare-only
bash examples/qwen3.8-27b-image-text-1node/run.sh --full-data --preflight-only
```

全量准备结果保存在独立的 `cache/qwen3.8-27b-image-text-1node/full-data/`，
包含真实监督窗口、源会话/assistant 监督段映射、图片内容指纹及上下文缩短范围。
保持 128K 总请求上限，为预填充验证生成的 1 token 预留位置。
所有选中的 assistant 正文、reasoning 和 tool calls 完整保留，
每段只在一个窗口中作为监督；重叠历史仅作为上下文。
超长上下文先移除可省略的历史；单个 user/tool 上下文无法容纳时，
按同一保留比例缩短文本前缀，并优先将完整监督段组合进可容纳的窗口。
图片与标记保持对应，中途 system 按原角色和顺序保留。
无法准备的记录明确报出源行，不静默跳过。

HTTPS 图片先缓存并实际解码；GIF 沿用静态首帧、EXIF 方向及白底转换行为。
仅优化 DSpark，目标模型、视觉编码器及共享 embedding/head/norm 保持冻结。
使用四卡 FSDP + 四卡 TP=4，global batch 为 16；
目标推理每次处理一个监督窗口，并为长窗口的隐藏态传输保留显存。
M 个真实监督窗口对应 `ceil(M/16)` 个 optimizer steps。
全量模式拒绝旧 smoke 的十步终点、十一学习率步数和 checkpoint 恢复参数；
零 loss padding 不计入源会话、监督段、样本或损失分母。

运行目录包含 `loss.png`、每个真实 optimizer metric step 的
`metrics.jsonl`、本地 TensorBoard 指标、`training.log`、actor 日志、
`coverage.json` 和最终 `checkpoints/`。
PNG 同时展示加权总 loss、CE、L1 和 confidence；`train/avg_loss` 继续表示
原有 CE 汇总值。完整遍历的依据是窗口 ID 与成功 optimizer 提交的对账，
并非日志步数或重复重载。运行失败时保留不完整覆盖报告。
DSpark 仍使用原有 anchor 抽样，不保证每个目标 token 在单轮都被抽中，
也不把一轮完成视为草稿质量验证。

从精确指标重新生成曲线：

```bash
python -m torchspec.utils.dspark_metrics \
  --input outputs/image-text-full/metrics.jsonl \
  --output outputs/image-text-full/loss.png
```

以下保留原三条源会话 smoke 示例的使用方式。

目标 checkpoint、草稿结构及四卡 FSDP + 四卡 TP=4 布局沿用
`../qwen3.8-27b-1node`。原文本示例保持独立。

2026-10-13 已通过三条图文 prefill、10 步训练及恢复第 11 步、原文本 GPU 回归。
实际结果及测试范围见 [验收记录](../../docs/qwen3_8_27b_multimodal_results.md)。

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

使用原文本示例的配置、loader 和默认仅最后 thinking 轮次监督规则做真实回归：

```bash
bash examples/qwen3.8-27b-image-text-1node/prefill.sh --legacy-text-only
```

此回归报告单独写入 `outputs/qwen3.8-27b-image-text-1node-text-prefill/`。

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
