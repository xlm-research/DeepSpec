# Loop/mHC 单机图文 DSpark

使用 `mydata/image-text-327.jsonl` 全部 327 条源记录，从零训练当前五层 Loop DSpark 草稿一轮。
冻结目标模型及视觉编码器，四卡 TP=4 目标推理、四卡 FSDP 训练，global batch=16。
本例复用相邻图文示例的全量驱动、窗口准备、覆盖核对和预检。

```bash
bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh --dry-run
bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh --prepare-only
bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh --preflight-only
bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh
```

`OUTPUT_DIR`、`CACHE_DIR`、`LOG_DIR` 可指定独立产物位置。每次训练使用新输出目录；
入口自动加载仓库的 H800 环境。`--prepare-only` 在准备后输出有效 `config.yaml`，
可直接检查它；`--dry-run` 不加载环境或启动 GPU。

目标为现有 `ckpt-225-bf16`，V2 direct capture 使用执行层 `[3,27,51,75,78]`，
独立 final 槽 80 已经 post-norm。图文适配允许视觉前向，保留 Loop 的其余限制。
五层草稿 hidden=4096，Q/KV heads=16/4、head_dim=256，沿用原 DSpark 目标。

128K 总请求长度包含一个预填充验证 token；图片展开后的输入至多 131071 tokens。
所有 assistant 正文、历史 reasoning 和 tool calls 保留并监督一次；窗口重叠历史只作上下文。
必要时省略历史或缩短 user/tool 文本前缀，记录裁剪范围。中途 system 保持原角色和顺序。
GIF 读取首帧，HTTPS 图片缓存后实际解码；图片失败或无法容纳完整监督段时明确报错。
原始 JSONL 和 checkpoint 不修改。源文件、准备语义及模型模板/processor 指纹不兼容时重建缓存。

M 个窗口对应 `ceil(M/16)` 次更新，尾批用零 loss dummy 补齐；覆盖按成功更新的窗口核对。
真实预检包含 17 个不同窗口的图文采集、Loop 原文本请求，以及两步含尾批训练。
正式全量训练使用另一个全新草稿状态。

运行目录保存 `loss.png`、`acceptance.png`、`metrics.jsonl`、本地 TensorBoard、
`coverage.json`、`training.log`、actor 日志及完整最终 checkpoint。
接受概率为 `1 - train/l1_loss / 2`，其中 L1 未乘以损失权重 0.9；
统计来自该步更新前的训练上下文，按有效监督和位置衰减加权，横轴为 optimizer step。
它是训练接受概率估计，不代表实际生成接受率。

从精确记录重绘：

```bash
source h800conda.sh
uv run --no-project --python .venv/bin/python .venv/bin/python -m torchspec.utils.dspark_metrics --input "$OUTPUT_DIR/metrics.jsonl" --output "$OUTPUT_DIR/loss.png"
uv run --no-project --python .venv/bin/python .venv/bin/python -m torchspec.utils.dspark_metrics --input "$OUTPUT_DIR/metrics.jsonl" --output "$OUTPUT_DIR/acceptance.png" --kind acceptance
```

当前实现及真实运行的验收状态见[需求与执行记录](../../.scratch/qwen3_5_loop_mhc_image_text_dspark/spec.md)。

## 已完成的运行（2026-10-15）

`outputs/loop-image-text-20261014/` 已正常完成一轮：327 条源记录生成 595 个窗口，
38 次 optimizer 更新，末批补 13 个零 loss dummy；28,872 个 assistant 段无缺失、无重复。
输入最长 131071 tokens，单窗口最多 22 图；324 个窗口有已记录的上下文省略或前缀缩短。

十七窗口真实图文采集、Loop 原纯文本回归、两步尾批训练预检均通过。
全量四个 rank 的真实参数更新、38 步有限指标、TensorBoard 一致性、
最终 checkpoint 的模型/optimizer/scheduler/RNG 均通过独立核验。
两张 PNG 从精确指标离线重绘后，与训练产物逐字节一致。
首步/末步加权总 loss 分别为 3.71817492 / 2.73061622；
训练接受概率估计分别为 0.2455% / 1.5488%，统计口径同上。

- [Loss PNG](../../outputs/loop-image-text-20261014/loss.png)、[训练接受概率 PNG](../../outputs/loop-image-text-20261014/acceptance.png)、[精确指标](../../outputs/loop-image-text-20261014/metrics.jsonl)。
- [完整 checkpoint：iter_0000038](../../outputs/loop-image-text-20261014/checkpoints/iter_0000038/)。
- [独立验收报告](../../outputs/loop-image-text-20261014/independent_acceptance.json)、[覆盖报告](../../outputs/loop-image-text-20261014/coverage.json)、[实际配置](../../outputs/loop-image-text-20261014/config.yaml)。
- [曲线重绘核验](../../outputs/loop-image-text-20261014/curve_regeneration_audit.json)、[原数据与模型文件身份核验](../../outputs/loop-image-text-20261014/input_identity_audit.json)。
