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
入口直接加载 `/tmp/deepspec_vllm_torchtitan_envs`，设置仓库的 H800 环境变量，
并直接使用该环境的 Python（子进程继承同一个解释器）。运行前需将环境解压至此路径。
不再调用 AFS 上的 Conda 初始化脚本。`TORCHSPEC_CONDA_ENV` 只允许指定 `/tmp` 下的环境，
本地环境缺失时报错，不回退到 AFS。其它使用 `h800conda-tmp.sh` 的入口也遵循此规则。
`h800conda.sh` 保留原有 AFS Conda 加载方式；日常运行使用 `h800conda-tmp.sh`。
多节点运行时，每个节点都需要提前准备相同路径的 `/tmp` 环境。

用 `ENV_MODE=afs` 可显式选择 `h800conda.sh`，默认 `ENV_MODE=tmp`。
临时性能对比可传 `--benchmark-config /absolute/path/config.yaml`，直接启动该配置的训练，
跳过全量数据准备和预检；配置必须自行指定临时数据、步数、输出及缓存目录。
`OUTPUT_DIR` 应与配置的 `output_dir` 一致，用于日志和独立编译缓存。
此模式用于实验，不替代全量数据覆盖验收。

```bash
ENV_MODE=afs OUTPUT_DIR=/tmp/experiment/afs bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh --benchmark-config /tmp/experiment/afs/config.yaml
# AFS 完成且 GPU 释放后，再运行下一组。
ENV_MODE=tmp OUTPUT_DIR=/tmp/experiment/tmp bash examples/qwen3.5-loop-mhc-image-text-dspark-1-node/run.sh --benchmark-config /tmp/experiment/tmp/config.yaml
```

`--prepare-only` 在准备后输出有效 `config.yaml`，
可直接检查它；`--dry-run` 不加载环境或启动 GPU。

2026-10-23 旧版入口（仍调用 AFS Conda 初始化脚本）在同一节点交替测量各 5 次
（未清理缓存），中位数如下；这些是历史对比数据，不代表当前直接加载方式：

| 测量范围 | AFS 环境 | `/tmp` 环境 |
| --- | ---: | ---: |
| Conda 初始化与激活 | 6.74 秒 | 5.78 秒 |
| 入口及 torch/ray/transformers/vllm 导入 | 75.22 秒 | 10.61 秒 |
| H800 环境初始化、上述导入、CUDA 可用性检查及进程退出 | 86.33 秒 | 17.92 秒 |

最后一项约加速 4.82 倍、减少 79.2% 耗时。各项中位数独立统计，不应直接相加。
这不是受控冷启动测试，不包含环境复制解压、模型权重加载或训练；仓库代码仍位于 AFS。
两种环境均为 PyTorch `2.13.0+cu130`，CUDA 可用。详见[逐次数据及测量口径](env-loading-benchmark.json)。

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
source h800conda-tmp.sh
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

## vLLM 推理优化（2026-10-23）

在本机 H800 上，以四个固定真实图文窗口测量预填充和完整隐藏状态读取，
三轮总耗时的中位数从 **61.8428 秒降至 46.9909 秒，减少 24.02%，加速 1.316 倍**。
优化后三轮分别为 46.9519、47.0165、46.9909 秒。窗口包含 11065、65484、131071、130903
个输入 tokens，以及 3、1、2、22 张图片；一次预热后计时三轮，计时覆盖 `generate`
至 Mooncake 完整特征读取，不包含模型加载、首次编译及随后的一致性检查。
这是该固定预填充工作负载的结果，不是整轮训练或生成阶段的吞吐指标。

继续使用 **TP=4、BF16、128K 上限、8192-token 分块**。入口默认设置
`VLLM_QWEN3_5_MHC_BACKEND=triton`、`VLLM_QWEN3_5_MHC_FUSION=all`，支持环境变量覆盖，
`--dry-run` 会显示实际选择。无需改动数据、模型权重或五层辅助捕获配置。

本地 vLLM 保留了两项优化：四个 token 的 mHC 合并使用块对角 Tensor Core 计算，
保留原来的 BF16 舍入边界；归一化调整线程布局并复用 FP32 临时缓冲区，
保留 PyTorch 的求均值和倒平方根。merge 的小输入及不支持的类型或布局继续使用原路径。
在 8192 tokens 的独立算子测速中，merge 从 2.5462 ms 降到 0.2036 ms，
normalize 从 0.8131 ms 降到 0.7399 ms；这些算子倍率不能直接当作整体加速倍率。

| 实验 | 四窗口中位耗时 | 处置 |
| --- | ---: | --- |
| 原始 TP=4 | 61.8428 秒 | 基线 |
| TP=2 | 67.8180 秒 | 更慢且特征检查失败，回退 |
| TP=1 | 77.8682 秒 | 更慢且特征检查失败，回退 |
| TP=4 + Tensor Core merge | 49.8277 秒 | 通过检查，保留 |
| 上述方案 + 16384-token 分块 | 47.9622 秒 | 特征检查失败，回退 |
| TP=4 + merge + 归一化优化 | **46.9909 秒** | **通过检查，最终保留** |

最终 16 次采集均通过完整 input IDs、监督 mask、特征形状和有限值检查，
五组辅助层及独立 post-norm final 状态的抽样特征与基线逐位一致，输出 token 一致。
92 项 vLLM 内核/配置测试和 35 项 TorchSpec capture/engine 测试通过；
候选文件的适用 pre-commit 检查通过。

完整数据见 [inference-benchmark.json](inference-benchmark.json)，
本机完整日志与控制器报告见 [autoresearch report](../../vllm/autoresearch-results/report.html)。
vLLM 分支为 `autoresearch/loop-image-prefill-20261023`，最终提交为
`e92cadc862730336efa13eb0967e560ae33969d8`。在 GPU 0–3 空闲时，从仓库根目录复现：

```bash
source h800conda-tmp.sh
bash vllm/benchmarks/qwen3_5_loop_autoresearch/run.sh
bash vllm/benchmarks/qwen3_5_loop_autoresearch/guard.sh
```

验证器使用本机 `outputs/loop-autoresearch-20261023/` 中冻结的样本、基线特征及源码指纹。
若输入或被锁定的 TorchSpec 文件已变化，会明确拒绝与旧基线比较。
