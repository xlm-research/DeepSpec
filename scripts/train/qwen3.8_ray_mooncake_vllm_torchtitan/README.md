# Qwen3.8：Ray + Mooncake + vLLM + TorchTitan

换窗口继续讨论或开发时，先读 [会话交接](HANDOFF.md)，其中记录已确认要求、版本、改动、运行证据和待办。

启动已通过真实模型验证的单机 8 卡 DSpark 训练流水线：一个 vLLM TP4 生产者、
一个 TorchTitan TP4 消费者，各占四张不同的 GPU。Ray 分配 GPU 和协调数据状态，
vLLM 管理推理 worker，TorchTitan 管理训练 rank。特征经 Mooncake CPU buffer
送入原生 DSpark 的 `fc → hidden_norm → context K/V` 训练路径。

## 启动

在项目根目录执行：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train.sh
```

也可以从其他目录使用脚本的绝对路径；脚本会自动进入项目根目录。
固定使用 `/tmp/deepspec_vllm_torchtitan_envs/bin/python`，无需激活环境，不使用 uv。
启动前需要整机 8 张 GPU 空闲且全部可见；Ray 为两端分配互不重叠的 GPU，
实际编号见运行目录的 `result.json`。本次已验证运行的分配为生产者 0–3、消费者 4–7。

## 默认配置

| 参数 | 默认值 |
| --- | --- |
| Target 模型 | `/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B` |
| 输入 JSONL | `outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl` |
| 生产者 | vLLM TP4 / DP1 / PP1 |
| 消费者 | TorchTitan TP4 / DP1 / CP1 / PP1，GAS4 |
| 上下文上限 | 4096 token |
| 优化器更新 | 3 次，共 12 个微批 |
| 特征窗口 / Store 池 | 8 批 / 4 GiB |
| 传输 | TCP 配置，消费端 pinned CPU 接收后搬到 GPU |
| 运行超时 | 1800 秒 |
| 输出目录 | `outputs/qwen3.8_ray_mooncake_vllm_torchtitan_<时间>_<PID>` |

默认配置用于复现已经跑通的短程训练。当前入口固定单机 4+4；扩大上下文或训练规模
需要相应调整容量和超时，已有结果不代表 128K、多机或 RDMA 已通过验证。

## 覆盖参数

脚本参数直接传给 `deepspec.pipeline.run`，同名参数覆盖默认值。例如：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train.sh \
  --source /absolute/path/to/train.jsonl \
  --output /absolute/path/to/new_run \
  --steps 10 \
  --timeout-seconds 3600
```

`--output` 指定的目录必须尚不存在。相对路径以项目根目录为基准。
数据必须满足原生预处理格式，并提供足够的有效样本；默认每次更新需要四个微批。

查看全部参数（不启动训练）：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train.sh --help
```

仅打印命令，不调用 Python、不创建输出目录或启动服务：

```bash
DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train.sh
```

`--prepare-only` 会生成 token 输入和训练计划，但不启动 Ray、Mooncake 或模型。

## 日志与结果

输出目录中保留 `pipeline.json`、`preparation.log`、`consumer.log`、
`mooncake-master.log`、`events.jsonl` 和 `checkpoints/step-<更新数>`。
`result.json` 在生产、训练、特征释放及 checkpoint 检查成功后写入。
Ray 日志目录记录在 `pipeline.json` 的 `ray_logs` 字段中。

## 导出 HF 模型

使用 [export_hf.sh](export_hf.sh) 将完整的 TorchTitan DCP checkpoint 导出为
`config.json` 和 `safetensors` 权重。脚本复用
[`export_checkpoint()`](../../../torchtitan/torchtitan/models/dspark_draft/export.py)，
使用 `/tmp/deepspec_vllm_torchtitan_envs/bin/python`，并设置 `CUDA_VISIBLE_DEVICES=''`
在 CPU 上合并分片；不需要启动 Ray、Mooncake 或训练进程。

在项目根目录执行，例如导出已经跑通的第 3 步：

```bash
RUN_DIR=outputs/dspark_pipeline_4plus4_20260916_run4
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/export_hf.sh \
  "$RUN_DIR/checkpoints/step-3" \
  "$RUN_DIR/draft-hf-step-3"
```

两个位置参数均为必填：

1. **checkpoint 目录**：具体的 `step-N` 目录，包含 `commit.json`、`.metadata` 和 DCP 分片。
2. **HF 输出目录**：用于存放导出的模型，建议与原 checkpoint 分开。

相对路径以项目根目录为基准，也可传绝对路径。导出器校验 checkpoint 的 metadata，
严格加载模型权重，输出配置、权重文件及 `export.json`；大模型会生成多个
`model-*.safetensors` 分片及 `model.safetensors.index.json`。

导出精度来自 `commit.json` 中的 `resolved_recipe.checkpoint.export_dtype`。
当前 `step-3` 为 `float32`，脚本沿用该精度；推理加载时可指定 BF16。
同一 checkpoint 的完整导出结果可以重复使用，具体行为由原生导出器处理。

查看参数或仅打印命令：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/export_hf.sh --help

DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/export_hf.sh \
  outputs/dspark_pipeline_4plus4_20260916_run4/checkpoints/step-3 \
  outputs/dspark_pipeline_4plus4_20260916_run4/draft-hf-step-3
```

导出后由 `Qwen3_8DSparkModel.from_pretrained()` 加载该 HF 目录，或将其配置为 vLLM
DSpark 的 `speculative_config.model`。当前 vLLM 分支的 DSpark 推理需要
`VLLM_USE_V2_MODEL_RUNNER=1`。导出脚本已检查语法、dry-run 和 `--help`；本次没有执行
`step-3` 的实际导出及 vLLM 推理验证。

具体实现和验证范围见 [流水线说明](../../../deepspec/pipeline/README.md)
与 [4+4 验收记录](../../../deepspec/pipeline/VALIDATION.md)。
