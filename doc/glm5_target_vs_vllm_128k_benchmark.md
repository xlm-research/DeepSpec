# GLM-5.3-Flash：自定义 target 与 vLLM 的 128k prefill 耗时（历史记录）

**本页只记录 prefill / 特征提取，未测量连续生成多个 token 的 decode 速度。下述 71.47 倍不能作为完整推理速度的结论。**

已补做 [完整自回归生成测量](glm5_generation_128k_benchmark.md)：同一组 GPU 0–3、同一条 128k 输入，各生成 256 个新 token。完整请求耗时为自定义 467.262 秒、vLLM 61.516 秒，耗时比为 7.60 倍；decode 速度分别为 1.709 和 4.482 tokens/s，速度比为 2.62 倍。

日期：2026-09-07。实测脚本：[`benchmark_glm5_target_vs_vllm.py`](../scripts/benchmark_glm5_target_vs_vllm.py)。

本次比较当前工作区中的 `Glm5NextOnlineTarget.forward_training_batch` 与 vLLM 的单请求 prefill。结果文件和日志保存在 [`output/glm5_target_vs_vllm_128k_20260907`](../output/glm5_target_vs_vllm_128k_20260907/)。

**同一组 4 张 B300、同一条 131,072-token 输入，预热后自定义 target 为 322.327 秒，vLLM 为 4.510 秒；当前两条路径的耗时比为 71.47 倍。** 该结果不包含模型初始化，且保留两边现有的精度和输出方式，区别见下表。

## 测试条件

- 模型：GLM-5.3-Flash，完整 45 层，纯文本输入。
- 硬件：同一台机器上的 GPU 0–3，4 × NVIDIA B300 SXM6 AC；两个后端顺序运行。
- 两次启动均显式设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`。自定义进程退出、这 4 张卡显存归零后，才启动 vLLM；卡号、PCI 地址、UUID 及 vLLM 进程的环境核对记录见 [`gpu_verification.json`](../output/glm5_target_vs_vllm_128k_20260907/gpu_verification.json)。
- 输入：仓库现有 `train_data/spec_o3_coldstartsft.repeat60.deepspec.automodel_context128k_1.jsonl` 中唯一一条 packed 样本。按当前 `glm5_next` 训练模板渲染，共 139,207 tokens，截取前 **131,072 tokens**；没有补齐或使用随机 token。
- 两边读取完全相同的 `input_ids.json`，batch size 均为 1。token IDs 的 SHA256（int32 little-endian）为 `9bba5916dcc4e096f94e418fa9aeadf6b5b3ab121a510f702f4b15f7a2a3ca23`。
- Python 环境：`/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs`；PyTorch 2.13.0+cu130，Transformers 5.16.1，FlashInfer 0.6.17。
- vLLM：`0.26.1rc1.dev716+g933876c38`，实际导入本仓库的 `vllm/vllm/__init__.py`，使用默认 V2 Model Runner。
- checkpoint 使用节点本地的已验证完整缓存；初始化时间单独记录，不计入单样本推理。

| 配置 | 自定义 target | vLLM |
| --- | --- | --- |
| 并行 | TP=4、EP=4、DP=1、CP=1，沿用现有 FSDP2 包装 | TP=4，单请求 |
| 精度 | checkpoint FP8 反量化为 BF16 | checkpoint 原生 FP8 |
| 前向 | 当前 bounded full prefill，保留默认 KDA/DSA/EP 分块参数 | chunked prefill，每次最多 8,192 tokens |
| 输出 | 第 2、22、42 层及最终层特征，拷回 CPU | prefill 后生成 1 个 token |
| 缓存 | `use_cache=False` | KV cache 开启，prefix caching 关闭 |
| 编译 | 当前实现，不开启额外 `torch.compile` | `enforce_eager=True` |

采用 4 卡是为了让两个后端都只处理一条样本。自定义 target 的默认 8 卡配置包含两个 DP 分片组；本次使用项目已支持的 4 卡 TP4/EP4 配置。

## 计时范围

每个后端首先完整运行一次 128k，再运行两次相同长度的请求。首次结果包含首次长序列遇到的初始化/JIT 等开销；预热后结果取后两次的中位数。vLLM 自身在引擎初始化期间也会做 kernel warmup，因此“首次 128k 请求”不等于完全冷启动。

自定义 target 在计时前后同步 CUDA，并取 4 个 rank 的最大耗时；包含特征拼接、GPU 到 CPU 的拷贝，不包含 tokenization、checkpoint 加载、特征写盘、draft 训练和 LM head。vLLM 计时从提交 token IDs 到收到生成结果，包含调度、KV cache、最后位置的 LM head 和采样，不包含 tokenization 和引擎初始化；检查每次 `num_cached_tokens == 0`。

这是一组当前实际实现路径的比较。两边的计算精度及输出工作量不同，不能把最终倍数单独归因于推理框架或单个 kernel，也不能将 vLLM 的生成耗时视为完整 target 特征缓存生成耗时。

## 实测结果

| 指标 | 自定义 target | vLLM |
| --- | ---: | ---: |
| 首次完整 128k 请求 | 342.747 s | 9.124 s |
| 预热后第 1 次 | 322.528 s | 4.581 s |
| 预热后第 2 次 | 322.126 s | 4.438 s |
| **预热后中位数** | **322.327 s（5 分 22 秒）** | **4.510 s** |
| 预热后输入吞吐 | 406.64 tokens/s | 29,063.90 tokens/s |

按预热后中位数计算，vLLM 耗时约为当前自定义路径的 **1/71.47**，减少 **98.60%**。每个后端只有两次预热后实测，原始次数和波动完整保留，不提供统计置信区间。

vLLM 每次输入均为 131,072 tokens、输出均为 1 token，三次 `num_cached_tokens` 全部为 0，`is_corrupted` 全部为 false，生成 token ID 均为 77。引擎内从首次调度到首 token 的耗时分别为 8.355、4.565、4.425 秒；主表使用外部 `LLM.generate()` 的完整调用耗时，因此也计入请求提交和结果返回。

自定义 target 两次预热后结果相差 0.125%。每次均完成全部 131,072 个 token，输出特征形状为 `[1, 131072, 12288]` 和 `[1, 131072, 4096]`，并通过抽样有限值检查；每个 rank 拷回 CPU 的 BF16 特征总量为 4 GiB。PyTorch 记录的单卡峰值 allocated 约 174.13 GiB，reserved 约 192.13 GiB（不等于 `nvidia-smi` 的全部进程显存）。

### 单独记录的初始化时间

| 指标 | 自定义 target | vLLM |
| --- | ---: | ---: |
| benchmark 模块入口到模型就绪，包含 Python 导入 | 238.392 s | 535.218 s |
| 模型/引擎构造函数 | 124.365 s | 386.097 s |

自定义的构造函数主要是模型构造和 DCP 加载；其中 DCP 日志为 117.1 秒。vLLM 的构造函数还包括模型架构检查、worker 启动、KV cache 分配和 kernel warmup；其中权重读取为 17.30 秒，worker 的整个模型加载阶段约 58.04 秒。因此这两个构造函数不是相同工作量。此次 vLLM 还执行了一次独立模型架构检查子进程，不能把 535 秒视为每次启动都固定需要的时间。

这些是模块内的阶段计时；自定义端从 rank 0 worker 的模块入口开始，不包含外层 torchrun 启动成本，不作为完整 shell 命令的端到端耗时。

### 自定义 target 的耗时分布

rank 0 使用 CUDA event 记录每个完整 decoder layer，不在层间插入同步。以下为两次预热后运行的平均值：

| 层类型 | 层数 | 单层平均耗时 | 合计耗时 |
| --- | ---: | ---: | ---: |
| 包含 KDA 的层 | 34 | 2.253 s | 76.588 s |
| 包含 DSA 的层 | 11 | 21.796 s | 239.760 s |

11 个 DSA 层约占完整单样本时间的 74.4%，是后续细分 profiling 应优先检查的位置。这些数字包括层内 MLP/MoE、HC 等操作，不能当作 attention 子模块的独立耗时。代码中的 DSA indexer 使用每块 128 个 query 的循环，原生 sparse MLA 使用每块 4,096 个 query；本次保留这些当前默认设置。

### 原始记录和验证

- [汇总结果](../output/glm5_target_vs_vllm_128k_20260907/summary.json)、[输入元数据](../output/glm5_target_vs_vllm_128k_20260907/input_metadata.json)、[代码与环境记录](../output/glm5_target_vs_vllm_128k_20260907/code_metadata.json)。
- [自定义逐次结果及层计时](../output/glm5_target_vs_vllm_128k_20260907/target.json)、[自定义日志](../output/glm5_target_vs_vllm_128k_20260907/target.log)。
- [vLLM 逐次结果及请求计时](../output/glm5_target_vs_vllm_128k_20260907/vllm.json)、[vLLM 日志](../output/glm5_target_vs_vllm_128k_20260907/vllm.log)。

两个 benchmark 命令均以 0 退出，结束后 8 张 GPU 显存均回到 0；vLLM 自带退出流程在 grace period 后终止剩余 worker 的日志发生在全部请求结果保存之后，不计入请求耗时。测试脚本通过 Ruff 检查和格式检查，输入 hash、GPU 数、TP 数、请求缓存数、输出数量和模型源文件 hash 均完成核对。实验新增测试脚本和本报告，未修改模型推理实现。

## 复现命令

在仓库根目录运行；后端顺序执行，等待前一个进程退出后再启动下一个。

```bash
BENCH_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
BENCH_MODEL=/tmp/deepspec-model-cache/glm5-460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883
BENCH_OUTPUT=output/glm5_target_vs_vllm_128k_20260907

"$BENCH_PYTHON" scripts/benchmark_glm5_target_vs_vllm.py prepare \
  --model "$BENCH_MODEL" --output-dir "$BENCH_OUTPUT"

env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  DEEPSPEC_DCP_LOAD_THREADS=8 TOKENIZERS_PARALLELISM=false \
  WANDB_DISABLED=true PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/benchmark_glm5_target_vs_vllm.py target \
  --model "$BENCH_MODEL" --output-dir "$BENCH_OUTPUT" \
  > "$BENCH_OUTPUT/target.log" 2>&1

source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs
env -u VLLM_WORKER_MULTIPROC_METHOD \
  CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u scripts/benchmark_glm5_target_vs_vllm.py vllm \
  --model "$BENCH_MODEL" --output-dir "$BENCH_OUTPUT" \
  > "$BENCH_OUTPUT/vllm.log" 2>&1
```

如果节点本地缓存不存在，将 `BENCH_MODEL` 替换为 `/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash`；加载耗时会受到存储位置影响。再次测试时建议使用新的输出目录保留原始记录。
