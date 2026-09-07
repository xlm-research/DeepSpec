# GLM-5.3-Flash：128k 输入后的完整自回归生成

日期：2026-09-07。本次实际生成多个新 token，分别记录首 token、后续 decode 和请求总耗时。此前的 [prefill / 特征提取结果](glm5_target_vs_vllm_128k_benchmark.md) 不代表完整生成速度。

**同一组 4 张 B300、同一条 131,072-token 输入，两边均实际生成 256 个新 token：自定义 target 总耗时 467.262 秒，vLLM 为 61.516 秒。完整请求的耗时比为 7.60 倍；后续 decode 的速度比为 2.62 倍。**

| 指标 | 自定义 target | vLLM |
| --- | ---: | ---: |
| 实际输入 / 输出 | 131,072 / 256 tokens | 131,072 / 256 tokens |
| 首 token（TTFT） | 318.075 s | 4.620 s |
| 后续 255 步 decode | 149.184 s | 56.897 s |
| Decode 速度 | 1.709 tokens/s | 4.482 tokens/s |
| 平均 decode 每 token | 585.036 ms | 223.127 ms |
| **完整请求总耗时** | **467.262 s（7 分 47 秒）** | **61.516 s（1 分 2 秒）** |

自定义端所有缓存检查通过，最终缓存长度为 131,327（最后一个输出 token 尚未作为输入再次送入模型）。PyTorch 记录的各 rank 峰值 allocated 显存最大为 197.105 GiB；进程以 0 退出并释放 GPU。

vLLM 返回的请求已结束，`finish_reason="length"`，实际生成数量及 `metrics.num_generation_tokens` 均为 **256**，`num_cached_tokens=0`，`is_corrupted=false`。两边生成的前 **217 个 token 完全一致**，之后出现分叉；本次测量生成速度，不据此声称跨引擎数值等价或生成质量相同。

vLLM 在结果保存后进入资源回收，日志包含等待 worker 超时后发送 SIGTERM / SIGKILL，以及 shared memory 回收警告。这些发生在推理完成之后，不计入请求耗时。主进程最终以 **0** 退出，所有 GPU 显存归零，见 [结束时 GPU 记录](../output/glm5_generation_128k_256_20260907/after_both_backends_gpu.csv)。这轮推理请求完成了，但退出清理仍有警告。

## 相同的输入与硬件

- 两边顺序使用同一组物理 **GPU 0、1、2、3，4 × NVIDIA B300 SXM6 AC，TP=4，batch size=1**。自定义 target 的 EP=4、DP=1、CP=1。
- 已比对两个后端进程占用的 GPU UUID，四张卡逐一相同：[自定义进程记录](../output/glm5_generation_128k_256_20260907/gpu_target.json)、[vLLM 进程记录](../output/glm5_generation_128k_256_20260907/gpu_vllm.json)。两次运行之间 GPU 显存归零，见 [切换时记录](../output/glm5_generation_128k_256_20260907/between_backends_gpu.csv)。
- 同一个 GLM-5.3-Flash checkpoint，完整 45 层。模型使用已验证的节点本地缓存。
- 同一条实际 packed 样本的前 **131,072 个 token**，读取同一个 `input_ids.json`；JSON 文件 SHA256 为 `e4c3c2cdd0fd0780b2279b072ea773e02cd4dab29a09f81f7fb84100085be9db`。样本来源与截断方式见 [输入元数据](../output/glm5_target_vs_vllm_128k_20260907/input_metadata.json)。
- 两边都用 greedy decoding，固定生成 **256 个新 token**，忽略 EOS。总请求长度为 131,328 tokens；首个输出来自 prefill，随后执行 **255 次单 token decode**。
- 每个后端先执行 128-token 输入、4-token 输出的短请求预热，再执行一次完整测量。每次使用新缓存；vLLM 关闭 prefix caching，检查完整请求的 `num_cached_tokens == 0`。这里没有完整 128k 预热后的重复测量或统计中位数。

## 自定义 target 如何生成

原有 `Glm5NextOnlineTarget` 面向训练特征提取：bounded attention 原先拒绝 KV cache，且使用的 `AutoModel` 没有 LM head。为完成真实生成，本次在原有 bounded prefill 中保存 KDA 的卷积状态、递归状态，以及 DSA 的 K/V 和 indexer 状态；之后使用同一模型的 Transformers 原生单 token 缓存解码路径。另从同一 checkpoint 读取 `lm_head.weight`，按词表分成 4 份并行计算和 greedy 选词。

每个输出 token 都来自上一步输出的自回归续写。没有在 decode 时重新计算完整 128k 输入，也没有提取、拼接或拷回训练中间层特征。当前缓存续接入口支持单 token decode。

[数值测试](../tests/test_glm5_generation_cache.py) 在包含 KDA 和 DSA 的小模型上，验证 73-token prefill 及后续 5 次缓存续接与完整前向重算一致，也检查缓存长度、状态有限值和无缓存路径的一致性；测试通过。完整模型的测试脚本还检查所有层的缓存状态及实际输出数量。

## 计时与配置差异

- **TTFT**：请求开始到首个输出 token，主要包含 prefill，也包含首个 token 的 LM head / 选词。
- **Decode**：从首个输出到第 256 个输出，共 255 步；tokens/s 用 `255 / decode 秒数` 计算。
- **总耗时**：完成整条请求的实际时间，包含 prefill 和 decode，排除 Python 导入、模型加载、引擎初始化、输入 tokenization 和最终文本解码。自定义端同步 CUDA 并记录最慢 rank；vLLM 总时间使用 `LLM.generate()` 外部计时，TTFT 与 decode 使用请求内部指标。
- 内部指标与外部计时的边界不同，子项相加与总耗时存在毫秒级差异。首次长输入触发的运行期 JIT 包含在本次请求耗时中。
- 自定义 target 保持现有 **BF16** 参数计算；vLLM 使用 checkpoint 的原生 **FP8**。这比较的是当前实现路径，不能把差异完全归因于框架。
- vLLM 使用 `enforce_eager=True`、`max_num_seqs=1`、`max_num_batched_tokens=8192`、chunked prefill 开启、prefix caching 关闭、`gpu_memory_utilization=0.8`，没有进行额外性能调参。

## 原始记录

- [汇总结果与倍数](../output/glm5_generation_128k_256_20260907/summary.json)。
- [实测脚本](../scripts/benchmark_glm5_generation.py)、[代码与环境元数据](../output/glm5_generation_128k_256_20260907/metadata.json)、[缓存数值测试日志](../output/glm5_generation_128k_256_20260907/cache_correctness_test.log)。
- 自定义：[结果及生成的 token / 文本](../output/glm5_generation_128k_256_20260907/target.json)、[运行日志](../output/glm5_generation_128k_256_20260907/target.log)。
- vLLM：[结果及生成的 token / 文本](../output/glm5_generation_128k_256_20260907/vllm.json)、[运行日志](../output/glm5_generation_128k_256_20260907/vllm.log)。

## 复现命令

在仓库根目录顺序执行两个后端，确保自定义进程退出并释放 GPU 后再启动 vLLM。重跑时使用新的输出目录，保留原始测量。

```bash
BENCH_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
BENCH_MODEL=/tmp/deepspec-model-cache/glm5-460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883
BENCH_INPUT=output/glm5_target_vs_vllm_128k_20260907/input_ids.json
BENCH_OUTPUT=output/glm5_generation_128k_256_20260907
mkdir -p "$BENCH_OUTPUT"

env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  DEEPSPEC_DCP_LOAD_THREADS=8 TOKENIZERS_PARALLELISM=false \
  WANDB_DISABLED=true PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/benchmark_glm5_generation.py target \
  --model "$BENCH_MODEL" --input-ids "$BENCH_INPUT" \
  --output-dir "$BENCH_OUTPUT" --new-tokens 256 \
  > "$BENCH_OUTPUT/target.log" 2>&1

source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs
env -u VLLM_WORKER_MULTIPROC_METHOD \
  CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u scripts/benchmark_glm5_generation.py vllm \
  --model "$BENCH_MODEL" --input-ids "$BENCH_INPUT" \
  --output-dir "$BENCH_OUTPUT" --new-tokens 256 \
  > "$BENCH_OUTPUT/vllm.log" 2>&1
```
