# GLM-5.3-Flash：单机 8 卡的 DSpark 特征生成对比

日期：2026-09-07。两边推理、文件保存及全量数值比较均已完成，成功 benchmark 和比较程序的退出码均为 0。

本次比较同一条 128k 样本的 target 特征生成和保存：两边输出相同指定层、相同 token 位置和相同布局的 hidden states。这里测量 DSpark 训练获取特征的整段前向，包含 CPU 拷贝和文件保存。

**结果：同一台机器的同一组 8 张卡上，第 2 次完整请求自定义耗时 235.36 秒，vLLM 耗时 21.19 秒，包含保存的总流程相差 11.10 倍；每份文件均约 4.00049 GiB。文件布局一致，但中后层数值有明显差异，两边各自重复运行也存在差异。** 这是当前 BF16 自定义实现与原生 FP8 vLLM 的实测结果，不能据此声称两边特征数值等价。

## 速度结果

| 完整请求 | 自定义：前向与 CPU 拷贝 | 自定义：保存与 fsync | 自定义：总耗时 | vLLM：前向与 CPU 拷贝 | vLLM：保存与 fsync | vLLM：总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第 1 次 | 227.317 s | 10.905 s | 237.998 s | 11.486 s | 9.527 s | 21.201 s |
| 第 2 次 | 225.825 s | 9.927 s | 235.358 s | 11.362 s | 9.519 s | 21.194 s |

**以第 2 次完整请求计算：vLLM 的前向＋CPU 特征拷贝速度为自定义的 19.88 倍；包含文件保存的总流程速度为 11.10 倍。** 完整流程输入吞吐分别为自定义 **556.91 tokens/s**、vLLM **6,184.26 tokens/s**。两边保存耗时接近，约 9.5–9.9 秒；这里只反映本次 AFS 目录及文件格式。

两边四份完整文件均为 **4,295,492,104 bytes（4.00049 GiB）**。vLLM 每个请求的 prefix cache 命中数为 0，8 个 rank 都确认捕获全部 131,072 tokens，并逐块核对输入 IDs 和位置。16 个连续块完整覆盖 `[0, 131072)`。

两个成功 benchmark 的主进程退出码均为 0。vLLM 在结果保存后回收 worker 时仍有 SIGTERM / SIGKILL 等清理警告，保留在日志中，不计入请求耗时。

每个后端只测量了两次完整请求，上表保留全部结果；没有统计置信区间。这里的 tokens/s 是输入特征生成吞吐，不能作为连续生成的 decode tokens/s。

## 保存的 hidden states：跨实现数值比较

以 `target_run_1.safetensors` 为 reference、`vllm_run_1.safetensors` 为 candidate。两边 metadata、shape、dtype 和全部 input IDs 一致；每组特征比较 **131,072 × 4,096 = 536,870,912** 个值，共比较 **2,147,483,648** 对特征值。所有值均有限。

| 特征（0-based） | 逐 token 平均余弦 | MAE | RMSE | 相对 L2 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 层 2 | 0.999655 | 0.000335 | 0.000430 | 2.624% | 0.005859 |
| 层 22 | 0.923879 | 0.012277 | 0.023676 | 18.267% | 66.000000 |
| 层 42 | 0.903675 | 0.152825 | 0.246793 | 26.053% | 272.000000 |
| 最终归一化层 | 0.885970 | 0.417509 | 0.653419 | 49.727% | 12.453125 |

层 2 很接近，但不是逐元素相等。中后层差异明显，且误差分布不均匀：最终归一化层逐 token 余弦的中位数为 **0.958990**，1% 分位数为 **0.317216**，最小值为 **0.110105**。层 22 和层 42 的最大绝对误差远大于各自 RMSE，不能仅凭平均指标忽略极端值。

上述差异同时受到 BF16/FP8 计算精度、算子、并行策略以及重复运行波动影响，本实验未分离各项贡献，也未通过 DSpark 训练效果验证其可替代性。完整指标和文件 SHA256 见 [跨实现比较 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/comparison.json)，每 4,096 个 token 的分段指标见 [CSV](../output/glm5_hidden_states_8gpu_128k_20260907/comparison.csv)。

## 同一实现重复运行的差异

以下数据分别来自两个后端 rank 0 两份完整 128k 文件的全量比较。各自的模型权重和输入不变，但两边中后层特征均出现明显的重复运行波动；本次尚未定位其根因。

| 特征（0-based） | 自定义：逐 token 平均余弦 | 自定义：相对 L2 | vLLM：逐 token 平均余弦 | vLLM：相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| 层 2 | 1.000000 | 0.000% | 1.000000 | 0.000% |
| 层 22 | 0.962294 | 9.049% | 0.951868 | 9.273% |
| 层 42 | 0.940826 | 19.058% | 0.927406 | 20.255% |
| 最终归一化层 | 0.929451 | 39.152% | 0.912582 | 43.595% |

两边各自的层 2 均逐元素数值相等，其余三组特征不完全相等；所有比较值均有限。自定义最终层的逐 token 余弦中位数为 0.979472，1% 分位数为 0.409373，最小值为 0.151496；vLLM 对应值为 0.974662、0.358938、0.116468。全局余弦会受到高范数 token 的较大影响，因此主表使用逐 token 余弦的平均值。

原始统计见 [自定义重复运行 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/target_repeat_comparison.json) / [CSV](../output/glm5_hidden_states_8gpu_128k_20260907/target_repeat_comparison.csv)，以及 [vLLM 重复运行 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/vllm_repeat_comparison.json) / [CSV](../output/glm5_hidden_states_8gpu_128k_20260907/vllm_repeat_comparison.csv)。

## 实验条件

- 同一节点、同一组 GPU 0–7，8 × NVIDIA B300 SXM6 AC，每卡 275,040 MiB。两个后端顺序运行，并核对实际进程占用的 8 个物理 GPU UUID 完全相同，见 [自定义 GPU 记录](../output/glm5_hidden_states_8gpu_128k_20260907/gpu_target.json) 和 [vLLM GPU 记录](../output/glm5_hidden_states_8gpu_128k_20260907/gpu_vllm.json)。
- 两边 TP=8、DP=1、CP=1、batch size=1；自定义 target 使用 EP=8，vLLM 使用 TP 切分专家。这是单请求实验；自定义训练配置默认的 TP4、DP-shard2 不用于本次计时。
- 同一 GLM-5.3-Flash checkpoint，完整 45 层、纯文本模式，使用已验证的节点本地模型缓存。
- 输入沿用前次的同一条实际 packed 样本，截取前 131,072 tokens。两边读取同一个 `input_ids.json`，文件 SHA256 为 `e4c3c2cdd0fd0780b2279b072ea773e02cd4dab29a09f81f7fb84100085be9db`。来源和渲染方式见 [输入元数据](../output/glm5_target_vs_vllm_128k_20260907/input_metadata.json)。
- 指定 decoder 层为 **[2, 22, 42]，从 0 开始计数**。同时保存 DSpark 所需的最终归一化特征。
- 自定义 target 将 checkpoint 的 FP8 权重反量化为 BF16 计算；vLLM 使用 checkpoint 原生 FP8。两边**保存的特征均为 BF16**。数值差异包含精度、算子及并行执行路径的差异。
- 两边先用前 8,192 tokens 做特征生成、保存预热，再各执行两次完整 128k 请求，两次都保存文件。第二次作为完整长度预热后的结果，两次原始耗时均保留。
- 软件：PyTorch `2.13.0+cu130`，vLLM 包版本 `0.26.1rc1.dev716+g933876c38`；实际 vLLM checkout 为 `a3c6cf9cec730e47a7097461279594c1ce4b5c19`。本仓库基于 `932be431ee8be510a6c238fd19bbc24242f22cd7` 的当前工作区运行，相关源文件 SHA256 保存在实验元数据中。

## 提取位置与文件布局

自定义端直接调用现有 `Glm5NextOnlineTarget.forward_training_batch`，`use_cache=False`，用 hook 捕获 decoder 层输出。GLM 的四路 HC 状态按现有 `hc_head` 求均值，得到每个 token 的 4,096 维特征；最终层另经过输出 RMSNorm。

vLLM 的 GLM 模型已有辅助 hidden-state 接口，其层编号含 embedding 偏移，因此映射为 **[3, 23, 43]**。该接口先完成对应层延迟执行的 HC post，再对 HC 状态求均值。导出 hook 收集这三层辅助输出和最终归一化输出，同时把正常的最终输出继续交还引擎。

导出通过 `worker_extension_cls` 注册的具名方法控制，RPC 只传路径、长度等参数。导出器通过 CPU 合成数据校验，覆盖分块拼接、层映射、文件内容、输入 IDs、正常模型返回值以及位置缺口检测。早先一次 vLLM 尝试因 Python 函数 RPC 序列化限制而在提交请求前结束，该次不进入速度表。

vLLM 现成的 `extract_hidden_states` 导出模式不支持 chunked prefill。本次通过已有模型辅助输出逐块收集特征，保留 8,192-token chunked prefill；检查每块绝对位置连续、无重叠、无缺失，直到覆盖全部 131,072 个 token。prefix caching 关闭。使用 `max_tokens=1` 完成请求，保存范围仅为输入 token；没有逐 token 的续写 decode。

两边都在全部 8 个 TP rank 上把特征拷到 CPU，由 rank 0 保存一份相同格式的 `.safetensors` 文件，避免保存 8 份重复文件。

| 张量 | 磁盘 shape | dtype | 数据量 |
| --- | --- | --- | ---: |
| `target_hidden_states` | `[131072, 12288]` | BF16 | 3 GiB |
| `target_last_hidden_states` | `[131072, 4096]` | BF16 | 1 GiB |
| `input_ids` | `[131072]` | INT32 | 512 KiB |

`target_hidden_states` 的最后一维按 `[层2的4096维, 层22的4096维, 层42的4096维]` 排列，可以 reshape 为 `[131072, 3, 4096]`。每份文件的特征 payload 为 4 GiB，另含 token IDs 和少量文件头元数据。

## 计时范围

- **前向与 CPU 特征拷贝**：模型已加载后，从提交 token IDs 到完整特征在 CPU 可用。自定义端同步 CUDA 并取最慢 rank；vLLM 使用 `LLM.generate()` 的外部耗时，也包含末位置的 LM head 和选出 1 个 token。
- **保存**：同一 AFS 输出目录、相同 safetensors 格式，计时包含保存函数和文件 `fsync`。
- **总耗时**：请求开始到保存完成，自定义端等待所有 rank，vLLM 等待 worker 导出 RPC 返回。导出控制开销计入总耗时。
- 自定义分项取跨 rank 的最大耗时，rank 0 的保存可与其他 rank 的前向收尾重叠，因此分项之和不要求等于总耗时。
- 模型导入、权重加载、引擎初始化、输入分词以及离线数值分析不计入上述耗时。
- vLLM 使用 eager 模式，`max_num_seqs=1`、`max_num_batched_tokens=8192`、`gpu_memory_utilization=0.8`，chunked prefill 开启、prefix caching 关闭。

## 数值对比方法

比较第二次完整请求实际保存的文件，首先核对 metadata、张量 shape、dtype 和所有 input IDs。然后按 token 块遍历全部特征值，分别报告三层和最终层的 MAE、RMSE、最大绝对误差、相对 L2 误差及逐 token 余弦相似度。

同时比较自定义两次完整运行、vLLM 两次完整运行各自的保存结果，以观察重复运行波动，避免把全部跨后端差异直接归因于 FP8。

跨实现相对 L2 定义为 `||vLLM - 自定义||₂ / ||自定义||₂`；同一实现重复运行时，以第一次完整请求为 reference、第二次为 candidate。BF16 值转为 FP32 计算差值，统计归约使用 FP64。余弦统计同时记录均值、中位数、最小值和 1% 分位数。额外记录逐元素数值相等比例及 `atol=0.01, rtol=0.01` 覆盖比例；这两个阈值用于描述差异，不代表训练效果验收标准。

首次 8k 预热已完成前向和保存，但额外的跨 TP rank 逐 bit 相等断言失败。该次不进入速度表；原始日志和文件保留在 `attempt1_exact_rank_check/`。后续检查改为对固定 token/维度样本量化 rank 间差异；它与跨后端的全量特征比较分别记录。

成功运行中的 rank 间检查仅采样 16 个 token 位置及每隔 128 维的特征。自定义样本存在 rank 间差异，vLLM 样本一致；这不代表 8 个 rank 的全量特征已逐一核对。主报告比较的文件均由 rank 0 保存。

## 复现与原始记录

- [实测脚本](../scripts/benchmark_glm5_hidden_states.py)、[全量数值比较脚本](../scripts/compare_glm5_hidden_states.py)。
- [结果汇总 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/summary.json)、[实验元数据](../output/glm5_hidden_states_8gpu_128k_20260907/metadata.json)。
- 自定义 [计时与 rank 检查 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/target.json) / [日志](../output/glm5_hidden_states_8gpu_128k_20260907/target.log)；vLLM [计时与分块捕获 JSON](../output/glm5_hidden_states_8gpu_128k_20260907/vllm.json) / [日志](../output/glm5_hidden_states_8gpu_128k_20260907/vllm.log)。
- 第 2 次实际特征文件：[自定义](../output/glm5_hidden_states_8gpu_128k_20260907/target_run_1.safetensors)、[vLLM](../output/glm5_hidden_states_8gpu_128k_20260907/vllm_run_1.safetensors)。第 1 次文件也保留在相同目录。
- [数值指标校验日志](../output/glm5_hidden_states_8gpu_128k_20260907/comparison_metric_check.log)、[导出器 CPU 校验日志](../output/glm5_hidden_states_8gpu_128k_20260907/exporter_smoke_test.log)。
- 原始文件目录：`output/glm5_hidden_states_8gpu_128k_20260907/`。

在仓库根目录运行，两个后端顺序执行，使用新的输出目录保留本次原始数据：

```bash
BENCH_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
BENCH_MODEL=/tmp/deepspec-model-cache/glm5-460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883
BENCH_INPUT=output/glm5_target_vs_vllm_128k_20260907/input_ids.json
BENCH_OUTPUT=output/glm5_hidden_states_8gpu_128k_rerun
mkdir -p "$BENCH_OUTPUT"

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  DEEPSPEC_DCP_LOAD_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/benchmark_glm5_hidden_states.py target \
  --model "$BENCH_MODEL" --input-ids "$BENCH_INPUT" \
  --output-dir "$BENCH_OUTPUT" --repeats 2 \
  > "$BENCH_OUTPUT/target.log" 2>&1

source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs
env -u VLLM_WORKER_MULTIPROC_METHOD \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  "$BENCH_PYTHON" -u scripts/benchmark_glm5_hidden_states.py vllm \
  --model "$BENCH_MODEL" --input-ids "$BENCH_INPUT" \
  --output-dir "$BENCH_OUTPUT" --repeats 2 \
  > "$BENCH_OUTPUT/vllm.log" 2>&1

OMP_NUM_THREADS=8 "$BENCH_PYTHON" -u scripts/compare_glm5_hidden_states.py \
  --reference "$BENCH_OUTPUT/target_run_1.safetensors" \
  --candidate "$BENCH_OUTPUT/vllm_run_1.safetensors" \
  --output "$BENCH_OUTPUT/comparison.json"

for BENCH_BACKEND in target vllm; do
  OMP_NUM_THREADS=8 "$BENCH_PYTHON" -u scripts/compare_glm5_hidden_states.py \
    --reference "$BENCH_OUTPUT/${BENCH_BACKEND}_run_0.safetensors" \
    --candidate "$BENCH_OUTPUT/${BENCH_BACKEND}_run_1.safetensors" \
    --output "$BENCH_OUTPUT/${BENCH_BACKEND}_repeat_comparison.json"
done
```
