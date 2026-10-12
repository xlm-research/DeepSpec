# Qwen3.5 Loop/mHC：128K 单 token TTFT 实施与验收

验收日期：2026-10-12。单机八张 H800、TP=8 的实际请求完成 131071 输入加 1 输出；
五次稳态客户端 TTFT 中位数由 **19.687744 秒降至 16.937371 秒**，减少
**2.750373 秒 / 13.97%**，加速比 **1.162×**。两组各卡峰值 allocated 均为
**72.089010 GiB**。本次覆盖的数值对照零差异，融合保持显式选择，默认仍为原 PyTorch 路径。

## 实现与启用

新增 Qwen Loop 专用的 FP32 log-Sinkhorn Triton kernel，将四流路由矩阵的完整
20 轮先行后列归一化及末尾指数融合到一次 kernel 调用。FP32 投影、sigmoid、
BF16 中间转换、四流残差和主干保持原计算路径；三轮 Loop、参数共享和各执行层
独立状态不变。RMS、聚合和残差回写的进一步融合未纳入本次实现。

普通服务设置 `VLLM_QWEN3_5_MHC_BACKEND=triton` 启用，设为 `torch` 回退。
只处理 NVIDIA CUDA、FP32、无梯度、三维四流矩阵、连续的最后两维以及 20 轮的输入；
其他形状、迭代数或梯度需求回退原实现并记录日志。参考路径保留自动求导能力。
实现不依赖 checkpoint 数值，但此次只验收下述 checkpoint、设备及软件组合。

新增基准客户端和可选 worker 诊断扩展。扩展通过既有 worker RPC 和 module hooks
采集数据，不改变常规模型公共 API，也不在普通服务中默认加载。测速前核对有效配置、
worker 实际载入源码的 SHA256 和实际执行后端，避免把旧进程或回退结果算作融合收益。
候选审计失败会恢复 `torch` 并拒绝性能验收。

## 冻结条件与证据身份

| 项目 | 本次取值 |
| --- | --- |
| GPU / 驱动 | 8 × NVIDIA H800，580.95.05；在八卡空闲窗口运行 |
| 系统 / Python | Ubuntu 22.04.5 / Python 3.12.14 |
| Torch / Triton / CUDA | 2.13.0+cu130 / 3.7.1 / 13.0 |
| 并行与调度 | TP=8、PP=1、DP=1，max-num-seqs=1 |
| 上下文 / chunk | 131072 总长度；8192 token 预算，分块 prefill 开启 |
| 缓存 / 执行 | prefix caching 关闭；eager；无草稿推测解码 |
| 精度 | BF16 主干、FP32 router/SSM；TF32=false；原 BF16 reduced-precision reduction=true |
| 显存预算 | gpu-memory-utilization=0.92，两组相同 |
| 采样 | temperature=0、seed=0、max_tokens=1、ignore_eos=true |
| 请求输入 | tokenizer 对固定中文句子编码后重复并截到 131071 个 token |
| 运行顺序 | 同一引擎：参考预热及五次计时 → 独立诊断 → 候选数值审计 → 候选预热及五次计时 → 独立诊断 |

权重目录：

```text
/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16
```

输入 SHA256：`3e084594c7f868b09a5c59d155a4bb1273a0a76f2727ef26cf4b462f376264f7`。
两份 JSON 的输入身份、有效配置、源码身份和 checkpoint 文件清单（名称、大小、mtime）
均相同。两组使用同一个已加载模型，未重载或修改权重。文件清单不是权重内容的密码学哈希。

测量时 vLLM Git 基点为 `cccc81c50da43a4f7cab3a400adf7a7444aa6a77`，实现尚未提交；
JSON 同时记录四个实现/基准文件的 SHA256，各 worker 在初始化时冻结实际导入文件的
SHA256 并与客户端核对。安装包版本字符串仍是历史生成值
`0.26.1rc1.dev719+g1ee54c40d.d20260912`，不能单独用它识别实际执行源码。

原始数据为完整 JSON 的 gzip 压缩版本，可用 `gzip -dc <文件>` 查看：

- [参考路径：配置、五次计时、各 rank 状态和 profiler](../vllm/examples/qwen3.5-loop-mhc/performance/reference.json.gz)
- [融合路径：完整数值审计、计时、状态和 profiler](../vllm/examples/qwen3.5-loop-mhc/performance/triton.json.gz)
- [最终服务启动及运行日志](../vllm/examples/qwen3.5-loop-mhc/performance/server.log.gz)
- [正确性、兼容回归、全套测试和静态检查日志](../vllm/examples/qwen3.5-loop-mhc/performance/validation.log.gz)

## TTFT 与资源结果

客户端从发起 HTTP POST 到收到首个包含生成 token ID 的 SSE choice 计时。
连接/控制事件不结束计时；只有 usage 而没有生成结果会失败；特殊 token 即使文本为空
也按真实 token 计数。每个有效请求还检查 `[DONE]`、`finish_reason=length` 和实际输入/输出计数。
正式计时没有 profiler 或数值采集 hooks。

| 测量 | PyTorch / 秒 | Triton / 秒 |
| --- | --- | --- |
| 预热（不计入统计） | 20.609725 | 16.918677 |
| 1 | 19.683743 | 16.929237 |
| 2 | 19.687744 | 16.948459 |
| 3 | 19.690190 | 16.940658 |
| 4 | 19.689506 | 16.930752 |
| 5 | 19.680670 | 16.937371 |
| 中位数 | 19.687744 | 16.937371 |
| 最小值–最大值 | 19.680670–19.690190 | 16.929237–16.948459 |
| 引擎 TTFT 中位数 | 19.615235 | 16.868655 |
| 引擎 prefill 指标中位数 | 19.606606 | 16.860158 |
| 各卡峰值 allocated / GiB | 72.089010 | 72.089010 |
| 各卡峰值 reserved 范围 / GiB | 73.388672–73.396484 | 73.388672–73.396484 |

全部长请求返回 token ID `110827`。引擎指标来自每次请求前后的 Prometheus histogram
增量，仅计数增量恰为一次时归属于该请求；它们不是纯 GPU 计算时间。
显存峰值在选择后端后重置，包含模型和预分配缓存池，不能解释为单请求 KV 大小。
每 rank 的预热、五次正式请求及一个短请求合计，参考为 `torch=13968, triton=0`，
候选为 `torch=0, triton=13968`，确认候选没有回退。

两组最长请求后都成功完成 5 输入加 1 输出的短请求，均返回 token ID `271`。
短请求 TTFT 分别为 0.427599 和 1.483814 秒，未预热该短形状，也未重复测量，
只用于生命周期回归，不作为短请求性能收益或退化的结论。

启动日志中 TP0 模型加载为 28.501073 秒；引擎初始化（profiling、缓存分配、模型预热）
为 32.32 秒。这是共享引擎的一次启动记录，不是两个后端分别冷启动的对照。
先前正确性测试已填充 JIT 缓存，未隔离测量纯冷 JIT 编译时间。候选审计首次请求的
23.746348 秒包含采集与哈希开销，不能作为纯编译耗时或正式 TTFT。

## 数值和训练兼容验收

首个采用通用 `tl.sum` 归约的候选与原实现最大绝对差为 `2.682209014892578e-07`，
未通过保守零差异门槛并被弃用，没有自行放宽容差。最终实现按当前 Torch CUDA 的
行/列归约顺序组织 FP32 运算，使用 libdevice exp/log，并关闭 FP fusion。
升级 Torch/Triton 或更换设备后应重新验收；有限用例通过不等于所有输入逐位一致的证明。

kernel 对照覆盖 N=1、7、16、8191、8192，尺度 0、1e-7、1、100，含非连续 token 步长；
另覆盖不支持流数/迭代数和自动求导回退。真实 router 测试核对 BF16 分支、post、mix
和原输入保留，没有仅凭输出 dtype 推断精度不变。

同一 TP8 引擎依次执行参考、参考重复、候选三次完整 128K 数值审计。
每个 rank 的每个 chunk 都采集执行层 3、27、51、75、78 的五份特征、独立 post-norm
最终状态和可用目标 logits，记录 shape、dtype、finite 及完整张量原始字节 SHA256。
三次审计各有 16 块，末块最终状态为 `[8191, 4096]`；所有采集张量有限，参考自身
可重复，候选所有哈希一致，生成 token 相同。各候选 rank 实际执行 2304 次融合且零回退。
审计 TTFT 分别为 26.629864、26.438667、23.746348 秒，均不进入性能统计。

既有监督 mask、已有回答来源、五份条件特征顺序、独立最终状态、direct capture
缓冲区复用及 decode 接口通过相应回归。数值 hooks 在审计后移除，普通生成不承担采集开销。

## 独立 GPU 诊断

每组正式计时结束后另发一次长请求，仅 profiling 第一个 8192-token chunk。
下表是八个 rank 的 GPU range 时间中位数，通信一行为实际 all-reduce kernel 时间。

| 阶段 | PyTorch / 毫秒 | Triton / 毫秒 |
| --- | --- | --- |
| mHC router 范围 | 495.989 | 324.590 |
| GDN 范围 | 86.793 | 68.761 |
| Attention 范围 | 23.225 | 23.008 |
| MLP 范围 | 93.393 | 89.105 |
| `multimem_all_reduce_kernel` | 139.785 | 97.748 |

`profile_first_chunk` 同时包含 CPU 关联项和同名 GPU range 项；上表只选择
`loop_stage/*` 中 `self_cpu_us=0` 的 GPU range。范围时间含等待和子算子，通信与
主干范围可能重叠，不能相加成总耗时，也不能混入同名 CPU 关联项或据此声称通信算法被优化。
单次 profiling 有扰动，候选日志出现 profiler 的 `Command Buffer Full` 提示；这份
诊断仅用于定位，不足以估计完整 128K 的 GPU 占比。CPU 排队和调度未单独插桩分解，
客户端与引擎指标差值也不能直接归因于调度。提速结论依据上面的无 profiler 端到端重复测量。

## 可复现命令与检查结果

从 TorchSpec 根目录、在空闲八卡窗口启动：

```bash
source h800conda.sh
VLLM_SERVER_DEV_MODE=1 uv run --no-project --python .venv/bin/python .venv/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --config vllm/examples/qwen3.5-loop-mhc/config.yaml \
  --max-model-len 131072 --host 127.0.0.1 --port 18328 \
  --worker-extension-cls benchmarks.qwen3_5_loop_benchmark_worker.LoopBenchmarkWorker
```

另一终端设置上述 checkpoint 路径，顺序执行：

```bash
loop_checkpoint=/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18328 --model "$loop_checkpoint" \
  --backend torch --profile --output reference.json
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18328 --model "$loop_checkpoint" \
  --backend triton --profile --output triton.json
```

以下 pytest 参数均通过 `uv run --no-project --python .venv/bin/python .venv/bin/python -m pytest`
执行；基准测试一行从 `vllm/` 目录执行，Python 路径相应为 `../.venv/bin/python`。

| 检查 | 参数 / 结果 |
| --- | --- |
| Kernel 数值及回退 | `--confcutdir=vllm/tests/kernels vllm/tests/kernels/test_qwen3_5_mhc.py -q`：22 通过 |
| SSE 与审计失败边界 | `--confcutdir=tests/benchmarks tests/benchmarks/test_qwen3_5_loop_ttft.py -q`：8 通过；先复现失败再修复 |
| 监督、decode、采集、真实 router | `tests/test_dynamic_last_turn_loss.py tests/test_vllm_engine_decode.py tests/test_loop_capture.py -q`：72 通过 |
| 缓冲区复用 | `tests/test_mooncake_connector_finished.py -q`：26 通过 |
| 静态检查 | 变更文件的 Ruff、类型检查和 vLLM pre-commit hooks 通过 |
| 全套 TorchSpec 测试，末尾执行一次 | `tests --maxfail=1 --timeout=120 -q`：69 通过、1 失败、30 subtests 通过，后续未执行 |

相关独立用例合计 **128 项通过**。全套测试失败于既有
`tests/test_capacity_handling.py::TestAsyncTrainingControllerPoolBytes::test_initial_pool_bytes_is_zero`，
全局 Ray mock 导致 `module 'ray' has no attribute '_private'`，退出清理另报
`No module named 'ray.dag'; 'ray' is not a package`。此前仓库历史验收已有相同记录，
本次未修改该无关问题，也不宣称全套通过。

规范与规格两路独立代码审查已完成。修复了 profiler 异常清理、非有限 logits 检查、
实际融合/回退识别及 worker 导入源码身份校验问题；最终两路均无剩余发现。
实现和验证使用了 Codex 辅助，未经远程发布。实验服务已停止，结束时八卡均回到约 4 MiB、
0% 利用率。本结果限于固定重复输入的一次硬件/权重组合，不代表长文本质量、并发吞吐、
多 token decode 或其他 checkpoint 的性能承诺。
