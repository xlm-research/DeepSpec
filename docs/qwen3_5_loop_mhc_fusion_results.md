# Loop-Qwen3.5：128K prefill 的第二阶段融合结果

2026-10-15，单机八张 H800、TP=8，严格 **131072 输入 + 1 输出 token**。
在已经启用 Triton Sinkhorn 的基线上，新增局部融合使五次稳态客户端 TTFT 中位数从
**16.958560 秒降到 14.029082 秒**，减少 **2.929479 秒 / 17.27%**，加速比 **1.209×**。
两组各卡峰值 allocated 均为 **72.689077 GiB**。完整长请求的已覆盖数值对照逐字节一致。

本轮收益只对应下述固定输入、权重和软硬件组合；没有重复计入
[第一阶段 Sinkhorn 融合](qwen3_5_loop_mhc_performance_results.md)的收益。
两阶段输入长度不同，不拼接旧秒数作为新的端到端对照。验收后按用户要求，将 vLLM
Loop 模型默认值设为 Sinkhorn=`triton`、局部融合=`all`；仍可显式覆盖。

## 实现与数值边界

代码位于 [mHC 算子](../vllm/vllm/model_executor/layers/qwen3_5_mhc.py)和
[Loop 模型接入](../vllm/vllm/model_executor/models/qwen3_5_loop.py)。
新增 `VLLM_QWEN3_5_MHC_FUSION=none|norm|pre|merge|all`，当前默认 `all`。
默认选择调整不改变下文显式指定 `none`/`all` 的原始测量条件和证据。

| 操作 | 本次实现 | 保留的边界 |
| --- | --- | --- |
| RMS 归一化 | 融合 BF16 读取、FP32 转换和平方；最后重新读取 BF16 输入并乘 scale，省去 FP32 flat 张量 | 原 Torch mean、epsilon 加法和 rsqrt，以及 FP32 归一化结果 |
| 四流聚合 | 四项按顺序用 FP32 FMA 累积，输出 BF16 | pre 系数先舍入到 BF16；覆盖用例匹配原 BF16 matmul 输出 |
| 残差回写 | 保留原 matmul，融合 post 缩放与相加 | 乘积先舍入到 BF16，再与 BF16 matmul 结果相加并输出 BF16 |

没有改动 FP32 router 投影、BF16 主干、FP32 SSM、三轮 Loop、四条流和完整 20 轮 Sinkhorn。
流展开/收敛只抽成可诊断函数，仍使用原计算。参数共享、各执行层独立状态及训练采集接口不变。
融合只处理支持的 NVIDIA CUDA、BF16 四流输入和布局；不支持的布局、dtype 或梯度需求
保留 Torch 路径并记录回退。独立计数器用于验证实际执行，不能仅凭环境变量认定融合成功。

尝试过把残差矩阵乘也替换为逐项 FMA 的完整融合，但 **未通过零差异门槛，已弃用**：
N=8191/8192、hidden=4096 时，非零尺度样本出现 6–16 个不同元素；尺度 1 时最大绝对差
为 0.015625，尺度 100 时为 1.0。没有放宽容差，也没有将此候选计入性能结果。
归一化同样保留原归约和 rsqrt，避免同时改变其 FP32 结果。

## 测量条件与身份

| 项目 | 固定条件 |
| --- | --- |
| 设备 | 8 × NVIDIA H800，NV8 互联；运行前八卡空闲 |
| 驱动 / 系统 | 580.95.05 / Ubuntu 22.04.5 |
| Python / Torch / Triton / CUDA | 3.12.14 / 2.13.0+cu130 / 3.7.1 / 13.0 |
| 并行 / 并发 | TP=8、PP=1、DP=1；max-num-seqs=1 |
| 长度 / 分块 | max-model-len=131073，8192 token 预算，16 个完整分块 |
| 执行 / 缓存 | eager、V2 runner、关闭 prefix caching、无推测解码 |
| 精度设置 | TF32=false，原 BF16 reduced-precision reduction=true，SSM=float32 |
| 显存预算 | gpu-memory-utilization=0.92，两组同一引擎、同一缓存池 |
| 采样 | temperature=0、seed=0、max_tokens=1、ignore_eos=true |
| 基线 | Sinkhorn=triton，fusion=none |
| 候选 | Sinkhorn=triton，fusion=all |

权重目录：

```text
/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16
```

输入为 tokenizer 对固定中文句子编码后重复并截断得到的 131072 个 token；SHA256：
`e5b4236976efb219cfe00988bde67a379dd3a079b39af646faf47e4a59b1fefe`。
两组的输入哈希、有效配置、checkpoint 文件清单及源码身份完全相同，未重载权重。
权重清单包含文件名、大小、mtime，并非权重内容哈希。

测量时 vLLM Git 基点为 `60bd0552dc99e587777b2b3370ca10608a51b987`，改动未提交。
JSON 记录模型、算子、客户端和 worker 扩展四个文件的 SHA256；每个 worker 在导入时
冻结实际载入文件的哈希，客户端逐项核对。安装包的历史生成版本字符串不能替代源码身份。

运行顺序为：基线重复性审计 → 预热及五次正式请求 → 三次独立分块诊断 →
候选对照审计 → 候选预热及五次正式请求 → 三次独立分块诊断。
正式请求无数值采集或 profiler hooks，诊断结束会恢复函数并移除 hooks。

## TTFT 与显存

TTFT 从客户端提交 HTTP POST 到首个包含生成 token ID 的 SSE choice 计时。
每次核对实际输入/输出计数、结束原因和 `[DONE]`；无生成 token 的连接事件不结束计时。

| 测量 | Sinkhorn 基线 / 秒 | 新增三项融合 / 秒 |
| --- | --- | --- |
| 预热，不计入统计 | 16.960720 | 14.025574 |
| 1 | 16.968384 | 14.023646 |
| 2 | 16.958560 | 14.029082 |
| 3 | 16.975252 | 14.031545 |
| 4 | 16.937570 | 14.030963 |
| 5 | 16.941895 | 14.006419 |
| 中位数 | 16.958560 | 14.029082 |
| 最小值–最大值 | 16.937570–16.975252 | 14.006419–14.031545 |
| 引擎 TTFT 中位数 | 16.894494 | 13.958833 |
| 引擎 prefill 指标中位数 | 16.883804 | 13.952627 |
| 各卡峰值 allocated / GiB | 72.689077 | 72.689077 |
| 各卡峰值 reserved 范围 / GiB | 74.052734–74.060547 | 74.052734–74.060547 |

全部长请求输出 token ID `97431`。引擎指标取每次请求前后的 histogram 增量，
只有计数增量等于 1 才归属于该请求；这些指标不是纯 GPU 时间。
显存峰值在切换路径后重置，包含模型及预分配缓存池，不能解释为单请求 KV 大小。
预热、五次正式长请求及一个短请求合计，每 rank 的 Sinkhorn 和三项操作各调用 13968 次；
候选三项均为 `triton=13968, torch=0`，八个 rank 全部无回退。

长请求后，两组均成功处理 5 输入加 1 输出的短请求，返回 token ID `271`。
短请求分别耗时 1.823050 和 5.238012 秒，包含未预热短形状的 JIT/运行开销；
它只验证长短请求切换，不用于判断短请求稳态速度。

共享引擎 TP0 的模型加载日志为 32.286633 秒，引擎初始化为 46.80 秒；这些都排除在
TTFT 外。候选审计的三次请求为 23.744145、23.620417、20.791242 秒，包含特征搬运、
哈希和首次路径初始化，也不进入统计。已有 kernel 测试填充过 JIT 缓存，本轮没有
隔离测量纯冷 JIT 编译成本。

## 完整请求的零差异验收

同一引擎执行参考、参考重复、候选三次审计。八个 rank 均检查实际 token 位置：
从 0 到 131071 连续覆盖，16 块各 8192 行；文本 mRoPE 的三个位置轴也验证一致。
每块采集执行层 3、27、51、75、78 的五份边界特征，以及全 token 的独立 post-norm
最终状态；另采集可用 logits，记录 shape、dtype、finite 和完整张量原始字节 SHA256。

参考自身可重复，候选与参考所有采集哈希一致，全部张量有限，生成 token 相同。
每个候选 rank 的三项融合各实际执行 2304 次，零 Torch 回退。审计工具会拒绝
缺失分块/特征行、非有限值、错误位置、未执行融合和回退；失败时恢复选定参考路径。
这些结果证明已覆盖样本的一致性，不是所有输入、设备和软件版本逐位一致的保证。

## 前、中、后分块诊断

每个分块使用独立请求诊断，八个 rank 的位置元数据均一致：
chunk 0 为 0–8191，chunk 7 为 57344–65535，chunk 15 为 122880–131071。
下表为八个 rank 的同名 GPU range 时间中位数，单位毫秒，单元格为“基线 → 候选”。

| 阶段 | chunk 0 | chunk 7 | chunk 15 |
| --- | --- | --- | --- |
| 归一化 | 201.278 → 104.873 | 202.591 → 104.868 | 202.625 → 104.873 |
| 四流聚合 | 46.955 → 15.205 | 47.888 → 15.203 | 47.940 → 15.204 |
| 残差回写 | 439.163 → 385.646 | 444.819 → 384.608 | 445.075 → 384.668 |
| router 其余部分，主要为 FP32 投影 | 73.865 → 75.144 | 74.618 → 75.160 | 74.724 → 75.163 |
| Sinkhorn | 1.669 → 1.729 | 1.728 → 1.731 | 1.732 → 1.731 |
| Attention | 22.475 → 22.552 | 47.903 → 47.750 | 76.898 → 76.850 |
| GDN | 68.969 → 68.154 | 67.202 → 94.126 | 66.045 → 65.945 |
| MLP | 87.187 → 87.228 | 87.503 → 87.048 | 87.228 → 87.173 |

只选择 `loop_stage/*` 中 `self_cpu_us=0` 的 GPU 条目；不混入同名 CPU 关联条目。
新增内层命名范围后，外层 `mhc` 条目只显示余下 GPU 活动，不能与旧版仅命名整个 router
的时间直接比较。范围与真实 kernel 条目也不能重复求和。
单次诊断受 profiler 和通信等待扰动，例如候选中段 GDN 增大；没有据此宣称主干或通信算法
发生变化，也不把三个样本外推为完整 128K 的 GPU 时间占比。端到端结论依据无 hooks 的五次计时。

**剩余主要空间在残差矩阵乘。** 候选 TP0 三个采样块的该 BF16 GEMM kernel 累计分别为
357.636、355.366、355.577 毫秒，仍是最大的单类 kernel。
本轮完整融合试验已因数值差异被拒绝；后续若继续，需要匹配其累积/舍入行为并重新审计。
FP32 router 投影、归约及通信亦可单独研究，现有证据不足以给出它们的可实现提速承诺。

8192-token 单算子微基准的中位数加速比分别为：归一化 1.895×、四流聚合 2.748×、
残差回写 1.171×。它们用于局部验证，不替代上面的完整请求收益。

## 复现、检查和原始证据

完整启动与基准命令见 [独立推理 README](../vllm/examples/qwen3.5-loop-mhc/README.md#继续融合与严格-128k-输入)。
普通服务已默认启用两阶段融合，可直接使用：

```bash
bash vllm/examples/qwen3.5-loop-mhc/run.sh --max-model-len 131073
```

将 `VLLM_QWEN3_5_MHC_FUSION=none` 即恢复本次 Sinkhorn 基线。
完整回退原 PyTorch 路径时，再同时设置 `VLLM_QWEN3_5_MHC_BACKEND=torch`。
TorchSpec 的 Ray 环境也透传这两个显式开关。
支持单项开关用于诊断；本轮完整模型性能对照为 `none` 与 `all`，没有把单项微基准
拼成未经实测的单项 TTFT 收益。

已完成以下 157 项测试，全部通过。命令均在 `source h800conda.sh` 后使用
`uv run --no-project --python .venv/bin/python .venv/bin/python -m pytest` 执行；
vLLM 测试从其子目录运行，Python 路径相应使用 `../.venv/bin/python`。

| 范围 | 测试入口 | 结果 |
| --- | --- | --- |
| 算子数值与回退 | `vllm/tests/kernels/test_qwen3_5_mhc.py` | 41 passed |
| 测量工具、覆盖校验和 hook 清理 | `vllm/tests/benchmarks/test_qwen3_5_loop_ttft.py` | 16 passed |
| router、采集和监督兼容性 | `tests/test_loop_capture.py tests/test_dynamic_last_turn_loss.py tests/test_vllm_engine_decode.py tests/test_mooncake_connector_finished.py` | 100 passed |

算子覆盖 N=1、7、8191、8192，尺度 0、1e-7、1、100，完整 hidden=4096，
含非连续系数步长、非连续 residual 回退、其他流数和梯度回退。真实 router 对照
覆盖 N=7/8192 的 branch、post、mix，保留既有采集顺序及独立最终状态测试。
vLLM 本次修改文件的 pre-commit hooks、mypy 3.10/3.12、根目录 router 测试文件的
Ruff 检查均通过。首次 hooks 执行期间同步更新 README，触发文件变化提示；相关 hook
在最终文件上复检通过。最终四个运行/基准文件的哈希仍与正式实测一致。
未运行无关的全仓测试。

原始 JSON 保留逐次结果、配置、源码身份、完整审计和所有 rank 的诊断数据，使用 gzip 压缩：

- [基线](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/reference.json.gz)
- [候选](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/candidate.json.gz)
- [服务日志](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/server_comparison.log.gz)
- [单算子微基准](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/kernel_benchmark.jsonl.gz)
- [被否决的完整残差融合试验](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/probe_mhc.jsonl.gz)
- [kernel 测试日志](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/kernel_tests.log.gz)
- [基准工具测试日志](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/benchmark_tests.log.gz)
- [训练兼容测试日志](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/compatibility_tests.log.gz)
- [静态检查日志](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/precommit.log.gz)、[最终复检](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/precommit_followup.log.gz)、[mypy 3.12](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/mypy312.log.gz)
- [被否决候选的试验代码](../vllm/examples/qwen3.5-loop-mhc/performance/fusion-20261015/probe_mhc.py.gz)

最初诊断工具把文本位置当作一维，遇到实际三轴 mRoPE 输入时报错；修正并增加测试后
重启引擎，上述所有正式数据均来自重启后的同一引擎。该失败启动和试验代码留在
`outputs/loop_mhc_fusion_20261015_k6iN1I/`。正式请求与诊断均结束后停止测试服务；
退出时出现既有 EngineDeadError/共享资源清理日志，八个 GPU worker 已退出并释放显存。
