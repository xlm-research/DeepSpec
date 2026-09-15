# vLLM 启动慢问题排查与修复报告

## 1. 结论

这次启动慢的主要原因不是 GPU、NCCL，也不是 306 GB 权重读取，而是 GLM-5.3-Flash 的模型导入链在父进程配置阶段提前初始化了 CUDA。

CUDA 提前初始化后，vLLM 为了保证多进程安全，将默认的 `fork` 强制切换为 `spawn`。当前 Python 环境和 vLLM 源码都位于 AFS/FUSE 文件系统上，每个新 Python 进程导入 vLLM、Transformers 和 Conda 包元数据需要约 130～160 秒。EngineCore 和 4 个 TP worker 因此重复支付了 5 次导入成本，而且 worker 基本串行出现，额外消耗接近 12 分钟。

第一阶段修复方式是消除父进程导入阶段的 CUDA 副作用，让 vLLM 在确认 CUDA 尚未初始化时自然使用其默认的 `fork` 路径。没有通过环境变量强制 `fork`，也没有关闭任何模型功能或推理后端。

后续复测发现，若把配置从报告中的 TP=4 改成 TP=8，会引入一组新的 8 卡 kernel 形状，冷启动 profile/JIT 从 92 秒左右增加到 283 秒；同时 InstantTensor 仍需读取完整 306 GB 权重，8 卡不会让 AFS 读取量减半。最终配置恢复为适合单请求 demo 的 TP=4，并优先使用节点上已有且 fingerprint 验证通过的完整模型缓存；缓存无效时自动回退 AFS。

修复后的真实端到端运行结果：

| 指标 | 修复前 | 消除 spawn 后 | 最终本地缓存方案 |
|---|---:|---:|---:|
| 首条 vLLM 日志到开始加载权重 | 13 分 09 秒 | 43 秒 | 约 70 秒 |
| 4 个 TP worker 启动 | 约 10 分钟，依次出现 | 约 1 秒内全部出现 | 约 1 秒内全部出现 |
| 权重读取 | 217.09 秒 | 217.59 秒 | **14.98 秒** |
| 完整命令到生成文本并退出 | 约 22 分钟 | 533.37 秒（8 分 53 秒） | **429 秒（7 分 09 秒）** |

其中，修复前的约 22 分钟由日志可见阶段 19 分 51 秒，加上单独测得的日志前首次导入约 130～148 秒估算得到。533.37 秒和最终 429 秒均为 shell 实际端到端计时，包含模型初始化、实际文本生成和进程退出。

## 2. 问题环境

- vLLM：`0.26.1rc1.dev716+g933876c38`
- 模型：`zai-org/GLM-5.3-Flash`
- 权重大小：约 306 GB，共 62 个 safetensors shard
- 加载格式：`instanttensor`
- 并行配置：TP=4
- GPU：NVIDIA B300 SXM6 AC
- Python：3.12.14
- PyTorch：2.13.0+cu130
- 模型、Conda 环境及 vLLM 源码均位于 `/mnt/afs-agentpro` 的 AFS/FUSE 文件系统

启动命令：

```bash
python vllm/vllm_demo.py
```

## 3. 原始日志时间线

原始日志中的关键时间点如下：

| 时间 | 事件 | 与上一关键事件的间隔 |
|---|---|---:|
| 20:31:38 | 第一条 vLLM 日志 | - |
| 20:32:18 | 检测到 CUDA 已初始化，被迫使用 `spawn` | 40 秒 |
| 20:34:39 | EngineCore 启动 | 2 分 21 秒 |
| 20:37:21 | TP rank 0 启动 | 2 分 42 秒 |
| 20:39:40 | TP rank 1 启动 | 2 分 19 秒 |
| 20:41:50 | TP rank 2 启动 | 2 分 10 秒 |
| 20:44:07 | TP rank 3 启动 | 2 分 17 秒 |
| 20:44:47 | 开始加载模型 | 40 秒 |
| 20:48:41 | 306 GB 权重读取完成 | 约 3 分 37 秒 |
| 20:50:52 | Engine profile/KV cache/warmup 完成 | 约 1 分 41 秒 |
| 20:51:29 | 多模态 warmup 完成 | 37 秒 |

最异常的特征是 4 个 rank 不是同时启动，而是每隔约 130～160 秒依次出现。这与独立测得的 Python 导入耗时完全吻合。

## 4. 根因定位过程

### 4.1 排除权重读取是主要异常

InstantTensor 读取 306 GB 权重耗时约 217 秒，平均吞吐约 1.5 GB/s。修复前后该耗时分别为 217.09 秒和 217.59 秒，基本一致。

因此，权重读取是大模型启动的正常固定成本，但不是这次十几分钟额外等待的来源。

### 4.2 测量 Python 导入成本

在干净 Python 进程中测量得到：

- `import torch`：约 16～17 秒
- `import vllm`：累计约 34 秒
- `from vllm import LLM, SamplingParams`：累计约 130～148 秒

中断一个正在启动的 TP worker 时，调用栈显示它正在执行：

```text
transformers.utils.import_utils
  -> importlib.metadata.packages_distributions()
  -> 遍历 Conda site-packages 元数据
  -> AFS 上大量小文件 stat()
```

这说明 `spawn` 出来的每个进程都在重新扫描 AFS 上的大量 Python 包和元数据。

### 4.3 确认为什么使用了 spawn

vLLM 默认允许使用 `fork`，但如果父进程已经初始化 CUDA，`vllm/utils/system_utils.py` 会为了 CUDA 多进程安全强制改成 `spawn`：

```text
We must use the `spawn` multiprocessing start method.
Reasons: CUDA is initialized
```

问题的因果链如下：

```text
解析 GLM-5.3 模型架构
        |
        v
导入 TileLang / FlashInfer / FLA 平台检测代码
        |
        v
父进程提前调用 torch.cuda.current_device()/get_device_capability()
        |
        v
CUDA context 在创建 EngineCore/TP worker 前已初始化
        |
        v
vLLM 为安全起见将 fork 改成 spawn
        |
        v
EngineCore + 4 个 TP worker 重复导入 Python/Conda 元数据
        |
        v
AFS 小文件访问将每次导入放大到约 130～160 秒
```

### 4.4 捕获第一次 CUDA 初始化调用栈

通过临时包装 `torch.cuda._lazy_init`，分别定位到以下导入副作用：

1. `has_tilelang()` 为了检查 TileLang 是否可用，提前导入了 `flashinfer.comm` 和 `tilelang`。
2. FLA 的 `_check_platform()` 在模块导入时调用 Triton：

   ```text
   triton.runtime.driver.active.get_current_target()
     -> get_current_device()
     -> torch.cuda.current_device()
     -> torch.cuda._lazy_init()
   ```

3. FLA 的 `is_nvidia_hopper` 在模块导入时直接调用：

   ```python
   torch.cuda.get_device_name(0)
   torch.cuda.get_device_capability()
   ```

修复前，完整 `EngineArgs.create_engine_config()` 结束时：

```text
CUDA initialized = True
```

修复后，同样的模型架构解析和配置构建结束时：

```text
CUDA initialized = False
FIRST CUDA INIT hook calls = 0
```

## 5. 具体修复

### 5.1 让 demo 对 spawn 重执行安全

文件：`vllm/vllm_demo.py`

原来在模块顶层导入 vLLM：

```python
from vllm import LLM, SamplingParams
```

现在将导入放入 `main()`：

```python
def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(...)


if __name__ == "__main__":
    main()
```

原因是 Python `spawn` 会用 `__mp_main__` 重新执行入口文件的模块级代码。即使将来因为 Ray、NUMA、WSL 或用户配置再次使用 `spawn`，worker 也不会重复执行 demo 顶层的重型 vLLM 导入。

验证结果：用 `runpy.run_path(..., run_name="__mp_main__")` 模拟 spawn 重执行时，脚本在 0.043 秒内完成，并确认 `vllm` 没有进入 `sys.modules`。

### 5.2 将 TileLang 能力探测改为无副作用检查

文件：`vllm/vllm/utils/import_utils.py`

原来的 `has_tilelang()` 会实际导入 FlashInfer 和 TileLang。现在只使用：

```python
if importlib.util.find_spec("tilelang") is None:
    return False
```

TileLang 的真正导入仍保留在 GPU worker 第一次使用 MHC kernel 的路径中。该路径原本就会先加载 `flashinfer.comm`，再加载 `tilelang`，因此仍然保持防止 `libcudart_stub.so` 干扰 FlashInfer 的必要导入顺序。

同时恢复了原有的 ROCm gfx1250 保护：该架构暂不启用 TileLang，不会因为本次修复扩大后端适用范围。

### 5.3 FLA 平台识别改用 vLLM 的无状态接口

文件：`vllm/vllm/third_party/flash_linear_attention/ops/utils.py`

原来的平台检查通过 Triton 查询当前 GPU，会间接初始化 CUDA。现在优先使用 vLLM 已经解析好的平台类型：

```python
if current_platform.is_cuda():
    return "nvidia"
if current_platform.is_rocm():
    return "amd"
if current_platform.is_xpu():
    return "intel"
```

原来的 SM90+ 判断：

```python
"NVIDIA H" in torch.cuda.get_device_name(0)
or torch.cuda.get_device_capability()[0] >= 9
```

改为：

```python
current_platform.has_device_capability(90)
```

在 NVIDIA 环境中，vLLM 的实现通过 NVML 查询可见 GPU 的 compute capability，不创建 CUDA context。判断结果仍然是 SM90 及以上，所以不会改变 Hopper/Blackwell kernel 的选择。

如果运行在没有 NVML 的特殊 CUDA 平台上，vLLM 的后备实现仍可以调用 PyTorch CUDA 查询；此时 vLLM 会继续安全地切换到 `spawn`，不会强制使用不安全的 `fork`。

### 5.4 没有强制设置 fork

本次没有设置：

```bash
VLLM_WORKER_MULTIPROC_METHOD=fork
```

这是有意为之。若 CUDA 确实已经由用户代码、Ray 或其他依赖初始化，强行 `fork` 可能导致 CUDA/NCCL 错误或死锁。

修复只是让配置导入阶段不再无意初始化 CUDA，之后由 vLLM 自己根据运行状态选择安全的多进程方式。

## 6. 为什么不会影响 vLLM 正常运行

本次没有修改以下配置或行为：

- 模型权重和权重名称映射
- FP8 量化及 MoE 计算路径
- `load_format="instanttensor"`
- `tensor_parallel_size=4`
- `max_model_len=8192`
- `gpu_memory_utilization=0.8`
- `enforce_eager=True`
- NCCL 通信配置
- KV cache 格式和大小
- FlashAttention、FlashInfer、DeepGEMM、TileLang 的实际 kernel 实现
- 模型输入、采样参数和生成逻辑

完整模型运行中确认以下后端仍正常选中：

- NCCL 2.29.7
- TP/EP 通信组
- FlashAttention vision attention
- FlashInfer sparse MLA
- DeepGEMM FP8 linear
- DeepGEMM FP8 MoE
- FlashInfer top-p/top-k sampler
- TileLang MHC 路径

显存行为也保持一致：每个 TP rank 加载约 76.32 GiB 权重，KV cache 仍约为 133.48 GiB。

## 7. 验证结果

### 7.1 CUDA 初始化回归测试

新增：

- `vllm/tests/cuda/scripts/check_glm5next_no_cuda_init.py`
- `vllm/tests/cuda/test_platform_no_cuda_init.py` 中的 GLM-5.3 用例

结果：

```text
1 passed, 14 warnings in 152.36s
```

该测试从干净子进程导入完整 GLM-5.3 模型类，并断言导入前后 `torch.cuda.is_initialized()` 都为 `False`。

### 7.2 TileLang 探测单测

在 `vllm/tests/utils_/test_import_utils.py` 中增加测试，确认：

- TileLang 已安装时返回 `True`，但不会调用 `importlib.import_module()`。
- TileLang 未安装时返回 `False`。
- 原有 `_has_module` 行为没有回归。

结果：

```text
8 passed, 14 warnings
```

### 7.3 静态检查

```text
Ruff check: passed
Ruff format --check: passed
git diff --check: passed
```

### 7.4 完整 306 GB 模型验证

修复后实际运行：

```bash
python vllm/vllm_demo.py
```

关键日志表现：

```text
22:02:02  第一条 vLLM 日志
22:02:19  EngineCore 启动
22:02:28  TP rank 0 启动
22:02:29  TP rank 1/2/3 几乎同时启动
22:02:45  开始加载模型
22:06:27  权重读取完成，217.59 秒
22:07:34  Engine profile/KV cache/warmup 完成
22:08:22  实际文本生成完成并退出
```

总耗时：

```text
TOTAL_SECONDS=533.370
```

进程退出后检查确认：

- 没有残留 EngineCore 或 VllmWorker 进程。
- 8 张 GPU 的显存占用均已恢复为 0 MiB。

## 8. 剩余的正常启动成本

最终方案仍有两块主要耗时：

1. 父进程首次导入约 130～150 秒。
   原因是 Conda 环境和源码都在 AFS 上，Transformers 会扫描大量包元数据。现在这份成本只支付一次，不再由五个子进程重复支付。
2. 节点本地 306 GB 权重读取约 15 秒，后续 FP8/MoE 权重处理约 46 秒。
   如果本地缓存不存在，demo 会回退 AFS，此时权重读取仍约 217～221 秒。

如果还需要进一步缩短到更低，下一步应把 Conda 环境和 vLLM 源码也放到节点本地盘，减少日志前的 AFS 小文件扫描。第一次向新节点复制 306 GB 权重本身仍需要时间，不能计作 cache hit 启动时间。

## 9. 修改文件清单

- `vllm/vllm_demo.py`
- `vllm/vllm/utils/import_utils.py`
- `vllm/vllm/third_party/flash_linear_attention/ops/utils.py`
- `vllm/tests/utils_/test_import_utils.py`
- `vllm/tests/cuda/test_platform_no_cuda_init.py`
- `vllm/tests/cuda/scripts/check_glm5next_no_cuda_init.py`
- `vllm/run.sh`
- `vllm_report.md`

当前修改尚未创建 Git commit。

## 10. TP=8 复测为何仍然慢

复测命令把 `tensor_parallel_size` 从 4 改成了 8，并暴露 8 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u vllm/vllm_demo.py
```

这和报告中 533.37 秒的 TP=4 基准不是同一配置。新日志同时证明第一阶段修复已经生效：8 个 worker 在 22:58:03～22:58:04 的约 1 秒内全部启动，且不再出现 `CUDA is initialized` 和强制 `spawn` 警告。

TP=8 的新增耗时主要来自：

- InstantTensor 仍读取完整 306 GB，耗时 220.94 秒。TP 只改变每卡最终保留的权重，不改变 checkpoint 的总读取量。
- 每卡权重从 76.32 GiB 降到 38.72 GiB，但对 `max_num_seqs=1` 的单请求 demo 没有启动收益。
- TP=8 使用不同的张量与通信形状，触发新的 CUTLASS/Triton kernel 编译；`init engine (profile, create kv cache, warmup model)` 耗时 283.15 秒。
- TP=8 每卡创建约 171 GiB KV cache，对 8192 token、单并发 demo 提供 1120 倍理论并发容量，属于没有实际用途的额外资源配置。

因此，8 卡版本的慢不是旧问题复发，也不是进程卡死，而是使用了和已测基准不同、对当前单请求不合算的并行配置。

## 11. 节点本地模型缓存优化

节点已有训练启动器生成的完整缓存：

```text
/tmp/deepspec-model-cache/glm5-460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883
```

切换前做了以下只读校验：

- AFS 源 checkpoint fingerprint：`460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883`
- 本地 checkpoint fingerprint：相同
- `.deepspec-cache-ready` marker：相同
- safetensors 分片：62/62
- AFS 与本地顶层 72 个文件名称、大小：完全一致
- 2 GiB direct-read 抽样：本地 0.156 秒，AFS 1.225 秒

`vllm_demo.py` 现在仅在 ready marker、config、index 和 62 个分片都存在时选择本地路径；否则自动回退原 AFS 路径，保证缓存缺失时仍可正常启动。

正确 Conda 环境下的完整验证结果：

```text
[vllm-demo] model=/tmp/deepspec-model-cache/glm5-460f95... tp=4
Loading weights took 14.98 seconds
Model loading took 76.32 GiB and 61.17 seconds
init engine (profile, create kv cache, warmup model) took 92.10 s
[vllm-demo-test] exit=0 total_seconds=429
```

生成文本成功，NCCL、FlashAttention、FlashInfer、DeepGEMM、FP8 MoE 等后端保持启用；4 个 worker 全部优雅退出，8 张 GPU 最终显存均恢复为 0 MiB。

## 12. 最终启动方式

推荐直接使用已经固化环境、GPU 数量和安全多进程设置的脚本：

```bash
cd /mnt/afs-agentpro/lezewei/DeepSpec
bash vllm/run.sh
```

等价的手工命令是：

```bash
cd /mnt/afs-agentpro/lezewei/DeepSpec
source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs
env -u VLLM_WORKER_MULTIPROC_METHOD \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -u vllm/vllm_demo.py
```

不要再把 demo 改成 TP=8 后与 TP=4 的 8 分 53 秒基准直接比较。当前节点存在本地模型缓存时，新实测目标是约 7 分 09 秒；若 `/tmp` 缓存随节点重建而消失，demo 会安全回退 AFS，权重读取会恢复到约 221 秒。
