# GLM-5.3-Flash FSDP2 启动日志与权重加载优化报告

日期：2026-09-03
目标脚本：[`scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh`](scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh)

## 1. 结论

这次修改同时解决了两个问题：启动阶段缺少可定位的日志，以及 GLM-5.3-Flash 权重加载过慢。

在当前单机 8 GPU、target TP=4、EP=8 的环境中，原始 target FSDP2 DCP 加载耗时为 299.0 秒；加入节点本地模型缓存、快速 safetensors 元数据解析、向量化 FP8 反量化并把 reader 并发调到实测最优的 8 线程后，加载耗时降到 69.0 秒，缩短 76.9%，速度约为原来的 4.33 倍。

此外，原来 draft 模型会在每个 rank 上先构造全部 288 个 routed experts，再切成 36 个本地 experts。这不仅慢，启动期间还采样到约 45～53 GiB/rank 的 GPU 占用和约 60～94 GiB/rank 的 CPU RSS。现在从构造阶段就只分配本 rank 的 36 个 experts，实测 draft 初始化为 86.8 秒；采样观察到 GPU 占用约 6.5～7.7 GiB/rank、CPU RSS 约 20 GiB/rank。

## 2. 基线与实测结果

模型 checkpoint 包含 62 个 safetensors 分片，目录约 306 GiB；索引中共有 76,108 个 tensor 条目，其中 37,338 个是 FP8 scale tensor。以下结果均来自当前机器上的真实 8 GPU 运行，不是空模型或小 checkpoint 推算。

| 指标 | 修改前 | 修改后 | 变化 |
| --- | ---: | ---: | ---: |
| target FSDP2 DCP 加载 | 299.0 s | 69.0 s（8 threads） | -76.9%，4.33× |
| 同一优化实现下的 target 加载 | 107.5 s（4 threads） | 69.0 s（8 threads） | -35.8%，1.56× |
| launch 到 `Running training` | 672 s | 258 s（4-thread 完整运行） | -61.6%，2.60× |
| draft routed experts/rank | 288 个先构造再切分 | 直接构造 36 个 | 参数分配量降为 1/8 |
| 节点本地 cache 首次构建 | 无 | 285 s，一次性 | 后续启动直接 cache hit |

对应证据：

- [修改前基线日志](output/glm5_3_flash_dspark_fsdp2/logs/node_rank_0.log)：20:09:29 启动，target 加载 299.0 秒，20:20:41 进入训练。
- [优化后的完整一步训练日志](output/glm5_load_benchmark_fast2/logs/node_rank_0.log)：本地 cache hit，draft 初始化 86.8 秒，target 加载 107.5 秒，完整 forward/backward/optimizer 成功，最终 `exit_code=0`。
- [8-thread target 加载日志](output/glm5_load_benchmark_threads8/logs/node_rank_0.log)：`reader_threads=8`，target 加载 69.0 秒，其中 `read_and_dequantize=68.9s`、`metadata_scan=1.6s`。该诊断任务在取得目标阶段数据后被主动停止，没有继续做无关的训练与保存。
- [首次本地缓存构建日志](output/glm5_load_benchmark_fast/logs/node_rank_0.log)：328,366,197,468 bytes，8 个分片复制 worker，285 秒完成。

需要注意：258 秒是采用 4 个 DCP reader threads 的完整端到端实测；69 秒是随后单独进行的 8-thread target 阶段实测。不能把两者拼成一个声称已经实测过的端到端数字。

## 3. 原实现为什么慢

### 3.1 共享 AFS 被所有 rank 同时随机读取

原来 8 个 rank 都直接从共享 checkpoint 目录读取自己需要的 tensor。GLM checkpoint 很大、tensor 数量很多，访问模式又不是一次连续顺序读，因此共享 AFS 同时承受大量交错的小范围读取。

对当前存储做的并发读取测试中，单路吞吐约 0.12 GiB/s，8 路约 1.07 GiB/s；继续增加到 16/32 路后反而回落到约 0.95 GiB/s。这说明增加并发有用，但不能无限加线程，8 路是当前机器更合理的拐点。

### 3.2 通用 DCP reader 的元数据扫描成本过高

PyTorch 通用 `QuantizedHuggingFaceStorageReader` 会对索引中的每个 tensor 多次调用 safetensors `get_slice` 来取得 shape/dtype。对 76,108 个条目而言，Python 调用和反复打开/查询分片的固定成本很高；独立诊断中旧扫描路径运行超过 40 秒仍未完成。

GLM 的这些 Hugging Face 分片本身不是 DCP-sharded 文件，shape、dtype 和 data offsets 已经完整写在每个 safetensors 文件头中。因此逐 tensor 调 `get_slice` 没有必要。

### 3.3 FP8 block 反量化存在 Python 内层循环

checkpoint 使用 128×128 block FP8 量化。PyTorch 通用 reader 会在 Python 中逐 block 执行反量化。一个典型的 2048×4096 expert matrix 有 512 个 block，而每个 rank 要加载数千个 expert matrix，Python 循环成本被大量放大。

在当前机器上对典型矩阵做的微基准中，通用 block 循环中位数约 0.156565 秒，向量化实现约 0.080081 秒，并且 BF16 输出逐 bit 相同。

### 3.4 draft expert 在切 EP 之前被重复构造

旧流程是：

1. 每个 rank 构造包含全部 288 个 experts 的 draft 模型；
2. 整个模型搬到 GPU；
3. parallel adapter 再按 EP=8 切成每 rank 36 个 experts。

也就是说，每个 rank 都为最终不会保留的 252 个 experts 付出了 CPU 初始化、内存分配和 H2D 搬运成本。这也是 target 加载日志出现以前长时间沉默、且内存异常高的主要原因。

## 4. 具体修改

### 4.1 重做启动日志边界

在 [`train_glm5_3_flash_dspark_fsdp2.sh`](scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh) 中：

- 真实运行的 node log 第一条记录固定为 `[deepspec-launch-start]`，包含当前时间、时区、host、PID、node rank 和唯一 launch id，例如：

  ```text
  [deepspec-launch-start] time=2026-09-03 21:15:37 +0800 host=... pid=... node_rank=0 launch_id=20260903_211537_pid...
  ```

- 使用 `EXIT` trap 输出 `[deepspec-launch-exit]`、退出码和诊断结论。worker 失败时优先指向 `error.json`，否则检查常见 traceback/OOM/NCCL timeout 记录。
- node summary 仍位于 `OUTPUT_ROOT/logs/node_rank_<n>.log`；每次 torchrun 的完整 per-rank stdout/stderr 放到带时间和 PID 的独立目录，避免不同启动相互覆盖。
- node summary 只 tee local rank 0，避免 8 个 rank 的重复输出淹没关键阶段；完整信息仍保留在 per-rank 文件中。
- 默认启用 `PYTHONUNBUFFERED=1`，减少日志因缓冲而延迟出现的问题。
- `LOGGING_STEPS` 可以控制训练指标间隔，默认值为 1。

同时增加两组阶段日志：

- `[deepspec-draft-load]`：拆出 config/tokenizer、模型构造和放置、embedding/lm_head 权重三个耗时，并打印本地 expert 数量。
- `[deepspec-target-load]`：打印 reader 类型、线程数以及 `state_dict`、读取和反量化、assign、metadata scan 的耗时。

这样日志不再只有一个笼统的“加载模型”，而能直接定位慢在哪个子阶段。

### 4.2 增加按节点复用的本地 checkpoint cache

launcher 默认把共享模型缓存到：

```text
${TMPDIR:-/tmp}/deepspec-model-cache/glm5-<fingerprint>
```

实现要点如下：

- fingerprint 包含 `config.json`、`model.safetensors.index.json` 的 SHA256，以及排序后的 62 个分片文件名和大小。
- 同一 fingerprint 使用 `flock`，避免同一节点上的并发 launcher 重复复制。
- 先复制到隐藏的 `.partial` 目录，默认用 8 个 worker 并行复制 62 个大分片。
- 发布前再次核对 fingerprint、分片数量、config 和 index，并写入 `.deepspec-cache-ready`。
- 最后通过同一文件系统内的原子 `mv` 发布；中断或未完成的目录不会被训练读取。
- 预留 5% 空间余量。自动默认 cache 无法使用时回退到共享源；用户显式指定的 cache 无法建立时则直接失败，避免悄悄偏离部署要求。

这项修改把“每个 rank 都从 AFS 随机读取”变成“每个物理节点顺序并行复制一次，后续所有 rank 从本地盘读取”。第一次会额外付出本机实测 285 秒，后续启动和 model-swap reload 都直接命中。本节点当前已经存在完成的 306 GiB cache。

### 4.3 快速解析 safetensors 元数据

在 [`deepspec/modeling/target/glm5_checkpoint.py`](deepspec/modeling/target/glm5_checkpoint.py) 中增加专用的 `Glm5QuantizedHuggingFaceStorageReader`：

- 每个分片只读取一次前 8 字节 header 长度和 JSON header；
- 直接由 header 构造 DCP `TensorStorageMetadata`；
- 37,338 个 scale tensor 不再重复建立 DCP destination metadata，因为量化 reader 会在读取对应 FP8 weight 时直接加载 scale；
- 如果遇到带 `DCP_SHARDING_INFO` 的非原生 checkpoint，则回退到 PyTorch 通用实现，避免错误套用假设。

真实加载中的 metadata scan 已降到 1.6～2.3 秒。

### 4.4 向量化 FP8 反量化

同一个 reader 覆盖 `_dequantize_tensor`：

- 对 GLM 常见的 block-aligned tensor，把权重 reshape 为 `[row_blocks, 128, col_blocks, 128]`；
- 将二维 scale grid 通过广播一次性乘到所有 block，计算仍在 FP32 中完成，最后转 BF16；
- 对 TP 非对齐 slice 或边缘 block，展开所需的小 scale grid 后精确裁切，保留兼容路径。

该改动减少的是 Python 调度次数，不改变存储布局，也不把 FP8 计算偷偷降精度。对非对齐 slice 的测试与 PyTorch 参考实现使用 `rtol=0, atol=0` 比较，结果完全相同。

### 4.5 draft 模型在构造时直接做 expert pre-sharding

在 [`deepspec/modeling/dspark/glm5_next/modeling.py`](deepspec/modeling/dspark/glm5_next/modeling.py) 中加入 `Glm5NextDSparkMoE`，并由 [`deepspec/trainer/dspark_trainer.py`](deepspec/trainer/dspark_trainer.py) 在构造模型时传入 EP size/rank：

- routed expert tensor 从一开始就只分配 `288 / EP` 个；当前 EP=8，即每 rank 36 个。
- router 仍保留全局 288 个输出，shared expert 也保持全局语义，因此 routing 逻辑没有变。
- 给本地 expert 标记 `_deepspec_expert_parameters_distributed`，让 parallel adapter 安装 all-to-all dispatch，但不再二次切片。
- 本地 experts 使用由基础 seed、EP rank 和 layer index 派生的确定性 seed 重新初始化。
- `torch.random.fork_rng` 保证 expert 初始化不污染全局 RNG；相同 EP rank 的 HSDP replica 得到相同 experts，不同 EP rank 得到不同 experts；dense/replicated 参数仍在各 rank 保持一致。

这不只是“少打印一点日志”，而是移除了实际的无效参数构造和搬运。

### 4.6 将 DCP reader 默认并发设为 8

[`train_glm5_3_flash_dspark_fsdp2.sh`](scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh) 现在默认导出：

```bash
DEEPSPEC_DCP_LOAD_THREADS=8
```

loader 自身的默认值也同步为 8。launcher 会校验它必须是正整数，并在启动摘要及 `[deepspec-target-load]` 中打印最终值。选择 8 不是猜测：同一本地 cache、同一模型和拓扑下，4 threads 为 107.5 秒，8 threads 为 69.0 秒。

## 5. 正确性与兼容性

为了避免“加载更快但权重错了”，修改保持了以下约束：

- DCP planner 仍只读取当前 TP/EP/FSDP rank 拥有的 slice，不先加载完整模型再切。
- fused Q/K/V conv1d、packed gate/up experts、down projection 和普通 TP tensor 的映射均有测试覆盖。
- FP8 aligned 和 unaligned 路径最终都按原逻辑执行 FP32 scale multiply，再转 BF16。
- meta model 经 DCP 物化后使用 `load_state_dict(..., assign=True)`，并检查 missing/unexpected keys 以及遗留 meta parameters。
- pre-sharded draft 保留全局 router，parallel adapter 不二次切 expert；不同 rank 的 expert 初始化独立，而 replicated dense 权重一致。
- 4-thread 完整 8 GPU smoke test 已跑完一步训练，包括 target features、draft forward/backward、optimizer、target unload，退出码为 0。

需要明确的兼容性差异：fresh training 时，本地 expert 使用新的确定性分片 seed，因此它与旧版“初始化完整 288 experts 后再切片”的逐 bit 初始值不相同，但初始化分布与分布式一致性约束保持不变。resume training 会由 checkpoint 覆盖这些初始 expert 权重。

## 6. 使用方式

推荐让每个调度节点直接执行原命令：

```bash
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

生产环境最好明确指向节点本地 NVMe，而不是假设 `/tmp` 的底层介质：

```bash
TARGET_MODEL_CACHE_DIR=/local-nvme/deepspec-model-cache \
  bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

常用开关：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TARGET_MODEL_PATH` | 共享 GLM-5.3-Flash 路径 | 原始 checkpoint |
| `TARGET_MODEL_CACHE_DIR` | `${TMPDIR:-/tmp}/deepspec-model-cache` | 节点模型 cache 根目录 |
| `TARGET_MODEL_CACHE_COPY_WORKERS` | `8` | 首次缓存时的分片复制并发 |
| `DEEPSPEC_DCP_LOAD_THREADS` | `8` | DCP 读取与反量化并发 |
| `LOGGING_STEPS` | `1` | 训练指标打印间隔 |
| `TORCHRUN_PER_RANK_LOGS` | `true` | 是否保留完整 per-rank 日志 |
| `OUTPUT_ROOT` | `output/glm5_3_flash_dspark_fsdp2` | node log、worker log 和输出根目录 |

如果节点没有足够本地空间，可以明确禁用模型 cache：

```bash
TARGET_MODEL_CACHE_DIR=off \
  bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

此时仍会使用快速元数据解析、向量化 FP8 反量化和 draft expert pre-sharding，但各 rank 会重新直接读取共享模型，速度会受到 AFS 竞争影响。

## 7. 验证项

修改后执行了以下验证：

```bash
bash -n scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
python -m unittest tests.test_glm5_multinode_launcher
python -m unittest \
  tests.test_glm5_next_dspark \
  tests.test_qwen38_dspark \
  tests.test_glm5_dcp_load
ruff check \
  deepspec/modeling/target/glm5_checkpoint.py \
  deepspec/modeling/dspark/glm5_next/modeling.py \
  deepspec/trainer/base_trainer.py \
  deepspec/trainer/dspark_trainer.py \
  tests/test_glm5_dcp_load.py \
  tests/test_glm5_next_dspark.py \
  tests/test_glm5_multinode_launcher.py
git diff --check
```

最终结果：核心模型/DCP 测试共 41 项通过、3 项跳过；launcher 测试共 19 项全部通过；shell 语法、Ruff 和 `git diff --check` 也全部通过。launcher 测试覆盖时间首行、失败诊断、cache 首次构建/cache hit、8-thread 默认值及环境变量覆盖。

## 8. 代价与后续注意事项

- 节点 cache 需要约 306 GiB 空间，而且当前实现不自动淘汰旧 fingerprint；模型版本频繁变化时需要纳入节点磁盘清理策略。
- 首次 cache miss 会付出一次复制成本，本机为 285 秒。它优化的是重复启动、model swap 和多 rank 共享读；只运行一次且本地盘很慢时，收益可能较小。
- fingerprint 对 config/index 做内容哈希，对大分片校验文件名和大小，并非逐字节哈希全部 306 GiB。它能识别版本变化和不完整复制，但不能发现“分片大小不变的静默位损坏”。如运行环境要求强校验，应由模型发布流程提供并校验 shard checksum manifest。
- 8 个 copy workers 和 8 个 DCP threads 是当前 AFS/本地盘上的实测最佳点，不保证适合所有机器；因此都保留环境变量覆盖能力。
- 最早发生在日志系统初始化之前的错误（例如 `PYTHON_BIN` 完全不可用）仍只能出现在调度器 stdout/stderr 中；一旦 node log 建立，其第一行就是带时区的启动时间记录。
- 8-thread 诊断在 target 加载完成后主动停止；完整一步训练成功记录来自相同实现的 4-thread 运行。后者已经覆盖了权重正确性和完整训练生命周期，前者用于选择 I/O 并发默认值。

## 9. 涉及文件

- [`scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh`](scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh)：日志、失败诊断、节点本地模型 cache、DCP 线程默认值。
- [`deepspec/modeling/target/glm5_checkpoint.py`](deepspec/modeling/target/glm5_checkpoint.py)：快速 metadata reader、向量化 FP8 反量化和分阶段耗时。
- [`deepspec/modeling/dspark/glm5_next/modeling.py`](deepspec/modeling/dspark/glm5_next/modeling.py)：draft expert 构造期 pre-sharding 与确定性本地初始化。
- [`deepspec/trainer/dspark_trainer.py`](deepspec/trainer/dspark_trainer.py)：把 EP size/rank 传入 draft 构造器。
- [`deepspec/trainer/base_trainer.py`](deepspec/trainer/base_trainer.py)：draft 初始化分阶段日志。
- [`tests/test_glm5_dcp_load.py`](tests/test_glm5_dcp_load.py)：DCP slice、FP8 精确等价和 meta materialization 测试。
- [`tests/test_glm5_next_dspark.py`](tests/test_glm5_next_dspark.py)：本地 expert 构造、router 与分布式一致性测试。
- [`tests/test_glm5_multinode_launcher.py`](tests/test_glm5_multinode_launcher.py)：日志、cache 和 reader thread 启动契约测试。
- [`scripts/data/README.md`](scripts/data/README.md)：运行参数与 cache 使用说明。
