# Qwen3.8：Ray + Mooncake + vLLM + TorchTitan

> 2026-09-20：`train.sh`、`train_multinode.sh`、`ray_mutilnode_train.sh train` 仍转入 `deepspec.pipeline.run`，该 Python 入口现统一执行 v3 preview/controller。先 `source ./h800conda.sh` 使用现有 `PIPELINE_PYTHON`；当前验证与六命令用法见 [流水线说明](../../../deepspec/pipeline/README.md) 和 [验收报告](../../../specs/001-unify-ray-topology/acceptance-report.md)。下文旧地址、训练结果和峰值压力模式属于历史记录；`retain_for_peak` 模式目前迁移时明确拒绝，勿将旧成功视为新入口验收。外部 Ray 由脚本管理，controller 仅回收本次登记资源。

实际操作请先读 [训练操作指南](TRAINING_GUIDE.md)：按顺序说明何时启动 Ray、何时自动启动 Mooncake，
以及 A/B 两台机器分别执行的命令、全十六卡参数、日志和退出方式。

当前多机推荐使用统一入口 [`ray_mutilnode_train.sh`](ray_mutilnode_train.sh)，按
[训练指南第 0 节](TRAINING_GUIDE.md#0-当前统一入口-ray_mutilnode_trainsh) 依次执行 `head`、`worker`、
`wait/status` 和 `train`。默认多机共享池为 B 机上的 64 GiB Mooncake Store；正常训练会自动管理
Master 和 FeatureBuffer，不需要手动启动 `start_mooncake.sh`。

换窗口继续讨论或开发时，先读 [会话交接](HANDOFF.md)，其中记录已确认要求、版本、改动、运行证据和待办。

当前单机八卡调试可用独立的 [大池调试脚本](DEBUG_SINGLE_NODE.md)：
`debug_single_node.sh` 默认检查环境、验证 1 TiB CPU 池，再依次执行 4K/128K 五步短测。
生产/消费各 TP4，生产 batch20/window40；`--stage peak` 运行 128K 接近满池压力测试，
显式使用两 epoch 并保留已消费对象至目标峰值。池容量不等于整条流水线的内存上限。

下一阶段的容量对照见 [B 机特征池扩容方案](B_POOL_EXPANSION_PLAN.md)：当前仅完成方案，
等第二台机器到位后评估 64/128 GiB 池和窗口；扩池配置尚未验收，脚本默认值保持原配置。

当前已验收配置为整机分离的全十六卡：A 机八卡 vLLM 生产，B 机八卡 TorchTitan 消费，
唯一 Mooncake CPU 特征池在 B 机。入口为 `ray_mutilnode_train.sh train --producer-dp 2 --consumer-dp 2`，
两端各使用 DP2 × TP4。Ray 分配 GPU 和协调数据状态，
vLLM 管理推理 worker，TorchTitan 管理训练 rank。特征经 Mooncake CPU buffer
送入原生 DSpark 的 `fc → hidden_norm → context K/V` 训练路径。

## Ray 与 Mooncake 启动

启动脚本均在本目录，固定使用 `/tmp/deepspec_vllm_torchtitan_envs/bin/python`，不使用 uv。
支持从其他目录用绝对路径调用；`--help` 查看参数，`DRY_RUN=true` 只打印命令。

日常两机训练优先使用上面的 `ray_mutilnode_train.sh` 统一入口。下面的 `start_ray.sh` 命令是底层等价用法，
适合只调试 Ray 服务时使用；正常训练无需单独执行 `start_mooncake.sh`。

| 脚本 | 用途 |
| --- | --- |
| [start_ray.sh](start_ray.sh) | `head` 启动 A 机 Head；`worker HEAD_IP:PORT` 将 B 机加入集群 |
| [start_mooncake.sh](start_mooncake.sh) | 独立启动 Mooncake Master，供单独调试控制服务 |
| [cluster.sh](cluster.sh) | 兼容旧命令，转发到 `start_ray.sh` |

A 机执行（生产节点 `172.20.1.195`）：

```bash
RAY_NODE_IP=172.20.1.195 PIPELINE_RAY_BLOCK=true \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_ray.sh head
```

B 机执行（消费及特征池节点 `172.20.5.39`）：

```bash
RAY_NODE_IP=172.20.5.39 PIPELINE_RAY_BLOCK=true \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_ray.sh worker 172.20.1.195:26379
```

这两个命令分别在各自终端或作业中常驻。默认每节点注册 8 GPU、24 CPU，Head 端口为 26379；
可通过 `RAY_NUM_GPUS`、`RAY_NUM_CPUS`、`RAY_HEAD_PORT` 修改。省略 `PIPELINE_RAY_BLOCK=true`
时，Ray 启动成功后命令返回。Ray 的 Head/worker 身份只负责集群连接；训练中的整机分离
由 `PRODUCER_NODE` 和 `CONSUMER_NODE` 固定。下一步在 A 机另一个终端按
[全十六卡命令](#全十六卡生产-dp2--tp4消费-dp2--tp4) 启动训练。

**正常训练会自动启动并清理本轮 Mooncake Master，同时在 B 机创建唯一 FeatureBuffer 内存池。**
Master 是元数据/控制服务，随训练 driver 在 A 机启动；实际特征数据存放在 B 机。
训练使用动态端口，地址记录在输出目录的 `pipeline.json`，日志为 `mooncake-master.log`。
Ray 的 128 MiB object store 与 Mooncake 的 4/64 GiB 特征池分别配置。

需要单独启动 Master 调试时，在 A 机运行：

```bash
MOONCAKE_RPC_ADDRESS=172.20.1.195 MOONCAKE_RPC_PORT=50051 MOONCAKE_METRICS_PORT=9003 \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_mooncake.sh
```

该脚本前台运行，日志输出到终端，Ctrl-C 退出。默认 lease 为 300 秒，关闭磁盘 offload/eviction，
可用 `MOONCAKE_KV_LEASE_TTL` 修改 lease，并在脚本后追加原生 Master 参数。
这个独立 Master 不分配特征池，也不会被现有训练入口自动复用；正常训练无需提前启动它。
当前训练仍使用 TCP/CPU，继续跳过 RDMA。

```bash
DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_ray.sh head
DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_mooncake.sh
```

## 单机历史入口

在项目根目录执行：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train.sh
```

也可以从其他目录使用脚本的绝对路径；脚本会自动进入项目根目录。
固定使用 `/tmp/deepspec_vllm_torchtitan_envs/bin/python`，无需激活环境，不使用 uv。

Mooncake 依赖 CUDA 12 runtime，即使 PyTorch 使用 CUDA 13。每个节点解包环境后，
如遇到 `ImportError: libcudart.so.12`，在该节点补装：

```bash
/tmp/deepspec_vllm_torchtitan_envs/bin/python -m pip install --no-deps nvidia-cuda-runtime-cu12==12.8.90
```

`start_ray.sh`、`start_mooncake.sh` 和 `train.sh` 会将环境中的 `nvidia/cuda_runtime/lib` 加入
`LD_LIBRARY_PATH`，供当前进程及其启动的 Ray 子进程使用。

该历史单机入口需要整机 8 张 GPU 空闲且全部可见，使用生产者 0–3、消费者 4–7 的同机布局。
当前整机分离要求使用上文 Ray 启动方式及下文全十六卡入口。

## 单机历史默认配置

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

默认配置用于复现已经跑通的短程训练。单机 128K 已完成 12 个微批、3 次更新和
48 次 rank 读取校验，结果见
[128K 单机记录](../../../outputs/qwen3.8_ray_mooncake_vllm_torchtitan_20260916_141217_61024/result.json)。
两机入口与验证范围见下文；扩大上下文时需要相应调整池容量和超时。

## 两机 4+4

两台机器使用相同的共享项目目录、模型路径及 `/tmp/deepspec_vllm_torchtitan_envs`
环境。第一台运行一个 vLLM TP4，第二台运行一个 TorchTitan TP4 / DP1 / GAS4。
第二台提供唯一的 Mooncake 特征池；生产端跨机写入一份，训练 rank 在第二台各自读取。
每个节点分别核算内存预算，特征对象在四个训练 rank 读取确认后删除。

先按上文 [Ray 与 Mooncake 启动](#ray-与-mooncake-启动) 将第二台加入第一台的 Head
（本次为 `172.20.1.195:26379`）。正常训练自动管理 Mooncake。

按 Ray 实际注册的节点 IP 设置生产和消费节点。SSH 地址可能与 Ray 节点 IP 不同。
本次第二台 SSH 地址为 `10.119.16.15`，Ray 内部地址为 `172.20.5.39`。

```bash
export RAY_HEAD_ADDRESS=172.20.1.195:26379
export PRODUCER_NODE=172.20.1.195
export CONSUMER_NODE=172.20.5.39

# 仅验证跨机 Store 写入、分块读取校验及删除，不加载 GPU 模型。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --context-length 4096 --pool-gib 4 --transport-only

# 真实模型短上下文验证。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --context-length 4096 --pool-gib 4

# 128K：默认 64 GiB 消费端池、窗口 8、3 次更新、TCP/CPU 接收。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh

# 延长至 20 次更新，需 80 条有效样本。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --steps 20 --timeout-seconds 7200
```

启动器核对两端 Python/依赖版本、源码 SHA256、共享配置和文件访问、GPU 空闲状态及
每节点预算，再固定资源组和 Store 所在节点。上述默认配置共使用八张 GPU；
整机分离的 DP2 配置见下节。验证顺序是组件回归、跨机 Store、4K 整链路、128K 整链路。
RDMA 和 GPU 直传需要各自的运行证据；目标 CUDA 节点的节点级探针结果见
[TARGET_CUDA_TRANSPORT_VALIDATION.md](TARGET_CUDA_TRANSPORT_VALIDATION.md)。

2026-09-16 的 [两机 128K 结果](../../../outputs/dspark_two_node_20260916_128k_run1/result.json)
已正常退出：12 个 131072-token 微批、48 次读取校验、3 次更新，特征全部释放；
独立读取 checkpoint 确认 optimizer step=3。四卡生产在 `172.20.1.195`，四卡消费及唯一池
在 `172.20.5.39`。这是 TCP/CPU 短程验证；完整证据与计时见
[独立核验](../../../outputs/dspark_two_node_20260916_128k_run1/verification-status.json)。

同日生产端转换优化后的 [128K 复测](../../../outputs/dspark_two_node_20260916_128k_opt1/result.json)
也已正常退出，校验、梯度、更新和 checkpoint 全部通过。相同配置下转换中位耗时
42.34→6.54 秒，三步日志 loss 和记录的梯度范数一致；有限值和完整 SHA256 检查仍保留。
启动命令不变，详细比较见 [验收记录](../../../deepspec/pipeline/VALIDATION.md#生产转换优化复测2026-09-16)。

随后 [TCP20 稳定性测试](../../../outputs/dspark_two_node_20260916_128k_tcp20/verification-status.json)
已正常退出：80 个不同的 128K 样本、320 次 rank 读取校验、各 rank 20 次更新，源对象全部释放；
checkpoint 独立读取 optimizer step=20。第 2–20 次更新间隔中位 93.56 秒，转换中位 6.55 秒/批。
池预留峰值仍为 45.012 GiB，没有内存压力；稳态消费进程 RSS 约 165 GiB，前后窗口增加 68 MiB。
这轮覆盖首次推理至消费者退出约 32.63 分钟，不能代替小时/天级稳定性测试。

RDMA 当前受目标容器网络配置阻塞：`mlx5_10` 的 QP 建连报 `No such device`，
第一台缺少第二台已有的 RoCE 网口；`mlx5_z0–z3` 则属于 NVLink 管理设备。
用户已明确暂时跳过 RDMA，后续两机流水线使用 TCP 继续。
目标节点的本机 GPU 目标缓冲区探针已经通过，但尚未完成跨机 RDMA、远端 GPU 直传
和 TCP/RDMA 性能对比。修复检查点与重测命令保留在 [RDMA 诊断](RDMA_DIAGNOSIS.md)。

每轮新建输出目录。`environment.json` 记录两端版本、GPU UUID 和内存上界；
`node-producer.jsonl` / `node-consumer.jsonl` 记录各节点实时内存余量、本轮 GPU 进程、
本轮进程 RSS/匿名页及指定设备可读的 RDMA 计数器。RSS 求和可能重复计算共享页，仅用于趋势；
实际准入使用节点/cgroup headroom。缺失的网卡计数器保留为空，不作为带宽证据；
`transport-probe.json` 记录预检查。`events.jsonl` 将 Store 传输、SHA256、生产转换和
消费物化计时分开。跨机事件统一在 FeatureBuffer 节点打时间戳；这些是主机计时，
不作为 GPU trace。`--transport-only` 的 `result.json` 明确记录 `models_started=false`。

退出时清理本轮 actor、资源组和 Mooncake Master；连接的 Ray 集群由其启动者管理。

## 两机消费者 DP2：4+8 卡（整机分离）

当前入口将生产与消费固定到不同机器。`--consumer-dp 2` 使用第一台四卡生产、
第二台八卡消费与唯一 Mooncake CPU 池。消费是一个原生 TorchTitan
TP4 × DP(shard)2 / GAS2 实例，在第二台由单个 `torchrun --standalone --nproc-per-node=8`
启动。两个 DP 组均在第二台，TP 与 DP 通信都留在消费节点。

消费 global rank 0–3 / DP0 读取偶数位置，4–7 / DP1 读取奇数位置；样本位置从 0 开始。
每个样本仅由所属四个 TP rank 读取，四个 ACK 齐全后释放。每次更新四个样本，
3 次更新使用 12 个不同样本、48 次读取校验；原生 DP 微步游标最终为 6，全局样本数为 12。
两机分别预算生产与消费内存，GPU 归属检查同时拒绝生产/消费共享物理节点。

此前 [十二卡 DP2 TCP20](../../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/verification-status.json)
采用每台均有消费 rank 的混布拓扑，80 样本、320 次读取和 20 次更新曾通过。
用户已纠正原要求是整机分离，因此这只是历史运行证据，不作为当前拓扑验收，也不重复运行。
新的十六卡验证使用下节命令。

## 全十六卡：生产 DP2 × TP4，消费 DP2 × TP4

两台各需八张空闲 GPU。第一台八卡全部用于生产，第二台八卡全部用于消费，并提供唯一 Mooncake CPU 池。
生产端使用一个原生 `AsyncLLM` frontend 与 Ray DP backend；vLLM 创建两个 EngineCore
及各自四个 TP worker，两个 EngineCore 均在第一台（`data_parallel_size_local=2`），
原生 placement 限制为第一台 IP。当前 dense target 不支持 offline DP，不能改用普通 `LLM` 加
`VLLM_DP_*` 环境变量。消费端继续使用上述原生 FSDP2 / TP4 / GAS2 / global batch 4。

```bash
export RAY_HEAD_ADDRESS=172.20.1.195:26379
export PRODUCER_NODE=172.20.1.195
export CONSUMER_NODE=172.20.5.39

bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --producer-dp 2 --consumer-dp 2 --context-length 4096 --pool-gib 4 \
  --steps 3 --output outputs/dspark_full16_4k_example

bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --producer-dp 2 --consumer-dp 2 --context-length 131072 --pool-gib 64 \
  --steps 3 --timeout-seconds 3600 --output outputs/dspark_full16_128k_example
```

frontend 统一按全局位置顺序预留，显式按 `position % 2` 路由；每组同时最多一个生成调用。
各组的 TP0 独立写入，允许 READY 乱序，Store 写入与在途特征仍受全局 window/容量约束。
写入开始前检查原生 `data_parallel_index`、节点与唯一归属，发布时再次校验。
请求或后台 writer 失败会取消其他请求并中止 buffer；全部生成及 Store 写入结束后才完成生产。
第一台预算两组生产暂存，第二台预算八 rank 读取/预取副本与池；推理区间先合并后
计算与消费计算的交集，避免重复计时。

整机分离小模型原生 DP2/TP4 的显式路由、第一台八 worker 归属与清理已通过，
整机分离组件检查 24 项通过，另有同节点混布拒绝检查。
真实整机分离 [4K/3 步](../../../outputs/dspark_two_node_20260916_4k_full16_separated1/verification-status.json)
和 [128K/3 步](../../../outputs/dspark_two_node_20260916_128k_full16_separated1/verification-status.json)
均已通过：每轮 12 个不同样本、48 次 SHA256、八 rank 各三次更新，全部释放；独立读取
fc optimizer step=3、native cursor6 和八个 checkpoint 分片通过。两机本轮进程清空，Ray 保留。
全十六卡 20 步延长尚未运行；完整记录见 [交接文件](HANDOFF.md)。
继续使用 TCP/CPU；目标节点的本机 GPU 目标缓冲区探针已通过，但跨机 RDMA、远端
GPU 直传和完整训练配置仍未验收，详见 [目标 CUDA 节点传输验证](TARGET_CUDA_TRANSPORT_VALIDATION.md)。

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
单机私有 Ray 的日志目录记录在 `pipeline.json` 的 `ray_logs` 字段中。
两机模式连接已有集群，Ray worker 日志位于各节点启动 Ray 时指定的临时目录下；
Head 默认使用 `/tmp/dsray-<时间>-<PID>/session_latest/logs`，
消费端原生训练日志另写入共享输出目录的 `consumer.log`。
当前 DP2 的八个消费 rank 都在第二台，日志统一为 `consumer.log`。
`consumer_rank_initialized` 事件记录八个 rank 的 DP/TP 坐标与主机。
历史混布运行中的 `consumer-node1.log` 属于第一台，不能据此解释当前整机分离布局。

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
`VLLM_USE_V2_MODEL_RUNNER=1`。用户已完成 `step-3` 导出；加载测试发现并修复了 vLLM
将 Qwen DSpark 误判为 MTP 的兼容问题。GPU 4–7、TP4、BF16 下模型加载及 warmup
已通过；参数抽查和短生成尚未完成。修复与证据见 [加载验证](VLLM_LOAD.md)。

具体实现和验证范围见 [流水线说明](../../../deepspec/pipeline/README.md)
与 [4+4 验收记录](../../../deepspec/pipeline/VALIDATION.md)。
