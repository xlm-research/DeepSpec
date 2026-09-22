# B300 单机验证与多机调试交接

本机：`dev-951e3d4b-0`，Ray IP `172.20.2.179`，8 × B300。
项目：`/mnt/afs-agentpro/lezewei/DeepSpec`。

## 环境与单机启动

```bash
cd /mnt/afs-agentpro/lezewei/DeepSpec
source ./env.sh
```

使用现有 conda 环境 `deepspec_vllm_torchtitan_envs`，默认位于
`/mnt/afs-agentpro/share/env/miniconda3/envs/`。`PIPELINE_PYTHON`、
`DEBUG_PYTHON`、`VLLM_PYTHON_BIN` 均指向该环境，CUDA 编译器与 CUDA 12
runtime 也取自该环境；项目中的 vLLM / TorchTitan 源码优先。
`TORCH_CUDA_ARCH_LIST` 未设置时从本机 GPU 检测，B300 为 `10.3`。
不同安装位置设置 `DEEPSPEC_CONDA_SH` 和 `DEEPSPEC_CONDA_ENV`；模型通过
`TARGET_MODEL_PATH` 覆盖，默认 `/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B`。

本次在 tmux 会话 `deepspec-b300-ray` 中启动了独立 Ray Head
`172.20.2.179:26379`，注册 8 GPU / 24 CPU。
先查询；节点重启或服务退出后才执行启动命令：

```bash
python -m ray.scripts.scripts status --address 172.20.2.179:26379

# 仅在 Head 未运行时执行，放在独立终端或 tmux 中常驻。
RAY_NODE_IP=172.20.2.179 RAY_HEAD_PORT=26379 PIPELINE_RAY_BLOCK=true \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_ray.sh head
```

使用外部 Ray，以便训练结束后独立 CPU verifier 仍可在冻结计划的原节点运行。
正常训练自动创建、清理每轮 Mooncake Master 和唯一特征池。
本次执行环境会回收短命令的后台子进程，因此 Ray 服务使用 `--block` 常驻；
通过 `tmux attach -t deepspec-b300-ray` 进入，Ctrl-b、d 离开并保留服务。

```bash
export RAY_ADDRESS=172.20.2.179:26379

# 默认仅执行 4K 三步验证，使用 64 GiB CPU 池；自动生成新输出目录。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/b300_debug.sh

# 按需选择额外阶段；每次仍需新的输出目录。本次最终验收范围仅为 4K。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/b300_debug.sh --stage pool
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/b300_debug.sh --stage 128k
```

配置：真实 Qwen3.8-27B，四卡 vLLM TP4 + 四卡原生 TorchTitan TP4，
GAS4/global batch4，生产 batch4、writer inflight2、window8，TCP/CPU。
分配期限为 600 秒，覆盖环境重验、探针和服务分配；训练期限为 3600 秒。
共享文件系统上的冷启动曾耗尽原来的 120 秒分配期限，失败清理已通过。
每个训练阶段应有 12 条实际指定长度样本、48 次完整读取校验、四 rank 各
3 次优化器更新、完整 checkpoint、独立 CPU 核验，以及源对象与本轮资源释放。
只运行一轮 GPU 任务；`debug-result.json.status=passed` 才表示整个命令验收通过。

## 9 月 21 日双机 128K：八卡生产、八卡消费、512 GiB 池

当前明确分工：`172.20.13.215`（`dev-73915620-0`）八卡 vLLM 生产；
`172.20.2.179`（`dev-951e3d4b-0`）八卡 draft 训练，并持有唯一的 512 GiB
Mooncake CPU 池。两端都是 DP2 × TP4；Ray Head 继续在消费机上，控制节点
身份与模型角色独立。两台使用相同源码、conda Python、模型与共享数据/输出路径。

Ray 由用户手动启动。第二台尚未加入集群时，在它的独立终端执行：

```bash
source ./env.sh
export RAY_NODE_IP=172.20.13.215
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/ray_mutilnode_train.sh \
  worker 172.20.2.179:26379
```

该 worker 命令默认前台常驻，放在独立终端。随后在当前 Head 另开终端：

```bash
source ./env.sh
export RAY_HEAD_ADDRESS=172.20.2.179:26379
export RAY_CONNECT_TIMEOUT=60
export PRODUCER_NODE=172.20.13.215
export CONSUMER_NODE=172.20.2.179

bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/ray_mutilnode_train.sh wait
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/ray_mutilnode_train.sh status

# 128K 三步；启动器先做跨机 TCP/CPU 传输检查，再加载模型。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/ray_mutilnode_train.sh train \
  --producer-dp 2 --consumer-dp 2 --context-length 131072 --pool-gib 512 \
  --gpu-sharing shared --steps 3 --window 8 --producer-batch-size 4 --writer-inflight 2 \
  --allocation-timeout-seconds 600 --timeout-seconds 3600
```

用户明确要求可与其他任务并行，因此此命令显式设置 `--gpu-sharing shared`。
该选项允许 GPU 上已有外部进程，并将共享策略写入冻结计划；仍保留 Ray 资源
预留、实际显存占用记录、节点内存预算、GPU/rank 归属检查和本轮资源清理。
一般入口默认仍为 `exclusive`。512 GiB 是池容量，窗口 8 和 3 步短测不构成满池压力测试。

本轮可复用入口为
[`run_128k_full16_pool512.sh`](../../../debug_logs/b300_two_node_20260921_114721/run_128k_full16_pool512.sh)，
它每次生成新的输出目录。继续使用 TCP/CPU。
跨节点 TP8（M2）和跨节点训练 DP（M3）需三台真实节点，按
[v3 quickstart](../../../specs/001-unify-ray-topology/quickstart.md) 生成 M2/M3
配置；环境步骤替换为本页的 `source ./env.sh`。
生成配置时设置 `value['timeouts_seconds']['allocation'] = 600`，随后重新 preview。

### 本轮双机验证结果

运行目录：
[`b300_full16_128k_pool512_shared_20260921_121704`](../../../outputs/b300_full16_128k_pool512_shared_20260921_121704)。
主流程以退出码 0 完成；[status.json](../../../outputs/b300_full16_128k_pool512_shared_20260921_121704/status.json)
为 `succeeded`，`cleanup_complete=true`。

- 已核验生产端 8 个 inference worker、消费端 8 个 training rank 的节点与 GPU 归属。
- 12 条实际 131072-token 样本、48 次完整特征读取校验、八个 rank 各 3 次优化器更新；
  GAS2、global batch4，三步 loss 为 3.91886、3.09035、3.89431。
- 完整 [step-3 checkpoint](../../../outputs/b300_full16_128k_pool512_shared_20260921_121704/checkpoints/step-3)
  已保存；独立 CPU verifier 读取全部 756 个字段，确认 40 个参数发生变化，未初始化 CUDA。
- [verification.json](../../../outputs/b300_full16_128k_pool512_shared_20260921_121704/verification.json)
  为 `verified=true`，资源指标无缺失；12 个源对象和 22 项本轮资源均已释放，外部 Ray 保留。
- GPU 共享策略、调度容量、默认独占行为及外部进程保留等相关回归：119 passed。

本轮使用 `shared` 策略，但先前的外部 GPU 任务在模型启动前已经退出，因此本次结果
验证了双机 128K 训练链路，未验证与该任务同时占用显存的稳定性。512 GiB 为配置并分配
的池容量；本轮三步短测未覆盖满池负载或长时间运行。

## 观察、取消与核验

将 `RUN_OUTPUT` 设为启动日志中实际训练目录；大池调试器会在其输出目录下创建
`4k/` 和 `128k/`。冻结计划生成后，不修改绑定的源码、环境与输入文件。

```bash
python -m deepspec.pipeline.cli status --run-dir "$RUN_OUTPUT" --json
python -m deepspec.pipeline.cli cancel --run-dir "$RUN_OUTPUT"
python -m deepspec.pipeline.cli verify --run-dir "$RUN_OUTPUT"
```

`status` 的 `produced`、`complete_reads`、`rank_updates` 与 `committed_ranks`
分别表示生产、完整读取、优化器更新和保存提交。初始化期间均为零是正常的；
启动耗时包含共享文件系统上的 Python 导入、CPU 预处理和 27B 权重加载。
本机一次完整 checkpoint 的独立 CPU 逐字段读取与哈希核验约 7–10 分钟；
`draining/verifying` 期间 GPU 已释放仍属正常，调试器随后还会做一次结束后独立核验。
查看训练日志使用 `tail -F "$RUN_OUTPUT/consumer.log"`；结构化失败原因在
`status.json.reason`，driver 日志为 `driver-request.log`。

取消仅针对该运行。训练结束后外部 Ray 保留，GPU 模型进程和每轮 Mooncake
服务应释放。当前 Head 日志在 `/tmp/deepspec-b300-ray-20260920/session_latest/logs/`。

## 本次修复与证据

- `env.sh` 改用本机 conda；修正 CUDA 架构、模型路径和解释器传播。
- 单机与多机启动脚本统一接受 `TARGET_MODEL_PATH`。
- 修复 v3 planner 的长度字段契约：`seq_len` 和 `context_chunk_len` 声明为
  原生转换函数实际使用的 `[1]`，保留严格 shape/dtype 校验及原生模型数学路径。
  回归把 planner、原生转换、序列化和发布校验串联，覆盖单 token 与多 token。
- 修复独立 DCP 核验的模型键名：原生 CheckpointManager 将模型参数保存到顶层，
  如 `fc.weight`。核验将初始化记录中的 `model.fc.weight` 一一映射到该键，
  继续严格检查完整字段集合、键名碰撞、形状、范围和参数更新；原生保存格式不变。
  CPU 回归使用原生 `_flattened_model_states_sd` 生成真实 DCP 文件。
- 修复 cgroup 页缓存导致的内存预算误拒绝：只将干净的非活跃文件缓存计入
  可回收额度，保留原始用量、缓存信用和各层额度观测；不计入 active cache、
  匿名内存、tmpfs 或交换空间。继续受物理可用内存、父级 cgroup、80% 静态
  上限及 64 GiB 预留限制。字段定义见
  [Linux cgroup v2 文档](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)。
- 独立资源采样对 `nvidia-smi` 的 10 秒超时最多追加两次查询，保留每次超时
  原因于事件的 `data.sample_retries`；连续三次超时仍记录 missing 并使完整
  核验失败，其他错误不重试。首次 128K 已完成训练、checkpoint 核验及实际
  资源释放，但退出期间一次 GPU 采样超时导致最终指标核验失败；原始失败
  保留在 `debug_logs/b300_20260920/128k-sampling-diagnostic.json`。
- 环境与八卡 BF16 运算：`debug_logs/b300_20260920/environment.json`。
- 本机 GPU 拓扑与容器网卡：`debug_logs/b300_20260920/gpu-topology.txt`、
  `debug_logs/b300_20260920/network-interfaces.txt`；Ray 节点 IP 对应本容器 `eth0`。
- 依赖：`debug_logs/b300_20260920/pip-check.txt`，无依赖冲突。
- 定向回归：`debug_logs/b300_20260920/regression.log`，161 passed。
- 内存/预算回归：`debug_logs/b300_20260920/memory-regression.log`，67 passed。
- 分配期限透传及相关回归：`debug_logs/b300_20260920/startup-regression.log`，27 passed。
- 原生特征契约及相关回归：`debug_logs/b300_20260920/feature-shape-green.log`，129 passed。
- 原生 checkpoint、执行与训练握手回归：`debug_logs/b300_20260920/checkpoint-keys-green.log`，103 passed。
- GPU 采样重试、持续故障拒绝及相关回归：`debug_logs/b300_20260920/gpu-sampling-green.log`，
  65 passed、1 skipped；跳过项需要分布式 GPU 测试环境，不计入通过。
- 各组回归的测试选择有重叠，不将通过数相加；最终 Ruff 与格式检查见
  `debug_logs/b300_20260920/ruff-final.log`，11 个修改的 Python 文件全部通过。
- 多机启动器的本机成员查询：`debug_logs/b300_20260920/ray-launcher-status.log`，
  单节点目标下 8 GPU ready，无待调度资源请求。

## 9 月 20 日单机运行结果

| 阶段 | 实际验证 | 结果与证据 |
|---|---|---|
| CPU 池探针 | 64 GiB 容量；独立 writer/reader 写读一条 4K 合成特征，SHA256 校验及 34 个分块删除 | [通过](../../../outputs/b300_20260920_final/pool-probe/result.json)；不是满池压力测试 |
| 4K | 12 条实际 4096 长度样本，48 次校验读取，四 rank 各 3 步更新 | [4k-result.json](../../../outputs/b300_20260920_final/4k-result.json)：`passed`；主流程及结束后独立复核均通过 |
| 128K | 原始运行已完成训练及 checkpoint 核验，最终指标核验因一次 GPU 采样超时失败；按用户最新要求不再重跑，不计入最终通过范围 | [历史失败记录](../../../debug_logs/b300_20260920/128k-sampling-diagnostic.json) |

4K 的 [verification.json](../../../outputs/b300_20260920_final/4k/verification.json)
确认独立 CPU 读取全部 732 个 checkpoint 字段、40 个参数变化、优化器步数 3，
12 条源对象与 15 项本轮资源均释放；结束后的独立 verifier 也已清理，错误列表为空。
本轮完整运行的指标无缺失，loss 均为有限值。

用户在 4K 结束前将范围缩小为仅验证 4K，因此原批量调试入口在 4K 主流程
成功清理后被定向中止，避免自动开始 128K。原 `debug-result.json` 保留该中止
记录，不能作为 4K 训练失败的判断；4K 的执行与独立核验证据位于上表目录。
调试器清理记录中强制终止和最终残留进程列表均为空。

汇总证据：[setup-result.json](../../../debug_logs/b300_20260920/setup-result.json)。
9 月 20 日的最终验收范围为单节点 4K；Ray Head 保留在 `172.20.2.179:26379`，供后续接入节点。

单机通过范围不等同于多节点或长时间稳定性验收。
