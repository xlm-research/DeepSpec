# 单机八卡与大容量 Mooncake 池调试

入口为 `debug_single_node.sh`，Python 实现在 `debug_single_node.py`。
使用一个 vLLM TP4 生产者、一个 TorchTitan TP4/DP1/GAS4 消费者，
Mooncake CPU 池与二者位于同一主机，传输使用 TCP。

默认池容量 **1024 GiB = 1 TiB ≈ 1.10 TB**，生产 batch 为 20，window 为 40，每次短测训练 5 步。
这是 Store 池容量，不是整个任务的 CPU 内存或 RSS 上限。
如果要求池严格不超过十进制 1 TB，可传 `--pool-gib 931`。
普通短测对象预留额度为池的 75%，同时受 window 限制。满池压力使用下面的 `peak` 阶段。

## 运行

在项目根目录执行：

```bash
# 只预览，不建目录、不启动服务或模型。
DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh

# 完整短测：检查 → 大池探针 → 4K/5步 → 128K/5步。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh

# 也可分别运行；每次调用创建新的输出目录。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh --stage check
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh --stage pool
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh --stage 4k
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh --stage 128k

# 真实 128K 特征接近满池：20 请求/批，35 次更新，140 条 epoch 样本。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.sh --stage peak
```

可选 `--output` 指定**尚不存在**的目录。失败立即停止，不继续下一阶段，重试使用新目录。
`--steps` 同时作用于所选的训练阶段；`--window` 不改变全局 batch=4 的训练语义。
`--timeout-seconds` 默认 3600；外层额外给准备、启动与清理保留 1200 秒。
`--probe-timeout` 默认 600 秒，分别限制大池探针的 owner、writer、reader 阶段。

`--producer-batch-size` 默认 20，同时控制传入 `LLM.generate()` 的请求数及 vLLM 的
`max_num_seqs`；GPU token 调度仍使用 chunked prefill，受 KV/特征页容量约束。
背压或最后一批可能小于设定值，`inference_batch` 事件记录每次实际提交数量。
`--writer-inflight` 默认 2，在提取/分配 pinned CPU 缓冲前限流，避免线程池队列持有大量特征。
消费者仍为原生 global batch 4、TP4/GAS4。

## 接近满池的容量压力模式

`--stage peak` 默认 `--steps 35 --epochs 2 --window 140 --timeout-seconds 10800`，
对象容量额度提高到 99%。每条真实 128K 特征为 8,055,160,848 字节；默认池可容纳
135 条，实际有效载荷目标 **1012.764 GiB（98.903%）**，保留余量供分配器使用。
若数据或窗口不足以达到池容量的 95%，准备阶段直接拒绝运行。

这个模式持续消费，但在首次达到目标前暂不删除已收到全部 ACK 的对象；达到目标时释放
已确认对象，此后恢复正常 ACK 后立即删除。没有通过累加历史写入字节来计算池峰值。
默认输入只有 80 条不同源样本，因此明确允许第二个 epoch；这是容量压力测试，
不能将 140 条 epoch 样本写成 140 条不同源样本或用作无重复数据的性能比较。

`pool_peak_reached` 记录 READY 且尚未删除的实际有效载荷、池 owner RSS、节点/cgroup
内存余量。`verification.json` 从 READY/删除事件独立重算 `peak_resident_bytes`，
检查目标已达到、每条特征完成四 rank SHA256 读取、全部对象最终删除和 checkpoint。
`peak_reserved_bytes` 包含尚未生成的预留，不作为驻留峰值证据。近满池不等于进程总 RSS
严格限制在 1 TiB；CPU staging、模型及运行时内存在池外，仍受统一内存预算和背压保护。

Python 优先使用 `/tmp/deepspec_vllm_torchtitan_envs/bin/python`，不存在时使用
`/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python`。
可通过 `DEBUG_PYTHON=/绝对路径/bin/python` 覆盖。脚本设置项目 PYTHONPATH 和 CUDA 12 runtime，
不创建环境、不安装依赖、不使用 uv。

默认模型为 `/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B`；默认输入为
`outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl`。
可传 `--model` / `--source`。预检查要求八张空闲 GPU，检查模型分片与当前 cgroup 内存预算。
真正训练仍由 `deepspec.pipeline.run` 准备输入并按实际样本大小再次检查预算。

无需手动启动 Ray 或 Mooncake。训练使用私有本地 Ray 和自动管理的 Master/FeatureBuffer；
外层复用现有进程 subreaper 清理其后代，并按独有 session 标识核查残留。
不会执行全局 `ray stop` 或按进程名批量结束任务。

## 探针与结果

大池探针启动独立 Master，以及三个分离的 CPU 进程：owner 提供完整配置容量的池，
writer 写入一个与真实 4K 特征形状一致的合成样本，reader 读取六个字段、校验 SHA256，
显式删除全部块并确认对象不存在。探针不使用 GPU，也不将整池填满。

当前训练入口等待 FeatureBuffer 初始化的时间固定为 60 秒。探针测得的 Store setup
若达到 45 秒，脚本会停止并提示先检查启动超时，避免直接进入模型训练。
`--probe-timeout` 不会修改训练入口的这个 60 秒限制。

每轮保存：

- `preflight.json`：环境版本、模型 identity、GPU 清单、4K/128K 合并内存预算。
- `pool-probe/`：大池初始化、独立写/读进程日志、描述符、校验和删除结果。
- `4k-launcher.log` / `128k-launcher.log` / `peak-launcher.log`：训练启动器日志。
- `4k/` / `128k/` / `peak/`：原生流水线全部产物及独立核验的 `verification.json`。
- `memory.jsonl`：阶段运行期间的节点/cgroup 内存余量；池探针有自己的同名记录。
- `debug-result.json`：已完成阶段、失败原因与最终进程清理结果。

独立训练核验包括真实长度、唯一 epoch/sample 身份、逐 rank 消费顺序、SHA256、ACK 后释放、
context 梯度、每个 rank 的更新日志、四个 checkpoint 分片范围、metadata hash，
并通过 DCP 单独读取 `fc.weight` 的 optimizer step。样本数按实际 `steps × 4` 核验，
没有硬编码为 12；普通短测默认一个 epoch，容量压力模式明确使用两个 epoch。

普通短测验证单机功能和大池初始化；只有 `peak` 的实际完成记录可作为近满池证据。
两者都不代表长期稳定性或两机性能结论。

## 当前验证记录（2026-09-17）

- **batch20 / 近满池已完整通过**：`outputs/dspark_single_debug_20260917_batch20_peak_run1/`，
  140 条 epoch 样本（80 个源样本，第一 epoch 80 条、第二 epoch 60 条），35 次更新。
  实际请求批次 `[20,20,20,20,20,20,15,5]`，末两批验证容量背压前后的部分批次准入。
  135 条特征同时驻留，峰值 **1012.763674 GiB / 1024 GiB = 98.902703%**；
  READY/删除事件重算与 Mooncake 自身分配峰值均为 **1,087,446,714,480 字节**。
  峰值时 owner RSS 1012.977 GiB；删除后复用池的其余页，owner RSS 高水位实测
  1024.196 GiB（含少量进程开销）。全程无驱逐、无落盘、无内存压力，最低 headroom 418.976 GiB。
  560 次 rank SHA256 读取、四 rank 梯度、35 次更新和 fc optimizer step=35 均通过；
  140 对象全部删除、最终分配字节 0、GPU 与本轮进程无残留，debug/启动器退出码 0。
  汇总见 `outputs/dspark_single_debug_20260917_batch20_summary.json`。
  本轮另存 `mooncake-allocation.jsonl`、`owner-rss-observation.json`、`pool-occupancy.csv`。
- `outputs/dspark_single_debug_20260917_batch20_4k_run1/`：4K/batch20/5 步完整通过，
  一次提交 20 条请求，80 次 rank 读取，全部删除，fc optimizer step=5，清理无残留。
  与近满池轮使用相同运行源码；26 项不同定向测试、Ruff 通过。
- `outputs/dspark_single_debug_20260917_1tib_run1/pool-probe/`：1 TiB 池初始化
  约 0.21 秒，独立进程完成 251,695,120 字节合成特征读写、SHA256 和 34 块删除。
- `outputs/dspark_single_debug_20260917_1tib_4k_run3/`：4K 三步完整通过，
  启动器退出码 0；12 样本、48 次读取、四 rank 梯度、三次更新、全部对象释放通过。
  独立读取 fc optimizer step=3，四 checkpoint 分片及 metadata hash 通过；无进程残留。
- `outputs/dspark_single_debug_20260917_1tib_128k_run1/`：128K 三步完整通过，
  启动器和 debug 退出码 0；12 个真实 131072-token 样本、48 次读取校验、四 rank
  梯度及三次更新通过，全部对象释放。独立读取 fc optimizer step=3，四分片及
  metadata hash 通过；池预留峰值约 60.016 GiB，无内存压力、GPU/本轮进程残留。
  两轮均为 1 TiB/window8，未测试接近满池的压力或长期稳定性。
- 首轮暴露单机 vLLM EngineCore 未显式连接私有 Ray 的问题，已在 `run.py` 传递
  `RAY_ADDRESS`。单机 GPU 监控现在结合本轮环境标识、父子关系和 PID/启动时间缓存，
  避免退出重托管误报；仍拒绝外部进程和 PID 复用，并记录未归属进程命令。
- `test_pipeline_cluster.py` 的 11 项检查通过；新增监控分支后定向退出/PID 复用检查通过。
  Shell 语法、Ruff、help/dry-run 和带独立会话子进程的超时清理检查通过。
- 新环境第一次运行触发了 FlashInfer SM103a 内核编译；观察到 `ninja/nvcc/ptxas`
  持续工作。首次初始化时间不用于评估训练吞吐。
